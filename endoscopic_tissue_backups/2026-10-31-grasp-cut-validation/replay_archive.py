"""Lossless frame-delta storage for local, trusted simulation replay records.

The reader reconstructs the exact original replay.bin byte stream. Archival
deletes only that generated source after a complete, independent hash check.
"""
import hashlib
import io
import json
from pathlib import Path
import pickle
import struct
import zlib

import numpy as np
try:
    import zstandard as zstd
except ImportError:
    zstd = None

MAGIC = b'RDX1'
RECORD = struct.Struct('<BII')


def frame_dictionary(previous):
    return zstd.ZstdCompressionDict(previous, dict_type=zstd.DICT_TYPE_RAWCONTENT) if previous else None


def exact(stream, count):
    data = stream.read(count)
    if len(data) != count:
        raise EOFError('Truncated replay archive')
    return data


def delta(data, previous):
    result = np.frombuffer(data, dtype=np.uint8).copy()
    count = min(len(data), len(previous))
    result[:count] ^= np.frombuffer(previous, dtype=np.uint8, count=count)
    return result.tobytes()


def encode_frame(raw, encoded, previous):
    if zstd is None:
        mode = 1
        alternative = zlib.compress(delta(raw, previous), 1)
    else:
        mode = 2
        parameters = zstd.ZstdCompressionParameters.from_level(
            3, window_log=24, hash_log=20, chain_log=20)
        alternative = zstd.ZstdCompressor(
            compression_params=parameters,
            dict_data=frame_dictionary(previous)).compress(raw)
    if len(alternative) < len(encoded) and zlib.compress(raw, 1) == encoded:
        return mode, alternative
    return 0, encoded


class DeltaReader(io.RawIOBase):
    def __init__(self, path):
        super().__init__()
        self.source = Path(path).open('rb')
        if exact(self.source, 4) != MAGIC:
            self.source.close()
            raise ValueError('Unknown replay archive format')
        self.pending = exact(self.source, struct.unpack('<I', exact(self.source, 4))[0])
        self.offset = 0
        self.previous = b''

    def readable(self):
        return True

    def readinto(self, target):
        if self.offset == len(self.pending):
            header = self.source.read(RECORD.size)
            if not header:
                return 0
            if len(header) != RECORD.size:
                raise EOFError('Truncated replay archive header')
            mode, original_length, stored_length = RECORD.unpack(header)
            data = exact(self.source, stored_length)
            if mode == 0:
                raw, encoded = zlib.decompress(data), data
            elif mode == 1:
                raw = delta(zlib.decompress(data), self.previous)
                encoded = zlib.compress(raw, 1)
            elif mode == 2:
                if zstd is None:
                    raise ImportError('Reading this lossless replay archive requires zstandard')
                raw = zstd.ZstdDecompressor(dict_data=frame_dictionary(self.previous)).decompress(data)
                encoded = zlib.compress(raw, 1)
            else:
                raise ValueError('Unknown replay archive record mode')
            if len(encoded) != original_length:
                raise ValueError('Replay reconstruction length mismatch')
            self.previous = raw
            self.pending = struct.pack('<I', original_length) + encoded
            self.offset = 0
        count = min(len(target), len(self.pending) - self.offset)
        target[:count] = self.pending[self.offset:self.offset + count]
        self.offset += count
        return count

    def close(self):
        if not self.closed:
            self.source.close()
        super().close()


def open_replay(path):
    path = Path(path)
    if path.exists():
        return path.open('rb')
    return io.BufferedReader(DeltaReader(path.with_suffix('.rdx')), buffer_size=1024 * 1024)


class ReplayArchiveWriter:
    """Optional diagnostic capture; changes storage only, preserving every frame."""
    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open('xb')
        self.header = bytearray()
        self.previous = b''
        self.digest = hashlib.sha256()
        self.original_bytes = self.frames = self.delta_frames = 0
        self.started = self.closed = False

    def write(self, data):
        if self.started:
            raise RuntimeError('Only the initial pickle header may use write()')
        self.header.extend(data)
        return len(data)

    def start(self):
        if not self.started:
            self.stream.write(MAGIC + struct.pack('<I', len(self.header)) + self.header)
            self.digest.update(self.header)
            self.original_bytes += len(self.header)
            self.started = True

    def write_frame(self, raw):
        self.start()
        encoded = zlib.compress(raw, 1)
        mode, stored = encode_frame(raw, encoded, self.previous)
        self.stream.write(RECORD.pack(mode, len(encoded), len(stored)))
        self.stream.write(stored)
        self.digest.update(struct.pack('<I', len(encoded)) + encoded)
        self.original_bytes += 4 + len(encoded)
        self.previous = raw
        self.frames += 1
        self.delta_frames += int(mode != 0)

    def flush(self):
        self.stream.flush()

    def close(self):
        if self.closed:
            return
        self.start()
        self.stream.close()
        self.closed = True
        recovered, recovered_size = hashlib.sha256(), 0
        with io.BufferedReader(DeltaReader(self.path)) as restored:
            for block in iter(lambda: restored.read(8 * 1024 * 1024), b''):
                recovered.update(block)
                recovered_size += len(block)
        if recovered.digest() != self.digest.digest() or recovered_size != self.original_bytes:
            raise RuntimeError('Direct replay archive verification failed; archive retained')
        result = dict(archive=str(self.path), source_materialized=False,
                      frames=self.frames, delta_frames=self.delta_frames,
                      original_bytes=self.original_bytes, archive_bytes=self.path.stat().st_size,
                      original_sha256=self.digest.hexdigest(), recovered_sha256=recovered.hexdigest(),
                      reconstructed_bytes=recovered_size, exact_encoded_byte_roundtrip=True)
        (self.path.parent/'replay-archive.json').write_text(json.dumps(result, indent=2), encoding='utf8')


def archive_replay(path, allowed_root):
    path, allowed_root = Path(path).resolve(), Path(allowed_root).resolve()
    if path.name != 'replay.bin' or not path.is_relative_to(allowed_root):
        raise ValueError('Only generated replay.bin files within the explicit output root may be archived')
    destination = path.with_suffix('.rdx')
    temporary = path.with_suffix('.rdx.partial')
    if destination.exists() or temporary.exists():
        raise FileExistsError('An existing archive is preserved')
    original_hash = hashlib.sha256()
    frames, delta_frames = 0, 0
    original_size = path.stat().st_size
    with path.open('rb') as source, temporary.open('xb') as output:
        # Only our own trusted replay header is unpickled to locate its boundary.
        pickle.load(source)
        header_length = source.tell()
        source.seek(0)
        header = exact(source, header_length)
        original_hash.update(header)
        output.write(MAGIC + struct.pack('<I', header_length) + header)
        previous = b''
        while prefix := source.read(4):
            if len(prefix) != 4:
                raise EOFError('Truncated original replay')
            length = struct.unpack('<I', prefix)[0]
            encoded = exact(source, length)
            original_hash.update(prefix + encoded)
            raw = zlib.decompress(encoded)
            # Retain the original encoded record whenever recompression would
            # differ, including files from another zlib release.
            mode, stored = encode_frame(raw, encoded, previous)
            output.write(RECORD.pack(mode, length, len(stored)))
            output.write(stored)
            previous = raw
            frames += 1
            delta_frames += int(mode != 0)
    recovered_hash = hashlib.sha256()
    recovered_size = 0
    with io.BufferedReader(DeltaReader(temporary)) as restored:
        for block in iter(lambda: restored.read(8 * 1024 * 1024), b''):
            recovered_hash.update(block)
            recovered_size += len(block)
    if recovered_hash.digest() != original_hash.digest() or recovered_size != original_size:
        raise RuntimeError('Original replay retained: full archive reconstruction failed')
    result = dict(source=str(path), archive=str(destination), frames=frames,
                  delta_frames=delta_frames, original_bytes=original_size,
                  archive_bytes=temporary.stat().st_size,
                  original_sha256=original_hash.hexdigest(),
                  recovered_sha256=recovered_hash.hexdigest(),
                  reconstructed_bytes=recovered_size,
                  exact_encoded_byte_roundtrip=True)
    temporary.replace(destination)
    (path.parent/'replay-archive.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    # The complete archive has already been read back and hash-verified.
    # This is a single generated file, never a recursive deletion.
    path.unlink()
    return result
