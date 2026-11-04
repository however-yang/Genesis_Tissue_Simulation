import io,pickle,struct,zlib,tempfile,sys,json
from pathlib import Path
W=Path(__file__).resolve().parent;P=W.parent/'2026-10-31-grasp-cut-validation';sys.path.insert(0,str(P))
from replay_archive import archive_replay,open_replay,DeltaReader,ReplayArchiveWriter
with tempfile.TemporaryDirectory(dir=W,prefix='archive-roundtrip-') as temp:
 root=Path(temp);src=root/'replay.bin';out=io.BytesIO();pickle.dump(dict(test=True),out,protocol=5)
 raw_frames=[b'ABCD'*10000,b'ABCD'*9999+b'xyz1',b'abc'*12001,b'ABCD'*4000,bytes(70000)]
 for i,raw in enumerate(raw_frames):
  payload=pickle.dumps((i,raw),protocol=5);packed=zlib.compress(payload,9 if i==2 else 1);out.write(struct.pack('<I',len(packed))+packed)
 original=out.getvalue();src.write_bytes(original);result=archive_replay(src,root)
 assert not src.exists() and result['frames']==5 and result['exact_encoded_byte_roundtrip']
 with open_replay(src) as f:
  assert pickle.load(f)==dict(test=True)
  for i,raw in enumerate(raw_frames):
   size=struct.unpack('<I',f.read(4))[0];assert pickle.loads(zlib.decompress(f.read(size)))==(i,raw)
  assert f.read()==b''
 with open_replay(src) as f:assert f.read()==original
 direct=root/'direct';direct.mkdir();writer=ReplayArchiveWriter(direct/'replay.rdx');header=pickle.dumps(dict(test=True),protocol=5);pickle.dump(dict(test=True),writer,protocol=5);expected=bytearray(header)
 for i,raw in enumerate(raw_frames):
  payload=pickle.dumps((i,raw),protocol=5);encoded=zlib.compress(payload,1);writer.write_frame(payload);expected.extend(struct.pack('<I',len(encoded))+encoded)
 writer.flush();writer.close();writer.close()
 with open_replay(direct/'replay.bin') as f:assert f.read()==bytes(expected)
 assert json.loads((direct/'replay-archive.json').read_text())['frames']==len(raw_frames)
 broken=root/'broken.rdx';broken.write_bytes(src.with_suffix('.rdx').read_bytes()[:-5])
 try:
  with io.BufferedReader(DeltaReader(broken)) as f:f.read()
 except (EOFError,zlib.error):pass
 else:raise AssertionError('Truncated archive accepted')
 print(json.dumps(result,indent=2));print('Archive exact-byte roundtrip, streaming pickle, changing lengths, non-default zlib, truncation checks passed.')
