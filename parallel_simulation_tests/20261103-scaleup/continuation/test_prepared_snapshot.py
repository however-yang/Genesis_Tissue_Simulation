import dataclasses,hashlib,json,tempfile,unittest
from pathlib import Path
import zstandard as zstd
from endoscopic_tissue.training.snapshot import EnvSnapshot
from prepared_snapshot import PreparedSnapshot

class PreparedTests(unittest.TestCase):
 def fixture(self):
  raw=bytes(range(256))*128;py=b'python state placeholder';nc=zstd.ZstdCompressor().compress(raw);pc=zstd.ZstdCompressor().compress(py)
  meta=dict(schema='environment_snapshot_v1',state_abi=1,native_bytes=len(raw),python_bytes=len(py),native_compressed_sha256=hashlib.sha256(nc).hexdigest(),python_compressed_sha256=hashlib.sha256(pc).hexdigest())
  return raw,EnvSnapshot(nc,pc,json.dumps(meta).encode())
 def test_budget_rejected_before_decompression(self):
  _,s=self.fixture()
  with self.assertRaises(MemoryError):PreparedSnapshot(s,max_host_bytes=1)
 def test_hot_cache_is_immutable_and_portable_bytes_unchanged(self):
  raw,s=self.fixture();p=PreparedSnapshot(s,max_host_bytes=100000)
  self.assertEqual(p.native_bytes(),raw);self.assertIs(p.native_bytes(),p.native_bytes());self.assertEqual(p.sha256,s.sha256)
  with self.assertRaises(dataclasses.FrozenInstanceError):p._native=b'corrupt'
  with self.assertRaises(dataclasses.FrozenInstanceError):p.native_compressed=b'corrupt'
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
   a,b=Path(d)/'a',Path(d)/'b';s.save(a);p.save(b);self.assertEqual(a.read_bytes(),b.read_bytes())
 def test_invalid_frame_size_still_rejected(self):
  _,s=self.fixture();meta=s.metadata;meta['native_bytes']+=1
  bad=EnvSnapshot(s.native_compressed,s.python_compressed,json.dumps(meta).encode())
  with self.assertRaises(ValueError):PreparedSnapshot(bad,max_host_bytes=100000)
 def test_explicit_budget_required(self):
  _,s=self.fixture()
  for v in [0,-1,True,1.5]:
   with self.assertRaises(ValueError):PreparedSnapshot(s,max_host_bytes=v)
if __name__=='__main__':unittest.main()
