"""Compatibility entry for the conservative isolated configuration."""
import os,sys
from pathlib import Path
args=['--physics' if x=='--checked-physics' else x for x in sys.argv[1:]]
os.execv(sys.executable,[sys.executable,str(Path(__file__).with_name('reproduce.py')),'--variant','conservative']+args)
