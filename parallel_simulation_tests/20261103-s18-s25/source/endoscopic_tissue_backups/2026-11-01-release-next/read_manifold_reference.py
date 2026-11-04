"""Local method check for the observed native boundary topology; no downloads."""
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parents[2]
source, = (ROOT/'参考文献').rglob('*Manifold_Dual_Contouring.pdf')
destination = Path(__file__).resolve().parent/'manifold-reference'
destination.mkdir(exist_ok=True)
with fitz.open(source) as pdf:
    for i, page in enumerate(pdf):
        text = page.get_text()
        (destination/f'page-{i+1:02d}.txt').write_text(text, encoding='utf8')
        if i < 4:
            page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4)).save(destination/f'page-{i+1:02d}.png')
    print(source, 'pages', len(pdf))
