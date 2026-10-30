"""Compose the stage stills of headless render runs into one comparison image: one row per run, one column per stage,
each still cropped to the jaws and the tumor.

Usage: python ab_montage.py <output png> <label>=<run folder> [<label>=<run folder> ...]
"""
import sys

from PIL import Image, ImageDraw, ImageFont

STAGES = ["close", "lift", "blade_approach", "incise_2", "hold"]
output, *specs = sys.argv[1:]
rows = []
for spec in specs:
    label, folder = spec.split("=", 1)
    stills = [Image.open(f"{folder}/{stage}.png") for stage in STAGES]
    width, height = stills[0].size
    box = (int(width * .28), 0, int(width * .72), int(height * .55))
    rows.append((label, [still.crop(box) for still in stills]))
cell_width, cell_height = rows[0][1][0].size
montage = Image.new("RGB", (cell_width * len(STAGES), (cell_height + 28) * len(rows)), "black")
draw = ImageDraw.Draw(montage)
font = ImageFont.truetype("arial.ttf", 22)
for r, (label, crops) in enumerate(rows):
    for c, crop in enumerate(crops):
        montage.paste(crop, (c * cell_width, r * (cell_height + 28) + 28))
        draw.text((c * cell_width + 6, r * (cell_height + 28) + 3), f"{label} - {STAGES[c]}" if c == 0 else STAGES[c],
                  fill="white", font=font)
montage.resize((montage.width * 3 // 5, montage.height * 3 // 5)).save(output)
