"""Read-only review of the named MP4, recorded cuts, and blade geometry."""
import hashlib
import json
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
CASE = ROOT / "endoscopic_tissue/outputs/interaction-fixes-20261031/r13-wall-rim-2"
sys.path.insert(0, str(ROOT))
from endoscopic_tissue.instrument_geometry import load_scissors, SCISSORS_EDGE_SPAN


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    report = json.loads((CASE / "report.json").read_text())
    rows = [json.loads(x) for x in (CASE / "trajectory.jsonl").read_text().splitlines()]
    track = json.loads((CASE / "wall-track.json").read_text())
    cumulative = 0
    stages = []
    for stage in report["metadata"]["stages"]:
        c = stage["command"]
        stages.append(dict(name=stage["name"], first=cumulative + 1,
                           last=cumulative + stage["steps"],
                           blade_center_cm=c["blade_center"],
                           angles_deg=[c["scissors_pitch_deg"], c["scissors_roll_deg"], c["scissors_yaw_deg"]],
                           cutting=c["scissors_cutting"]))
        cumulative += stage["steps"]
    events = []
    count = 0
    for row in rows:
        delta = row["cut_edges"] - count
        if delta:
            events.append(dict(step=row["step"], seconds=row["step"] / 30,
                               stage=row["stage"], new_edges=delta, total_edges=row["cut_edges"]))
        count = row["cut_edges"]
    z = np.load(CASE / "wall-track.npz")
    pose = z["scissors_pose"]
    axes = pose[:, 3:6].astype(float)
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    max_angle = np.degrees(np.arccos(np.clip(axes @ axes[0], -1, 1))).max()

    geo = load_scissors()
    fit = []
    for i, (toward, vertices) in enumerate(zip((-1, 1), geo.jaws.vertices)):
        span = vertices[(vertices[:, 0] >= SCISSORS_EDGE_SPAN[0]) &
                        (vertices[:, 0] <= SCISSORS_EDGE_SPAN[1])]
        slices = np.round(span[:, 0] / .025).astype(int)
        order = np.lexsort((-toward * span[:, 1], slices))
        first = np.r_[True, slices[order][1:] != slices[order][:-1]]
        sharp = span[order][first]
        a, b = geo.edges[i].astype(float)
        t = np.clip((sharp-a) @ (b-a) / np.dot(b-a, b-a), 0, 1)
        dist = np.linalg.norm(sharp-a-t[:, None]*(b-a), axis=1) * 10
        fitted_opening = a[1] + (sharp[:, 0]-a[0])*(b[1]-a[1])/(b[0]-a[0])
        opening_error = np.abs(sharp[:, 1]-fitted_opening) * 10
        fit.append(dict(blade=i, sampled_slices=len(sharp),
                        max_sampled_vertex_to_fitted_segment_mm=float(dist.max()),
                        rms_sampled_vertex_to_fitted_segment_mm=float(np.sqrt(np.mean(dist**2))),
                        max_opening_coordinate_fit_residual_mm=float(opening_error.max()),
                        rms_opening_coordinate_fit_residual_mm=float(np.sqrt(np.mean(opening_error**2))),
                        note="Same per-slice sharp-vertex rule as the loader. The 3D distance includes vertex width offsets from the averaged shear plane. These sampled asset discrepancies are not cutting-path errors or proof of a true physical edge location."))

    movie = CASE / "review/interactions-h264.mp4"
    verified = json.loads((CASE / "review/video-verification.json").read_text())
    movie_sha = digest(movie)
    assert movie_sha == verified["movie_sha256"]
    event_steps = [row["step"] for i, row in enumerate(track) if i and row["fractures"] > track[i-1]["fractures"]]
    selected = set(range(0, 1186, 30)) | {1185, 150, 300, 420, 570, 720, 840, 990, 1110}
    for event in event_steps:
        selected.update(range(event-10, event+11, 5))
    reader = imageio_ffmpeg.read_frames(str(movie), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata["size"]
    images = {}
    decoded = 0
    for frame_id, raw in enumerate(reader):
        decoded += 1
        if frame_id in selected:
            images[frame_id] = Image.frombytes("RGB", (width, height), raw)
    assert decoded == verified["expected_frames"] == 1186
    assert set(images) == selected
    directory = WORK / "frames"
    directory.mkdir(exist_ok=True)
    sheet_paths = []
    ordered = sorted(images)
    for sheet_number, offset in enumerate(range(0, len(ordered), 12), 1):
        sheet = Image.new("RGB", (1280, 1392), "#101010")
        draw = ImageDraw.Draw(sheet)
        for cell, frame_id in enumerate(ordered[offset:offset+12]):
            x, y = (cell % 3)*426, (cell // 3)*348
            frame = images[frame_id].resize((426, 320))
            sheet.paste(frame, (x, y+28))
            draw.text((x+7, y+7), f"{frame_id / 30:.3f}s | step {frame_id}", fill="white")
        path = directory / f"mp4-review-{sheet_number:02}.jpg"
        sheet.save(path, quality=94)
        sheet_paths.append(path.relative_to(WORK).as_posix())
    detail_steps = [300, 570, 720, 840, 990, 1110]
    detail = Image.new("RGB", (1240, 1350), "#101010")
    draw = ImageDraw.Draw(detail)
    for cell, frame_id in enumerate(detail_steps):
        x, y = (cell % 2)*620, (cell // 2)*450
        crop = images[frame_id].crop((500, 300, 1120, 720))
        detail.paste(crop, (x, y+30))
        draw.text((x+8, y+8), f"{frame_id / 30:.2f}s | step {frame_id} | original MP4 crop", fill="white")
    detail.save(directory / "cut-detail.jpg", quality=96)
    result = dict(case=str(CASE), movie=str(movie), movie_sha256=movie_sha,
                  decoded_frames=decoded, fps=metadata["fps"],
                  duration_s=decoded/metadata["fps"], sampled_frames=ordered,
                  sheets=sheet_paths, source_hashes={
                      f: digest(CASE/f) for f in ["report.json", "trajectory.jsonl", "wall-track.npz"]},
                  stages=stages, edge_change_events=events,
                  total_cut_edges=count, fracture_steps=event_steps,
                  solved_axis_max_angle_from_first_deg=float(max_angle),
                  blade_geometry=fit,
                  grid_spacing_mm=[10*x for x in report["metadata"]["cell_size_cm"]],
                  original_acceptance_passed=report["original_metrics"]["acceptance_passed"],
                  max_blade_overlap_mm=report["max_blade_overlap_mm"],
                  cuts_while_disabled=report["cuts_while_disabled"],
                  cuts_before_any_fracture=report["cuts_before_any_fracture"],
                  healed_connections=report["healed_connections"],
                  max_invalid_tets=report["max_invalid_tets"],
                  note="Offline inspection only; no solver run, parameter change, or performance claim.")
    (WORK / "video-cut-edge-audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["decoded_frames", "movie_sha256", "total_cut_edges", "edge_change_events", "solved_axis_max_angle_from_first_deg", "blade_geometry", "max_blade_overlap_mm"]}, indent=2))


if __name__ == "__main__":
    main()
