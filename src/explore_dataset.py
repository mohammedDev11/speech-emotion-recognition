"""Reproduce CREMA-D metadata checks, speaker split, and SVG figures.

Usage: python3 src/explore_dataset.py --metadata data/metadata/SentenceFilenames.csv
       python3 src/explore_dataset.py --metadata data/metadata/SentenceFilenames.csv --audio-dir data/raw/AudioWAV
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import struct
import wave
from collections import Counter, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
LABELS = {"ANG": "Angry", "DIS": "Disgust", "FEA": "Fear", "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad"}
SAMPLE_STEMS = [
    "1001_IEO_ANG_MD", "1001_IEO_DIS_MD", "1001_IEO_FEA_MD",
    "1001_IEO_HAP_MD", "1001_IEO_NEU_XX", "1001_IEO_SAD_MD",
]
SEED = 471


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        stem = row["Filename"].strip()
        fields = stem.split("_")
        if len(fields) != 4 or fields[2] not in LABELS or not fields[0].isdigit():
            raise ValueError(f"Unexpected CREMA-D filename: {stem}")
        result.append({"filename": stem + ".wav", "speaker_id": fields[0], "emotion": LABELS[fields[2]]})
    if len(result) != 7442 or len({r["speaker_id"] for r in result}) != 91:
        raise ValueError("Expected the complete official 7,442-file, 91-speaker index")
    if len({r["filename"] for r in result}) != len(result):
        raise ValueError("Duplicate filenames in metadata")
    return result


def assign_splits(rows: list[dict[str, str]]) -> None:
    speakers = sorted({r["speaker_id"] for r in rows})
    random.Random(SEED).shuffle(speakers)
    # Fixed speaker counts: 64 train, 13 validation, 14 test.
    groups = {speaker: ("train" if i < 64 else "validation" if i < 77 else "test") for i, speaker in enumerate(speakers)}
    for row in rows:
        row["split"] = groups[row["speaker_id"]]
    assert len({r["speaker_id"] for r in rows if r["split"] == "train"} &
               {r["speaker_id"] for r in rows if r["split"] == "test"}) == 0
    assert all(len({r["split"] for r in rows if r["speaker_id"] == speaker}) == 1 for speaker in speakers)


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "speaker_id", "emotion", "split"])
        writer.writeheader()
        writer.writerows(rows)


def svg_document(width: int, height: int, body: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'role="img">\n<rect width="100%" height="100%" fill="#fbfaf7"/>\n' + "\n".join(body) + "\n</svg>\n")


def distribution_figure(rows: list[dict[str, str]], path: Path) -> None:
    counts = Counter(r["emotion"] for r in rows)
    body = ['<title>CREMA-D emotion distribution</title>',
            '<text x="38" y="48" font-size="25" font-family="Arial" font-weight="bold" fill="#1d2933">CREMA-D emotion distribution</text>',
            '<text x="38" y="75" font-size="14" font-family="Arial" fill="#52606d">7,442 clips · 91 speakers · intended emotion from official filenames</text>']
    max_count = max(counts.values())
    for i, label in enumerate(LABELS.values()):
        y = 112 + i * 57
        bar_width = round(560 * counts[label] / max_count)
        body.extend([f'<text x="38" y="{y + 21}" font-size="17" font-family="Arial" fill="#1d2933">{label}</text>',
                     f'<rect x="145" y="{y}" width="{bar_width}" height="30" rx="4" fill="#247f87"/>',
                     f'<text x="{157 + bar_width}" y="{y + 21}" font-size="16" font-family="Arial" fill="#1d2933">{counts[label]:,} ({counts[label] / len(rows):.1%})</text>'])
    path.write_text(svg_document(850, 470, body))


def inspect_wav(path: Path) -> tuple[int, float, float, float, list[float]]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError(f"Expected mono 16-bit PCM WAV: {path}")
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    peak = max(abs(x) for x in samples) / 32768
    rms = math.sqrt(sum(x * x for x in samples) / len(samples)) / 32768
    # Max absolute amplitude within each small time bin preserves visible transients.
    bins = 260
    envelope = [max(abs(x) for x in samples[i * len(samples) // bins:(i + 1) * len(samples) // bins]) / 32768 for i in range(bins)]
    return sample_rate, len(samples) / sample_rate, peak, rms, envelope


def sample_figure(audio_dir: Path, figure_path: Path, inspection_path: Path) -> None:
    body = ['<title>Six real CREMA-D speech waveforms</title>',
            '<text x="36" y="42" font-size="25" font-family="Arial" font-weight="bold" fill="#1d2933">Real CREMA-D audio samples</text>',
            '<text x="36" y="67" font-size="14" font-family="Arial" fill="#52606d">One clip per emotion · actor 1001 · IEO sentence · medium intensity where available</text>']
    records = []
    for index, stem in enumerate(SAMPLE_STEMS):
        path = audio_dir / (stem + ".wav")
        rate, duration, peak, rms, envelope = inspect_wav(path)
        label = LABELS[stem.split("_")[2]]
        y = 103 + index * 91
        points = " ".join(f"{255 + i * 2},{y + 25 - 22 * value:.1f}" for i, value in enumerate(envelope))
        body.extend([f'<text x="36" y="{y + 27}" font-size="16" font-family="Arial" fill="#1d2933">{escape(label)}</text>',
                     f'<line x1="255" y1="{y + 25}" x2="775" y2="{y + 25}" stroke="#cbd2d8"/>',
                     f'<polyline points="{points}" fill="none" stroke="#247f87" stroke-width="1.5"/>',
                     f'<text x="790" y="{y + 27}" font-size="13" font-family="Arial" fill="#52606d">{duration:.2f}s</text>'])
        records.append({"filename": path.name, "emotion": label, "sample_rate_hz": rate,
                        "duration_s": f"{duration:.3f}", "peak_amplitude": f"{peak:.4f}", "rms_amplitude": f"{rms:.4f}"})
    figure_path.write_text(svg_document(900, 670, body))
    with inspection_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/metadata/SentenceFilenames.csv")
    parser.add_argument("--audio-dir", type=Path, help="Directory containing six sample WAV files")
    args = parser.parse_args()
    rows = read_metadata(args.metadata)
    assign_splits(rows)
    (ROOT / "data/metadata").mkdir(parents=True, exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    write_manifest(rows, ROOT / "data/metadata/splits.csv")
    distribution_figure(rows, ROOT / "figures/emotion_distribution.svg")
    if args.audio_dir:
        sample_figure(args.audio_dir, ROOT / "figures/sample_waveforms.svg", ROOT / "data/metadata/sample_inspection.csv")
    counts = Counter(r["split"] for r in rows)
    by_split = defaultdict(Counter)
    for row in rows:
        by_split[row["split"]][row["emotion"]] += 1
    print(f"Total: {len(rows)} clips; {len({r['speaker_id'] for r in rows})} speakers")
    for split in ("train", "validation", "test"):
        print(f"{split}: {counts[split]} clips; {dict(by_split[split])}")


if __name__ == "__main__":
    main()
