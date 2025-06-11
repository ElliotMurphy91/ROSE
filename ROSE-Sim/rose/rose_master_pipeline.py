#!/usr/bin/env python3
"""ROSE‑Sim master pipeline

Analyze naturalistic language‑processing datasets (e.g. patients listening to a
podcast) and extract the key computational/oscillatory signatures predicted by
ROSE.

Usage ───────────────────────────────────────────────────────────────────────
$ python rose_master_pipeline.py \
    --audio path/to/audio.wav \
    --eeg   path/to/iEEG.h5    \
    --montage path/to/elecs.tsv \
    --outdir results/session01  \
    --config config.yml

Pipeline stages
---------------
1.  *Audio ⇒ word‑aligned transcript*  (FFmpeg + Whisper‑cpp)
2.  *Parsing*                          (left_corner_mg; action trace)
3.  *Lexico‑semantic motifs*           (dynamical_motif_rnn)
4.  *Headedness δ–θ–γ PAC*             (delta_theta_gamma_cfc)
5.  *Traveling‑wave metrics*           (wave_field_2d)
6.  *Ephaptic gain control*            (ephaptic_field)
7.  *Visualisations & HDF5 summary*

All heavy lifting is delegated to existing modules in `rose.*`.  This script
just orchestrates I/O, scheduling, and reproducible logging.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

# ── ROSE‑Sim imports ──────────────────────────────────────────────────────
from rose.parsers.left_corner_mg import LeftCornerMGParser
from rose.motifs.dynamical_motif_rnn import MotifRNN
from rose.oscillators.delta_theta_gamma_cfc import compute_pac_headedness
from rose.traveling_waves.wave_field_2d import TravellingWaveAnalyzer
from rose.interfaces.ephaptic_field import EphapticField

# Utility for HDF5 saving
try:
    import h5py
except ImportError as exc:
    raise SystemExit("h5py required: pip install h5py") from exc

# ──────────────────────────────────────────────────────────────────────────

ROOT = pathlib.Path(__file__).resolve().parent


def run_whisper(audio_path: pathlib.Path, model: str = "base.en") -> pathlib.Path:
    """Run whisper‑cpp (local) and return JSON transcript path."""
    out_json = audio_path.with_suffix(".json")
    if out_json.exists():
        logging.info("Whisper transcript exists → %s", out_json.name)
        return out_json
    cmd = [
        "whisper",  # assumes whisper‑cpp in PATH
        str(audio_path),
        "--model", model,
        "--output_format", "json"
    ]
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_json


def load_ieeg(eeg_path: pathlib.Path) -> Tuple[np.ndarray, dict]:
    """Load iEEG data (BrainVision, EDF, HDF5) → (data, info dict)."""
    suffix = eeg_path.suffix.lower()
    if suffix == ".h5":
        with h5py.File(eeg_path, "r") as h5:
            data = h5["data"][:]
            info = json.loads(h5["meta"].asstr()[()])
    else:
        raise NotImplementedError(f"Unsupported iEEG format: {suffix}")
    return data, info


@dataclass
class Config:
    low_freqs: List[float]
    high_freqs: List[float]
    pac_method: str = "modulation_index"
    align_tolerance_ms: int = 30

    @staticmethod
    def from_yaml(path: pathlib.Path) -> "Config":
        import yaml  # local import to keep base deps slim
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return Config(**raw)


# ── Main orchestration ────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ROSE‑Sim naturalistic pipeline")
    parser.add_argument("--audio", type=pathlib.Path, required=True)
    parser.add_argument("--eeg", type=pathlib.Path, required=True)
    parser.add_argument("--montage", type=pathlib.Path, required=True, help="TSV with electrode coords (x,y,z,label)")
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=False)
    parser.add_argument("--whisper_model", default="base.en")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.outdir.mkdir(exist_ok=True, parents=True)

    # ── 1. Transcribe audio ───────────────────────────────────────────────
    transcript_json = run_whisper(args.audio, args.whisper_model)
    with open(transcript_json) as fh:
        whisper_data = json.load(fh)
    words = whisper_data["segments"]  # list of dicts: {"text","start","end"}

    # ── 2. Parse transcript ──────────────────────────────────────────────
    parser_lc = LeftCornerMGParser()
    parse_events: List[dict] = []
    for w in words:
        actions = parser_lc.parse_incremental(w["text"].strip())
        for act in actions:
            parse_events.append({
                "word": w["text"],
                "t_start": w["start"],
                "t_end": w["end"],
                **asdict(act)
            })
    parse_df = pd.DataFrame(parse_events)
    parse_df.to_csv(args.outdir / "parse_trace.csv", index=False)
    logging.info("Saved parse trace with %d events", len(parse_df))

    # ── 3. Load iEEG ─────────────────────────────────────────────────────
    ieeg_data, ieeg_info = load_ieeg(args.eeg)
    sfreq = ieeg_info["sfreq"]

    # ── 4. PAC headedness analysis ───────────────────────────────────────
    cfg = Config.from_yaml(args.config) if args.config else Config(low_freqs=[1,4], high_freqs=[30,80])

    pac_results = compute_pac_headedness(
        data=ieeg_data, sfreq=sfreq,
        low_freqs=cfg.low_freqs, high_freqs=cfg.high_freqs,
        events=parse_df,  # align by t_start/t_end columns
        align_tolerance_ms=cfg.align_tolerance_ms,
        method=cfg.pac_method,
    )
    pac_results.to_csv(args.outdir / "pac_headedness.csv", index=False)

    # ── 5. Dynamical motifs / lexical semantics ──────────────────────────
    motif_model = MotifRNN.load_pretrained()
    motif_states = motif_model.run_transcript([w["text"] for w in words])
    np.save(args.outdir / "motif_states.npy", motif_states)

    # ── 6. Traveling wave metrics (optional) ─────────────────────────────
    tw_analyzer = TravellingWaveAnalyzer(
        montage_path=args.montage,
        data=ieeg_data,
        sfreq=sfreq,
        low_freq_band=cfg.low_freqs,
    )
    tw_metrics = tw_analyzer.compute_metrics()
    tw_metrics.to_csv(args.outdir / "traveling_waves.csv", index=False)

    # ── 7. Ephaptic coupling demo (optional) ─────────────────────────────
    eph = EphapticField(num_neurons=100, field_radius=4.0)
    eph_sim = eph.simulate(duration=2.0, dt=1e-3)
    np.save(args.outdir / "ephaptic_field.npy", eph_sim)

    # ── 8. Save metadata snapshot ────────────────────────────────────────
    meta = {
        "timestamp": datetime.utcnow().isoformat(),
        "args": vars(args),
        "config": asdict(cfg),
        "git_rev": _git_revision(),
        "rose_sim_version": _pkg_version(),
    }
    with open(args.outdir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    logging.info("Analysis complete → %s", args.outdir)


# ── Helpers ───────────────────────────────────────────────────────────────

def _git_revision() -> str | None:
    """Return current git hash if repository exists."""
    try:
        rev = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True)
        return rev.strip()
    except Exception:
        return None


def _pkg_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("rose-sim")
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
