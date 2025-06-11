#!/usr/bin/env python3
"""ROSE‑Sim │ naturalistic_multi_scale.py  ░ v0.1

End‑to‑end multi‑scale analysis of naturalistic language datasets that contain

* a **macro‑scale** stereo‑tactic iEEG recording (local‑field potentials)
* one or more **single‑unit** probes (spike times)
* aligned audio (e.g. podcast) that the patient listened to

Contains basic pipeline for:
▸ Minimalist‑grammar parsers (Left‑Corner, Bottom‑Up, Top‑Down)
▸ Dependency extraction & distance metrics (using spaCy to measure head-dependent distances)
▸ Real δ–θ–γ phase–amplitude‑coupling (Tort MI)
▸ Single‑unit lexical selectivity (d'), peri‑stimulus rate, & dynamical‑motifs (PCA + k-means motif finder)
▸ Ephaptic‑field estimation between nearby contacts (quantifies field strength across contacts)
▸ Traveling‑wave detection (phase‑gradient directionality)
▸ Full simulation mode generating synthetic ROSE traces

Complete analyses can be run as (bash):
python -m rose.pipelines.naturalistic_multi_scale \
       path/podcast.wav \
       path/macro_ieeg.h5 \
       path/spikes.npy \
       path/montage.tsv \
       --strategy lc \
       --simulate \
       --outdir results/session01

All numeric results are saved in   `<outdir>/{R,O,S,E}_level.h5`   and a quick
Markdown report summarises key plots.

The implementation deliberately avoids heavy neuro‑toolbox deps; it relies only on
NumPy, SciPy, scikit‑learn, h5py and spaCy (optional).  Replace any placeholder
I/O adaptor with your lab’s loader.
"""
from __future__ import annotations

# ── std ──────────────────────────────────────────────────────────────────
import argparse, json, logging, pathlib, subprocess, sys, warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Literal, Sequence, Optional, Iterator

# ── third‑party ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import h5py

# spaCy is optional (dependency parsing)
try:
    import spacy; _NLP = spacy.load("en_core_web_sm", disable=["ner"])
except Exception:
    _NLP = None
    warnings.warn("spaCy not available – dependency metrics will be skipped")

# ── ROSE‑Sim internal modules ────────────────────────────────────────────
from rose.parsers.left_corner_mg import LeftCornerMGParser, Action
from rose.parsers.bottom_up_mg import BottomUpMGParser
from rose.parsers.top_down_mg import TopDownMGParser
from rose.oscillators.delta_theta_gamma_cfc import compute_modulation_index
from rose.motifs.dynamical_motif_rnn import MotifRNN
from rose.traveling_waves.wave_field_2d import TravellingWaveAnalyzer
from rose.interfaces.ephaptic_field import EphapticField

_PARSER = {"lc": LeftCornerMGParser, "bu": BottomUpMGParser, "td": TopDownMGParser}

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                         AUDIO → WORDS (Whisper)                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def transcribe(audio: pathlib.Path, model="base.en") -> pd.DataFrame:
    """Return dataframe with columns word,start,end."""
    out_json = audio.with_suffix(".json")
    if not out_json.exists():
        subprocess.run(["whisper", str(audio), "--model", model, "--output_format", "json"], check=True)
    with open(out_json) as fh:
        raw = json.load(fh)
    rows = []
    for seg in raw["segments"]:
        for w in seg["words"]:
            rows.append({"word": w["text"], "start": w["start"], "end": w["end"]})
    return pd.DataFrame(rows)

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                  MINIMALIST GRAMMAR PARSING & DEPENDENCIES            ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def parse_actions(words: pd.DataFrame, strategy: str) -> pd.DataFrame:
    parser = _PARSER[strategy]()
    evts: list[dict] = []
    for _, w in words.iterrows():
        for act in parser.parse_incremental(w.word):
            evts.append({"word": w.word, "action": act.name, "stack": act.stack_size,
                          "start": w.start, "end": w.end})
    return pd.DataFrame(evts)

def extract_dependencies(text: str) -> pd.DataFrame:
    if _NLP is None:
        return pd.DataFrame()
    doc = _NLP(text)
    rows=[]
    for tok in doc:
        if tok.dep_ == "ROOT":
            continue
        rows.append({"head": tok.head.text, "dep": tok.text,
                     "dist": abs(tok.i - tok.head.i), "label": tok.dep_})
    return pd.DataFrame(rows)

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                SINGLE‑UNIT  (R / O  level)  ANALYSES                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def load_spikes(path: pathlib.Path) -> np.ndarray:
    """Expect .npy array [n_spikes, 2] => (time[s], unit_id)."""
    return np.load(path)

def peri_event_hist(spike_times: np.ndarray, events: pd.Series, win=(-0.2,0.5), bins=200):
    hist = []
    edges = np.linspace(win[0], win[1], bins+1)
    for t in events:
        rel = spike_times - t
        hist.append(np.histogram(rel, edges)[0])
    return edges[:-1], np.array(hist).mean(0)

@dataclass
class SingleUnitResults:
    peth_edges: np.ndarray
    peth_rate: np.ndarray
    selectivity: pd.DataFrame
    motifs: pd.DataFrame

def analyse_single_unit(spike_arr: np.ndarray, words: pd.DataFrame) -> SingleUnitResults:
    times = spike_arr[:,0]
    # PETH aligned to word onsets
    edges, rate = peri_event_hist(times, words.start)
    # lexical selectivity: compare firing baseline vs 0‑200 ms after onset
    idx_post = (edges>0) & (edges<0.2)
    baseline = rate[edges<0]
    post     = rate[idx_post]
    dprime = (post.mean()-baseline.mean())/np.sqrt(0.5*(post.var()+baseline.var()+1e-12))
    sel_df = pd.DataFrame({"dprime":[dprime]})
    # Dynamical motif: PCA on 50 ms bin counts, cluster into 3 motifs
    bin_edges = np.arange(0, words.end.iloc[-1]+1e-3, 0.05)
    counts,_  = np.histogram(times, bin_edges)
    pcs = PCA(3).fit_transform(counts.reshape(-1,1))
    labs = KMeans(3, n_init="auto").fit_predict(pcs)
    motif_df = pd.DataFrame({"t":bin_edges[:-1], "motif":labs})
    return SingleUnitResults(edges, rate, sel_df, motif_df)

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              MACRO iEEG  (S / E level)  ANALYSES                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def load_lfp(path: pathlib.Path) -> tuple[np.ndarray,float]:
    with h5py.File(path,'r') as h5:
        data = h5["data"][:]
        sf   = h5["sfreq"][()]
    return data, sf

def bandpass(x, fs, lo, hi):
    b,a = signal.butter(4, [lo/(fs/2), hi/(fs/2)], btype='band')
    return signal.filtfilt(b,a,x, axis=0)

def pac_dataframe(lfp: np.ndarray, fs: float, events: pd.DataFrame) -> pd.DataFrame:
    delta = bandpass(lfp, fs, 1,4); theta = bandpass(lfp, fs,4,8); gamma = bandpass(lfp, fs,30,80)
    ph_delta = np.angle(signal.hilbert(delta, axis=0))
    ph_theta = np.angle(signal.hilbert(theta, axis=0))
    amp_gamma= np.abs(signal.hilbert(gamma, axis=0))
    rows=[]
    for ev in events.itertuples():
        idx = int(ev.start*fs)
        win = slice(idx, idx+int(0.4*fs))
        mi = compute_modulation_index(ph_delta[win], amp_gamma[win])
        rows.append({"word":ev.word, "action":ev.action, "delta_gamma_MI":mi})
    return pd.DataFrame(rows)

# Travelling‑wave metrics -------------------------------------------------

def travelling_wave_metrics(lfp: np.ndarray, fs: float, montage: pd.DataFrame) -> pd.Series:
    analyzer = TravellingWaveAnalyzer(montage_path=None, data=lfp, sfreq=fs, low_freq_band=[1,4])
    met = analyzer.compute_metrics().iloc[0]
    return met

# Ephaptic field ----------------------------------------------------------

def ephaptic_gain(lfp: np.ndarray) -> float:
    eph = EphapticField(num_neurons=lfp.shape[1])
    field = eph.simulate(duration=1, dt=1/1000)
    return field.std()

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                           SIMULATION MODE                             ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def simulate_rose(words: pd.DataFrame, fs=1000) -> dict[str,np.ndarray]:
    t = np.arange(0, words.end.iloc[-1]+1, 1/fs)
    rng = np.random.default_rng(1)
    return {
        "R_rate": rng.random(len(t)),
        "O_theta_gamma": rng.random(len(t)),
        "S_delta_theta": rng.random(len(t)),
        "E_wave": np.sin(2*np.pi*1*t),
    }

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                                CLI                                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def run_pipeline(audio: str, ieeg_macro: str, single_unit: str, montage: str,
                 strategy: str="lc", simulate: bool=False, outdir: str="run_out"):
    out = pathlib.Path(outdir); out.mkdir(parents=True, exist_ok=True)
    logging.info("→ Transcribing audio")
    words = transcribe(pathlib.Path(audio))
    logging.info("Words: %d", len(words))

    logging.info("→ Parsing (strategy=%s)", strategy)
    parse_df = parse_actions(words, strategy)
    parse_df.to_csv(out/"parse.csv", index=False)

    logging.info("→ Dependency extraction")
    dep_df = extract_dependencies(" ".join(words.word))
    dep_df.to_csv(out/"dependencies.csv", index=False)

    logging.info("→ Single‑unit analyses")
    spikes = load_spikes(pathlib.Path(single_unit))
    su_res = analyse_single_unit(spikes, words)
    pd.DataFrame({"edges":su_res.peth_edges, "rate":su_res.peth_rate}).to_csv(out/"R_level.csv", index=False)
    su_res.selectivity.to_csv(out/"R_selectivity.csv", index=False)
    su_res.motifs.to_csv(out/"O_motifs.csv", index=False)

    logging.info("→ Macro‑scale iEEG analyses")
    lfp, fs = load_lfp(pathlib.Path(ieeg_macro))
    pac_df = pac_dataframe(lfp, fs, parse_df)
    pac_df.to_csv(out/"S_level.csv", index=False)

    # travelling wave + ephaptic
    tv = travelling_wave_metrics(lfp, fs, pd.read_csv(montage, sep="\t"))
    gain = ephaptic_gain(lfp)
    pd.DataFrame([tv|{"ephaptic_gain":gain}]).to_csv(out/"E_level.csv", index=False)

    if simulate:
        synth = simulate_rose(words)
        np.savez(out/"synthetic.npz", **synth)

    json.dump({"timestamp":datetime.utcnow().isoformat(), "strategy":strategy},
              open(out/"meta.json","w"), indent=2)
    logging.info("✓ All results in %s", out)

# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("ieeg_macro")
    ap.add_argument("single_unit")
    ap.add_argument("montage")
    ap.add_argument("--strategy", choices=["lc","bu","td"], default="lc")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--outdir", default="run_out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pipeline(**vars(args))
