# Copyright (C) 2026 Fossett Laboratory, Washington University in St. Louis
#
# This file is part of gnss-nlos-notebook and is licensed under the GNU General
# Public License v3.0 or later. See the LICENSE file at the repository root.
#
# The feature definitions here match the KLTDataset labelling convention
# (https://github.com/ebhrz/KLTDataset, GPL-3.0; Hu, Wen & Hsu 2023, IEEE ITSC),
# which is why this work inherits GPL-3.0.

"""
Extract per-(epoch, satellite) LOS/NLOS training features from KLTDataset
runs.

For each collection run we read the rover RINEX observation file plus the
HKSC reference-station RINEX files (which carry the broadcast ephemeris),
run a single-point-positioning (SPP) solve per epoch with RTKLIB (via
pyrtklib), and read off the per-satellite quantities RTKLIB computes:
elevation/azimuth, pseudorange residual, and validity. We combine those
with the raw C/N0 from the observation record and two derived time-series
features, then attach the LOS/NLOS label from the run's nlos.pkl.

Features per (epoch, satellite):
    cn0                 carrier-to-noise density C/N0 (dB-Hz), from RINEX
    elev_deg            satellite elevation angle (degrees), from SPP geometry
    pr_resid_norm       pseudorange post-fit residual normalised by the
                        epoch's robust residual scale (dimensionless)
    pr_rate_consistency |measured range rate - Doppler-predicted range rate|
                        between consecutive epochs (m/s); NaN at arc start
    cn0_std             rolling standard deviation of C/N0 over a short
                        window (dB-Hz), trailing, min_periods=1

Label:
    nlos = 1 if the satellite is in the epoch's all-satellite set but not
    its LOS set; 0 if in the LOS set. Satellites not present in the label
    (unlabelled constellations) are dropped. Only GPS ('G') and BeiDou
    ('C') are labelled in KLTDataset, so only those are kept.

The feature-computation functions here are imported by both train-time
extraction (with labels) and inference (classify.py, no labels), so the two
paths compute features identically.
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyrtklib as prl

# The KLTDataset nlos.pkl files come from an external source, so we never
# deserialise them with a bare pickle.load (arbitrary-code-execution risk).
# The label pickles contain only numpy arrays and basic containers; this
# restricted unpickler permits exactly the numpy reconstruction primitives
# those need and rejects anything else, which neutralises the RCE vector.
_SAFE_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in _SAFE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"blocked non-whitelisted global {module}.{name} in label pickle"
        )


def safe_load_labels(path: Path):
    with open(path, "rb") as fh:
        return _SafeUnpickler(io.BytesIO(fh.read())).load()

# GPS L1 C/A and BeiDou B1I carrier frequencies (Hz). Used to convert the
# RINEX Doppler (Hz) to a range rate (m/s) for the consistency feature.
FREQ_L1 = 1575.42e6      # GPS L1
FREQ_B1I = 1561.098e6    # BeiDou B1I
CLIGHT = prl.CLIGHT

KEEP_SYS = {"G", "C"}    # constellations labelled in KLTDataset
FEATURES = ["cn0", "elev_deg", "pr_resid_norm", "pr_rate_consistency", "cn0_std"]


def sat_name(sat_no: int) -> str:
    buf = prl.Arr1Dchar(4)
    prl.satno2id(sat_no, buf)
    return buf.ptr


def _next_epoch_span(obs: "prl.obs_t", i: int) -> int:
    """Number of consecutive records at obs.data[i]'s epoch (matches
    rtk_util.nextobsf: same time within 0.05 s)."""
    n = 0
    while i + n < obs.n:
        if abs(prl.timediff(obs.data[i + n].time, obs.data[i].time)) > 0.05:
            break
        n += 1
    return n


def epoch_slices(obs: "prl.obs_t") -> list[tuple[int, int]]:
    slices, i = [], 0
    while i < obs.n:
        m = _next_epoch_span(obs, i)
        if m == 0:
            break
        slices.append((i, m))
        i += m
    return slices


def make_prcopt() -> "prl.prcopt_t":
    opt = prl.prcopt_default
    opt.mode = prl.PMODE_SINGLE
    opt.navsys = prl.SYS_GPS | prl.SYS_CMP
    opt.nf = 1
    opt.elmin = 0.0                    # keep low-elevation sats; we label them
    opt.ionoopt = prl.IONOOPT_BRDC
    opt.tropopt = prl.TROPOPT_SAAS
    opt.sateph = prl.EPHOPT_BRDC
    return opt


def rover_epoch_obs(obs: "prl.obs_t", i: int, m: int) -> "prl.obs_t":
    """Build an obs_t holding only the rover (rcv==1) records of one epoch."""
    tmp = prl.obs_t()
    tmp.data = prl.Arr1Dobsd_t(m)
    k = 0
    for j in range(m):
        if obs.data[i + j].rcv == 1:
            tmp.data[k] = obs.data[i + j]
            k += 1
    tmp.n = k
    tmp.nmax = k
    return tmp


def read_rinex(rover_obs: Path, nav_files: list[Path]) -> tuple["prl.obs_t", "prl.nav_t"]:
    obs = prl.obs_t()
    nav = prl.nav_t()
    sta = prl.sta_t()
    prl.readrnx(str(rover_obs), 1, "", obs, nav, sta)
    for f in nav_files:
        prl.readrnx(str(f), 2, "", obs, nav, sta)
    prl.sortobs(obs)
    return obs, nav


def epoch_utc(t: "prl.gtime_t") -> float:
    return t.time + t.sec - 18.0    # GPS time -> UTC (18 leap seconds)


def raw_epoch_rows(
    obs: "prl.obs_t",
    nav: "prl.nav_t",
    prcopt: "prl.prcopt_t",
    label_lookup: dict[int, tuple[set, set]] | None = None,
    epoch_index: int | None = None,
) -> list[dict]:
    """Per-satellite SPP-derived rows for the epochs of ``obs``.

    If ``label_lookup`` maps epoch-index -> (all_sat0, los_sat0) sets, only
    labelled GPS/BeiDou satellites are kept and an ``nlos`` column is set.
    Otherwise every GPS/BeiDou satellite is kept with ``nlos = NaN`` (used
    at inference time).
    """
    rows: list[dict] = []
    for e, (i, m) in enumerate(epoch_slices(obs)):
        utc = epoch_utc(obs.data[i].time)
        lab = None
        if label_lookup is not None:
            if e not in label_lookup:
                continue
            lab = label_lookup[e]

        o = rover_epoch_obs(obs, i, m)
        if o.n < 4:
            continue

        sol = prl.sol_t()
        sol.time = o.data[0].time
        ssat = prl.Arr1Dssat_t(prl.MAXSAT)
        azel = prl.Arr1Ddouble(o.n * 2)
        msg = prl.Arr1Dchar(128)
        prl.pntpos(o.data.ptr, o.n, nav, prcopt, sol, azel, ssat.ptr, msg)

        for k in range(o.n):
            obsd = o.data[k]
            sat = obsd.sat
            sat0 = sat - 1
            name = sat_name(sat)
            sys_char = name[0]
            if sys_char not in KEEP_SYS:
                continue
            if lab is not None:
                all_idx, los_idx = lab
                if sat0 not in all_idx:
                    continue
                nlos = 0 if sat0 in los_idx else 1
            else:
                nlos = np.nan

            st = ssat[sat0]
            elev_deg = float(st.azel[1]) * 180.0 / np.pi
            # Use the overall valid-satellite flag `vs` (set when the sat was
            # used in the SPP solution), NOT the per-frequency `vsat[NFREQ]`
            # array, which RTKLIB leaves 0 in this single-frequency pntpos
            # path even though the post-fit residual `resp[0]` is populated.
            vsat = int(st.vs)
            resp = float(st.resp[0])
            cn0 = float(obsd.SNR[0]) / 1000.0     # RINEX SNR stored x1000 dB-Hz
            if cn0 <= 0:
                cn0 = max(float(st.snr[0]) * 0.25, 0.0)  # ssat.snr in 0.25 dB-Hz
            freq = FREQ_L1 if sys_char == "G" else FREQ_B1I

            rows.append({
                "epoch": e if epoch_index is None else epoch_index,
                "utc": utc,
                "sat": name,
                "sys": sys_char,
                "elev_deg": elev_deg,
                "cn0": cn0,
                "pr_resid": resp if vsat else np.nan,
                "pseudorange": float(obsd.P[0]) if obsd.P[0] != 0 else np.nan,
                "doppler": float(obsd.D[0]),
                "freq": freq,
                "nlos": nlos,
            })
    return rows


def add_derived_features(df: pd.DataFrame, cn0_window: int = 5) -> pd.DataFrame:
    """Add pr_resid_norm, pr_rate_consistency, cn0_std to a raw-row frame.
    Operates within each satellite arc / epoch; identical at train and
    inference time."""
    if df.empty:
        for c in ("pr_resid_norm", "pr_rate_consistency", "cn0_std"):
            df[c] = pd.Series(dtype=float)
        return df

    # Feature 3: per-epoch robust-normalised pseudorange residual.
    def _norm_resid(g: pd.DataFrame) -> pd.Series:
        r = g["pr_resid"]
        scale = 1.4826 * r.abs().median()
        if not np.isfinite(scale) or scale < 1e-6:
            mean_abs = r.abs().mean()
            scale = mean_abs if np.isfinite(mean_abs) and mean_abs > 1e-6 else 1.0
        return r / scale
    df["pr_resid_norm"] = df.groupby("epoch", group_keys=False).apply(_norm_resid)

    # Feature 4: pseudorange-rate consistency (measured vs Doppler-predicted).
    df = df.sort_values(["sat", "epoch"]).reset_index(drop=True)
    lam = CLIGHT / df["freq"]
    df["_pred_rate"] = -lam * df["doppler"]
    parts = []
    for _, g in df.groupby("sat", sort=False):
        dt = g["utc"].diff()
        meas_rate = g["pseudorange"].diff() / dt
        pred_rate = 0.5 * (g["_pred_rate"] + g["_pred_rate"].shift(1))
        c = (meas_rate - pred_rate).abs()
        c[(dt <= 0) | (dt > 5.0)] = np.nan     # only adjacent ~1 Hz epochs
        parts.append(c)
    df["pr_rate_consistency"] = pd.concat(parts)

    # Feature 5: trailing rolling std of C/N0 per satellite arc.
    df["cn0_std"] = (
        df.groupby("sat", sort=False)["cn0"]
        .transform(lambda s: s.rolling(cn0_window, min_periods=1).std())
        .fillna(0.0)
    )

    return df.drop(columns=["_pred_rate", "freq", "doppler", "pseudorange"])


def _labels_to_lookup(labels: list) -> dict[int, tuple[set, set]]:
    lut = {}
    for e, label in enumerate(labels):
        all_idx = set(int(x) for x in label[1])
        los_idx = set(int(x) for x in label[2])
        lut[e] = (all_idx, los_idx)
    return lut


def extract_run(run: str, rover_obs: Path, nav_files: list[Path],
                label_pkl: Path, gt_csv: Path, start_utc: float, end_utc: float,
                cn0_window: int = 5) -> pd.DataFrame:
    obs, nav = read_rinex(rover_obs, nav_files)
    labels = safe_load_labels(label_pkl)

    # Match rover epochs to the labelled window using the config UTC bounds
    # (the label pickle's own timestamps are GPS time, 18 s ahead of UTC, so
    # we do NOT derive the window from them). The upstream demo filters obs to
    # [start_utc, end_utc] and aligns the surviving epochs 1:1 with the labels.
    slices = epoch_slices(obs)
    kept = [(i, m) for (i, m) in slices
            if start_utc - 0.5 <= epoch_utc(obs.data[i].time) <= end_utc + 0.5]
    if len(kept) != len(labels):
        print(f"[{run}] WARNING epoch/label mismatch: {len(kept)} vs "
              f"{len(labels)}; aligning to min", file=sys.stderr)

    # Build a windowed obs_t so raw_epoch_rows enumerates exactly the kept
    # epochs in order, and pass the label lookup keyed by that same order.
    n = min(len(kept), len(labels))
    win = prl.obs_t()
    total = sum(m for (_, m) in kept[:n])
    win.data = prl.Arr1Dobsd_t(total)
    w = 0
    for (i, m) in kept[:n]:
        for j in range(m):
            win.data[w] = obs.data[i + j]
            w += 1
    win.n = total
    win.nmax = total

    lut = _labels_to_lookup(labels[:n])
    rows = raw_epoch_rows(obs=win, nav=nav, prcopt=make_prcopt(), label_lookup=lut)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = add_derived_features(df, cn0_window=cn0_window)
    df.insert(0, "run", run)
    return df.sort_values(["run", "epoch", "sat"]).reset_index(drop=True)


def discover_nav_files(gnss_day_dir: Path, sta_prefix: str) -> list[Path]:
    """RINEX navigation files supplying broadcast ephemeris for the day.

    Two sources, in order of preference:
      1. A public IGS broadcast file under ``<day>/brdc/`` (e.g. the BKG
         BRDC00WRD_R_*_MN.rnx multi-GNSS navigation file). This is the
         portable, unrestricted ephemeris source and does not depend on the
         Hong Kong reference-station files.
      2. The KLTDataset reference-station files under ``<day>/sta/`` matching
         ``sta_prefix`` (their RINEX navigation records), excluding the bulky
         ``.??o`` observation file.
    """
    brdc_dir = gnss_day_dir / "brdc"
    brdc = sorted(brdc_dir.glob("*.rnx")) + sorted(brdc_dir.glob("*.[0-9][0-9][pnglfcNGLFC]"))
    if brdc:
        return brdc
    sta_dir = gnss_day_dir / "sta"
    files = sorted(sta_dir.glob(f"{sta_prefix}*"))
    nav = [f for f in files if not f.name.lower().endswith("o")]
    return nav if nav else files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("/opt/klt_raw"),
                    help="root produced by fetch_klt_data.py")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="output features table (.csv or .csv.gz)")
    ap.add_argument("--cn0-window", type=int, default=5)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from fetch_klt_data import RUNS

    frames = []
    for run in args.runs:
        meta = RUNS[run]
        day_dir = args.raw / "data" / "GNSS" / meta["day"]
        rover_obs = day_dir / meta["obs"]
        nav_files = discover_nav_files(day_dir, meta["sta_prefix"])
        label_pkl = args.raw / "label" / run / "nlos.pkl"
        gt_csv = args.raw / "label" / run / "gt.csv"
        print(f"[{run}] obs={rover_obs.name} nav={[f.name for f in nav_files]}",
              file=sys.stderr)
        df = extract_run(run, rover_obs, nav_files, label_pkl, gt_csv,
                         start_utc=meta["start"], end_utc=meta["end"],
                         cn0_window=args.cn0_window)
        if df.empty:
            print(f"[{run}] no rows", file=sys.stderr)
        else:
            print(f"[{run}] {len(df)} labelled (epoch,sat) rows; "
                  f"NLOS fraction {df['nlos'].mean():.3f}", file=sys.stderr)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows to {args.out}", file=sys.stderr)
    print(out.groupby(["run", "nlos"]).size().to_string(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
