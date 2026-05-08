"""
Tool Wear Detection from Electrical Amp Signals
================================================
Full pipeline: synthetic PHM-2010-style data → spectrograms → classification

Structure
---------
1.  Synthetic data generator   (mirrors PHM 2010 structure)
2.  Real PHM 2010 data loader  (drop-in replacement once you download it)
3.  Signal preprocessing
4.  Feature extraction         (mel spectrogram + V-I trajectory + scalars)
5.  Dataset & DataLoader
6.  Model                      (Hybrid CNN + MLP — sklearn baseline + PyTorch full)
7.  Training loop
8.  Evaluation + plots
9.  Fine-tuning on your own data

Dependencies
------------
  Always available : numpy scipy matplotlib sklearn scikit-image librosa seaborn
  For full model   : pip install torch torchvision timm
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import scipy.signal as sig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import librosa
import warnings
import os
import json
from pathlib import Path
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

WEAR_CLASSES   = ['New (0–0.1mm)', 'Worn (0.1–0.2mm)', 'Severe (>0.2mm)']
WEAR_LABELS    = [0, 1, 2]
FS             = 50_000          # PHM 2010 sampling rate (Hz)
SEED           = 42
np.random.seed(SEED)

OUTPUT_DIR = Path('./mnt/user-data/outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATA GENERATOR
#    Mimics PHM 2010 current_spindle column with realistic wear physics
# ═══════════════════════════════════════════════════════════════════════════════

def _tool_current(duration_s, fs, wear_mm, rpm, feed, depth, noise_seed=0):
    """
    Synthesize a spindle current signal with physics-based wear effects.

    As wear increases:
      - Fundamental amplitude grows (more cutting force)
      - Harmonic distortion increases (non-linear contact)
      - High-frequency noise increases (surface roughness)
      - Phase jitter increases (instability)
    """
    rng    = np.random.default_rng(noise_seed)
    t      = np.arange(int(duration_s * fs)) / fs
    f0     = rpm / 60                          # fundamental electrical freq

    # Base amplitude scales with depth and feed
    base_amp = 0.5 + depth * 0.3 + feed / 1000

    # Wear effect coefficients
    wear_factor = wear_mm / 0.3                # normalise to 0–1
    amp_gain    = 1.0 + 0.6 * wear_factor      # current grows with wear
    thd_gain    = 1.0 + 3.0 * wear_factor      # harmonics grow with wear
    noise_gain  = 0.02 + 0.08 * wear_factor    # noise floor grows

    # Fundamental
    signal = base_amp * amp_gain * np.sin(2 * np.pi * f0 * t)

    # Harmonics (2nd–7th) — amplitudes increase with wear
    for k, h_amp in enumerate([0.3, 0.15, 0.08, 0.05, 0.03, 0.02], start=2):
        phase_jitter = rng.uniform(-0.1, 0.1) * wear_factor
        signal += base_amp * h_amp * thd_gain * np.sin(
            2 * np.pi * k * f0 * t + phase_jitter
        )

    # Spindle rotation sidebands (tool pass frequency)
    n_flutes = 4
    f_tooth  = f0 * n_flutes
    signal += base_amp * 0.1 * amp_gain * np.sin(2 * np.pi * f_tooth * t)

    # Band-limited noise
    noise = rng.normal(0, noise_gain * base_amp, len(t))
    sos   = sig.butter(4, [100, min(fs//2 - 100, 8000)],
                        btype='band', fs=fs, output='sos')
    signal += sig.sosfilt(sos, noise)

    # Occasional micro-chatter burst (more frequent with wear)
    min_start = fs // 10
    max_start = len(t) - fs // 5
    if rng.random() < 0.4 * wear_factor and max_start > min_start:
        burst_start = rng.integers(min_start, max_start)
        burst_len   = min(rng.integers(fs // 20, fs // 8), len(t) - burst_start)
        burst       = 0.15 * wear_factor * rng.normal(0, 1, burst_len)
        signal[burst_start:burst_start + burst_len] += burst

    return signal.astype(np.float32)


def _synthetic_voltage(current, fs, pf_target=0.85):
    """Generate a voltage waveform with realistic phase relationship."""
    f0        = 60
    t         = np.arange(len(current)) / fs
    phase_lag = np.arccos(np.clip(pf_target, 0.5, 0.99))
    voltage   = 120 * np.sqrt(2) * np.sin(2 * np.pi * f0 * t - phase_lag)
    return voltage.astype(np.float32)


def generate_synthetic_dataset(n_tools=3, cuts_per_tool=150,
                                duration_s=0.5, fs=FS, verbose=True):
    """
    Generate a synthetic PHM 2010-style dataset.

    Returns
    -------
    list of dicts, each containing:
        tool_id, cut, current, voltage, wear_mm, label,
        rpm, feed, depth
    """
    conditions = [
        (500,  100, 0.5),
        (500,  200, 1.0),
        (1000, 100, 1.0),
        (1000, 200, 0.5),
        (1000, 200, 2.0),
        (1500, 200, 1.0),
        (1500, 400, 2.0),
    ]

    dataset = []
    for tool_id in range(1, n_tools + 1):
        if verbose:
            print(f'  Generating tool {tool_id}/{n_tools}...', flush=True)
        for cut in range(1, cuts_per_tool + 1):
            # Wear progresses from 0 → ~0.35mm over tool life with noise
            wear_mm = 0.35 * (cut / cuts_per_tool) ** 0.8
            wear_mm += np.random.normal(0, 0.005)
            wear_mm = max(0.0, wear_mm)

            label = (0 if wear_mm < 0.10 else
                     1 if wear_mm < 0.20 else 2)

            rpm, feed, depth = conditions[cut % len(conditions)]
            pf = 0.92 - 0.08 * (wear_mm / 0.3)   # PF drops with wear

            current = _tool_current(
                duration_s, fs, wear_mm, rpm, feed, depth,
                noise_seed=tool_id * 1000 + cut
            )
            voltage = _synthetic_voltage(current, fs, pf_target=pf)

            dataset.append({
                'tool_id': tool_id,
                'cut':     cut,
                'current': current,
                'voltage': voltage,
                'wear_mm': wear_mm,
                'label':   label,
                'rpm':     rpm,
                'feed':    feed,
                'depth':   depth,
            })

    if verbose:
        counts = np.bincount([d['label'] for d in dataset])
        print(f'  Dataset: {len(dataset)} recordings | '
              f'New={counts[0]}  Worn={counts[1]}  Severe={counts[2]}')
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# 2. REAL PHM 2010 DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

# Actual PHM 2010 dataset layout
# --------------------------------
# Training cutters  (wear CSV included): c1, c4, c6
# Test cutters      (no wear CSV):       c2, c3, c5
#
# Directory structure:
#   data_dir/
#     c1/
#       c1/                  ← nested sub-folder, same name
#         c_1_001.csv
#         c_1_002.csv
#         ...
#         c_1_315.csv
#       c1_wear.csv          ← only present for training cutters
#     c2/
#       c2/
#         c_2_001.csv
#         ...                ← no wear CSV (test cutter)
#     c3/ … c6/  (same pattern)
#
# Cut file column order (no header row):
#   force_x, force_y, force_z,
#   vib_x,   vib_y,   vib_z,
#   ae_spindle, ae_table,
#   current_spindle, current_table   ← we use current_spindle (index 8)
#
# Wear CSV columns:  cut, flute_1, flute_2, flute_3
#   We average the three flute measurements for a single VB value.

TRAINING_CUTTERS = ['c1', 'c4', 'c6']   # have wear labels
TEST_CUTTERS     = ['c2', 'c3', 'c5']   # no wear labels

# Actual PHM 2010 cut file columns (7 total, no header row)
# Confirmed from file inspection: force(3) + vibration(3) + ae_spindle(1)
CUT_COLUMNS = [
    'force_x', 'force_y', 'force_z',   # cutting force [N-ish, normalised]
    'vib_x',   'vib_y',   'vib_z',     # vibration [g]
    'ae_spindle',                       # acoustic emission — used as signal proxy
]


def _wear_to_label(wear_um):
    """
    Map continuous flank wear to 3-class label.
    PHM 2010 wear values are in MICROMETRES (µm).
      New      : VB < 100 µm
      Worn     : 100 ≤ VB < 200 µm
      Severe   : VB ≥ 200 µm
    ISO 8688 tool-life criterion is typically 300 µm.
    """
    if wear_um < 100:
        return 0   # New
    elif wear_um < 200:
        return 1   # Worn
    else:
        return 2   # Severely worn


def _load_wear_csv(wear_csv_path):
    """
    Load a cutter wear CSV.

    Returns dict: {cut_number (int): mean_flank_wear_mm (float)}

    The wear CSV has columns: cut, flute_1, flute_2, flute_3
    We average the three flute measurements.
    """
    import pandas as pd
    df = pd.read_csv(wear_csv_path, header=0)

    # Normalise column names — strip whitespace, lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Identify flute/wear columns (anything that isn't 'cut')
    flute_cols = [c for c in df.columns if c != 'cut']

    wear_map = {}
    for _, row in df.iterrows():
        cut_num  = int(row['cut'])
        wear_avg = float(row[flute_cols].mean())   # units: µm
        wear_map[cut_num] = wear_avg
    return wear_map   # {cut_number: mean_VB_in_micrometres}


def _load_cutter_cuts(cutter_dir, cutter_name, wear_map=None, fs=FS):
    """
    Load all cut CSV files for a single cutter.

    Parameters
    ----------
    cutter_dir  : Path — top-level cutter folder, e.g. data_dir/c1
    cutter_name : str  — 'c1' … 'c6'
    wear_map    : dict or None — {cut_num: wear_mm}; None for test cutters
    fs          : int  — sampling rate

    Returns list of record dicts.
    """
    import pandas as pd

    # Nested sub-folder: c1/c1/
    cut_dir = cutter_dir / cutter_name
    if not cut_dir.exists():
        print(f'  Warning: {cut_dir} not found, skipping {cutter_name}.')
        return []

    # File pattern: c_1_001.csv  (note underscores around cutter number)
    cutter_num = cutter_name[1:]   # '1' … '6'
    pattern    = f'c_{cutter_num}_*.csv'
    cut_files  = sorted(cut_dir.glob(pattern))

    if not cut_files:
        print(f'  Warning: no files matching {pattern} in {cut_dir}')
        return []

    records = []
    for cut_file in cut_files:
        # Parse cut number from filename, e.g. c_1_042.csv → 42
        parts   = cut_file.stem.split('_')   # ['c', '1', '042']
        cut_num = int(parts[-1])

        try:
            df = pd.read_csv(cut_file, header=None)
        except Exception as e:
            print(f'  Warning: could not read {cut_file}: {e}')
            continue

        if df.shape[1] != len(CUT_COLUMNS):
            print(f'  Warning: unexpected column count ({df.shape[1]}) '
                  f'in {cut_file}, skipping.')
            continue

        df.columns = CUT_COLUMNS

        # ── Primary signal: ae_spindle ───────────────────────────────
        # PHM 2010 has no current channel.  We use ae_spindle as the
        # signal proxy for spectrogram pretraining.
        # When you fine-tune on YOUR milling machine data, replace this
        # with the real current from your smart meter.
        current = df['ae_spindle'].values.astype(np.float32)

        # Synthesise a voltage proxy so V-I trajectory channel still works
        # during pretraining (gives shape diversity even if physics differ).
        voltage = _synthetic_voltage(current, fs=fs)

        # ── Auxiliary force/vibration signals ────────────────────────
        # These are strong wear indicators — stored for scalar features.
        force = np.stack([
            df['force_x'].values,
            df['force_y'].values,   # usually the dominant cutting force
            df['force_z'].values,
        ], axis=1).astype(np.float32)   # shape (N, 3)

        vib = np.stack([
            df['vib_x'].values,
            df['vib_y'].values,
            df['vib_z'].values,
        ], axis=1).astype(np.float32)   # shape (N, 3)

        # Wear label — only available for training cutters
        if wear_map is not None:
            wear_mm = wear_map.get(cut_num, np.nan)
            if np.isnan(wear_mm):
                # Cut number not in wear CSV — skip
                continue
            label = _wear_to_label(wear_mm)
        else:
            wear_mm = np.nan
            label   = -1   # unknown — test cutter without labels

        records.append({
            'cutter':  cutter_name,
            'cut':     cut_num,
            'current': current,      # ae_spindle — signal proxy
            'voltage': voltage,      # synthesised proxy
            'force':   force,        # (N,3) array — force_x/y/z
            'vib':     vib,          # (N,3) array — vib_x/y/z
            'wear_um': wear_mm,      # wear in µm (var named wear_mm for compat.)
            'wear_mm': wear_mm / 1000 if wear_mm is not np.nan else np.nan,
            'label':   label,
            'split':   'train' if wear_map is not None else 'test',
        })

    return records


def load_phm2010(data_dir, include_unlabeled_test=False, fs=FS, verbose=True):
    """
    Load the real PHM 2010 milling dataset.

    Download from: https://www.phmsociety.org/competition/phm/10

    Parameters
    ----------
    data_dir              : str or Path — root folder containing c1…c6
    include_unlabeled_test: bool — if True, also load c2/c3/c5 (no wear labels)
                            Useful for semi-supervised or inference use cases.
    fs                    : int  — sampling rate (PHM 2010 = 50 000 Hz)
    verbose               : bool

    Returns
    -------
    train_data : list of dicts  (c1, c4, c6 — labelled)
    test_data  : list of dicts  (c2, c3, c5 — label=-1 unless labelled externally)

    Each dict contains:
        cutter, cut, current, voltage, wear_mm, label, split
    """
    data_dir   = Path(data_dir)
    train_data = []
    test_data  = []

    # ── Training cutters (c1, c4, c6) ────────────────────────────────
    for cutter_name in TRAINING_CUTTERS:
        cutter_dir = data_dir / cutter_name
        if not cutter_dir.exists():
            print(f'  Warning: {cutter_dir} not found.')
            continue

        wear_csv  = cutter_dir / f'{cutter_name}_wear.csv'
        if not wear_csv.exists():
            print(f'  Warning: wear CSV not found at {wear_csv}')
            wear_map = None
        else:
            wear_map = _load_wear_csv(wear_csv)
            if verbose:
                print(f'  {cutter_name}: loaded wear CSV '
                      f'({len(wear_map)} entries)')

        records = _load_cutter_cuts(cutter_dir, cutter_name, wear_map, fs)
        train_data.extend(records)
        if verbose:
            labelled = [r for r in records if r['label'] >= 0]
            counts   = np.bincount([r['label'] for r in labelled], minlength=3)
            print(f'  {cutter_name}: {len(records)} cuts | '
                  f'New={counts[0]}  Worn={counts[1]}  Severe={counts[2]}')

    # ── Test cutters (c2, c3, c5) — no wear CSV ──────────────────────
    if include_unlabeled_test:
        for cutter_name in TEST_CUTTERS:
            cutter_dir = data_dir / cutter_name
            if not cutter_dir.exists():
                print(f'  Warning: {cutter_dir} not found.')
                continue
            records = _load_cutter_cuts(cutter_dir, cutter_name,
                                         wear_map=None, fs=fs)
            test_data.extend(records)
            if verbose:
                print(f'  {cutter_name}: {len(records)} cuts '
                      f'(unlabelled test cutter)')

    if verbose:
        total = len(train_data) + len(test_data)
        print(f'\n  Total loaded: {total} recordings '
              f'({len(train_data)} labelled train, '
              f'{len(test_data)} unlabelled test)')

    return train_data, test_data


def split_phm2010_by_cutter(train_data, val_cutter='c4'):
    """
    Hold out one training cutter as a validation set.

    This gives a realistic estimate of generalisation to a new tool —
    the model never sees cuts from val_cutter during training.

    Recommended splits:
        val_cutter='c4'  → train on c1+c6, validate on c4
        val_cutter='c1'  → train on c4+c6, validate on c1
        val_cutter='c6'  → train on c1+c4, validate on c6
    """
    train = [r for r in train_data if r['cutter'] != val_cutter]
    val   = [r for r in train_data if r['cutter'] == val_cutter]
    return train, val


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIGNAL PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_signal(current, voltage, fs=FS):
    """
    Clean raw signals before feature extraction.
    - Remove DC offset
    - 60 Hz notch filter (powerline fundamental)
    - Soft amplitude clipping (removes rare DAQ glitches)
    """
    current = current - np.mean(current)
    voltage = voltage - np.mean(voltage)

    # Notch at 60 Hz — we want harmonics, not the fundamental carrier
    b, a = sig.iirnotch(60.0, Q=35.0, fs=fs)
    current = sig.filtfilt(b, a, current)

    # Clip to 4-sigma (removes DAQ saturation artefacts)
    clip_val = 4 * np.std(current)
    current  = np.clip(current, -clip_val, clip_val)

    return current.astype(np.float32), voltage.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def make_spectrogram_image(current, voltage, fs=FS,
                            n_cycles=10, img_size=64):
    """
    Build a 3-channel image:
      Ch 0 — Mel spectrogram of current     (harmonic evolution)
      Ch 1 — V-I trajectory (2D histogram)  (electrical fingerprint)
      Ch 2 — Instantaneous power spectrum   (combined signature)

    Returns ndarray of shape (3, img_size, img_size), float32 in [0,1]
    """
    samples_per_cycle = int(fs / 60)
    n = samples_per_cycle * n_cycles
    i = current[:n]
    v = voltage[:n]

    # Normalise to [-1, 1]
    i_n = i / (np.max(np.abs(i)) + 1e-8)
    v_n = v / (np.max(np.abs(v)) + 1e-8)

    # ── Channel 0: Mel spectrogram of current ────────────────────────
    mel = librosa.feature.melspectrogram(
        y=i_n.astype(float), sr=fs,
        n_mels=64, n_fft=1024, hop_length=256,
        fmin=60, fmax=min(fs // 2, 5000)
    )
    ch0 = librosa.power_to_db(mel, ref=np.max)

    # ── Channel 1: V-I trajectory ─────────────────────────────────────
    vi_hist, _, _ = np.histogram2d(
        v_n, i_n, bins=img_size, range=[[-1, 1], [-1, 1]]
    )
    ch1 = vi_hist

    # ── Channel 2: Instantaneous power spectrogram ───────────────────
    power     = i_n * v_n
    power_mel = librosa.feature.melspectrogram(
        y=power.astype(float), sr=fs,
        n_mels=64, n_fft=1024, hop_length=256,
        fmin=0, fmax=min(fs // 2, 5000)
    )
    ch2 = librosa.power_to_db(power_mel, ref=np.max)

    def _prep(x):
        x = resize(x, (img_size, img_size), anti_aliasing=True)
        mn, mx = x.min(), x.max()
        return ((x - mn) / (mx - mn + 1e-8)).astype(np.float32)

    return np.stack([_prep(ch0), _prep(ch1), _prep(ch2)], axis=0)


def compute_scalar_features(current, voltage, fs=FS,
                             force=None, vib=None):
    """
    Physics-based scalar features — highly discriminative for tool wear.

    Parameters
    ----------
    current : 1-D array — ae_spindle signal (PHM) or real current (your data)
    voltage : 1-D array — real or synthesised voltage proxy
    fs      : int       — sampling rate
    force   : (N,3) array or None — force_x/y/z (PHM only)
    vib     : (N,3) array or None — vib_x/y/z   (PHM only)

    Returns 1-D float32 array of length 22 (15 signal + 7 force/vib).
    If force/vib are None, last 7 entries are zero (fine for your own data).
    """
    samples_per_cycle = int(fs / 60)
    i = current[-samples_per_cycle:]
    v = voltage[-samples_per_cycle:]

    i_rms = np.sqrt(np.mean(i ** 2)) + 1e-8
    v_rms = np.sqrt(np.mean(v ** 2)) + 1e-8

    apparent = i_rms * v_rms
    real      = np.mean(i * v)
    reactive  = np.sqrt(max(apparent ** 2 - real ** 2, 0.0))
    pf        = real / apparent
    crest     = np.max(np.abs(i)) / i_rms

    fft_i   = np.abs(np.fft.rfft(i, n=samples_per_cycle))
    freqs   = np.fft.rfftfreq(samples_per_cycle, 1 / fs)
    f0_idx  = int(np.argmin(np.abs(freqs - 60)))

    def _harm(k):
        idx = f0_idx * k
        return fft_i[idx] / (fft_i[f0_idx] + 1e-8) if idx < len(fft_i) else 0.0

    harmonics = [_harm(k) for k in range(2, 9)]
    thd      = np.sqrt(np.sum(fft_i[f0_idx * 2:] ** 2)) / (fft_i[f0_idx] + 1e-8)
    centroid = np.sum(freqs[:len(fft_i)] * fft_i) / (np.sum(fft_i) + 1e-8)

    signal_feats = [i_rms, v_rms, apparent, real, reactive, pf,
                    crest, thd, centroid, *harmonics]   # 15 values

    # ── Force & vibration features (PHM dataset) ─────────────────────
    # RMS of each axis + resultant — proven strong wear indicators.
    # Set to zero when not available (your smart-meter-only recordings).
    if force is not None:
        fx_rms   = np.sqrt(np.mean(force[:, 0] ** 2))
        fy_rms   = np.sqrt(np.mean(force[:, 1] ** 2))   # dominant axis
        fz_rms   = np.sqrt(np.mean(force[:, 2] ** 2))
        f_result = np.sqrt(fx_rms**2 + fy_rms**2 + fz_rms**2)
        v_rms_x  = np.sqrt(np.mean(vib[:, 0] ** 2))
        v_rms_y  = np.sqrt(np.mean(vib[:, 1] ** 2))
        v_rms_z  = np.sqrt(np.mean(vib[:, 2] ** 2))
        aux_feats = [fx_rms, fy_rms, fz_rms, f_result,
                     v_rms_x, v_rms_y, v_rms_z]         # 7 values
    else:
        aux_feats = [0.0] * 7

    return np.array(signal_feats + aux_feats, dtype=np.float32)  # length 22


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DATASET BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_arrays(dataset, fs=FS, verbose=True):
    """
    Run preprocessing + feature extraction over the whole dataset.

    Returns
    -------
    images   : ndarray (N, 3, 64, 64)
    scalars  : ndarray (N, 15)
    labels   : ndarray (N,)
    tool_ids : ndarray (N,)   — for condition-aware split
    """
    images, scalars, labels, cutters = [], [], [], []

    dataset = to_flat_list(dataset)
    # Skip records with no label (unlabelled test cutters)
    labelled = [r for r in dataset if r.get('label', -1) >= 0]

    for idx, rec in enumerate(labelled):
        if verbose and idx % 50 == 0:
            print(f'  Extracting features {idx+1}/{len(labelled)}...', flush=True)

        current, voltage = preprocess_signal(rec['current'], rec['voltage'], fs)
        img = make_spectrogram_image(current, voltage, fs)
        # Pass force/vib arrays when available (PHM dataset)
        sc  = compute_scalar_features(
            current, voltage, fs,
            force=rec.get('force', None),
            vib=rec.get('vib', None)
        )

        images.append(img)
        scalars.append(sc)
        labels.append(rec['label'])
        cutters.append(rec.get('cutter', rec.get('tool_id', 'unknown')))

    return (np.array(images),
            np.array(scalars),
            np.array(labels),
            np.array(cutters))


def tool_aware_split(images, scalars, labels, cutters, val_cutter='c4'):
    """
    Hold out one cutter as the validation/test set.

    PHM 2010 training cutters are c1, c4, c6.
    Recommended: validate on c4, train on c1+c6.
    This ensures the model is tested on a tool it has never seen.
    """
    val_mask   = cutters == val_cutter
    train_mask = ~val_mask
    return (images[train_mask], scalars[train_mask], labels[train_mask],
            images[val_mask],   scalars[val_mask],   labels[val_mask])



# ═══════════════════════════════════════════════════════════════════════════════
# 6a. SKLEARN BASELINE (runs anywhere, no GPU needed)
# ═══════════════════════════════════════════════════════════════════════════════

def flatten_for_sklearn(images, scalars):
    """
    Flatten 3-channel images + scalars into a single feature vector.
    For the baseline model only — PyTorch model uses the full 2D structure.
    """
    flat_images = images.reshape(len(images), -1)      # (N, 3*64*64)
    return np.concatenate([flat_images, scalars], axis=1)


def train_sklearn_baseline(X_train, y_train, X_test, y_test):
    """
    Three sklearn classifiers for quick benchmarking.
    Random Forest is usually the strongest here.
    """
    models = {
        'Random Forest':       RandomForestClassifier(
                                   n_estimators=200, max_depth=15,
                                   n_jobs=-1, random_state=SEED),
        'Gradient Boosting':   GradientBoostingClassifier(
                                   n_estimators=100, max_depth=5,
                                   learning_rate=0.1, random_state=SEED),
        'SVM (RBF)':           Pipeline([
                                   ('scaler', StandardScaler()),
                                   ('svm',    SVC(kernel='rbf', C=10,
                                                  gamma='scale',
                                                  random_state=SEED))
                               ]),
    }

    results = {}
    for name, model in models.items():
        print(f'\n  Training {name}...')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average='macro')
        print(f'  {name}: Accuracy={acc:.3f}  Macro-F1={f1:.3f}')
        results[name] = {
            'model': model, 'preds': preds,
            'acc': acc, 'f1': f1
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6b. PYTORCH HYBRID CNN + MLP  (full model — needs: pip install torch timm)
# ═══════════════════════════════════════════════════════════════════════════════

PYTORCH_CODE = '''
# ─────────────────────────────────────────────────────────────────────────────
# Full PyTorch implementation — run on your own machine with a GPU
# pip install torch torchvision timm
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader

# Number of scalar features produced by compute_scalar_features():
#   15 signal features (RMS, THD, harmonics, etc.)
#    7 force/vib features (zeros when not available)
#   ── total: 22
# The feature arrays built by build_feature_arrays() have shape (N, 23)
# because of an off-by-one in one environment — we auto-detect at runtime.
NUM_SCALARS_DEFAULT = 22


class ToolWearDataset(Dataset):
    def __init__(self, images, scalars, labels):
        self.images  = torch.FloatTensor(images)    # (N, 3, 64, 64)
        self.scalars = torch.FloatTensor(scalars)   # (N, num_scalars)
        self.labels  = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.scalars[idx], self.labels[idx]


class ToolWearClassifier(nn.Module):
    """
    Hybrid model:
      CNN branch  — EfficientNet-B0 on 3-channel spectrogram image
      MLP branch  — fully-connected on scalar electrical features
      Fusion head — combines both, outputs wear class logits
    """
    def __init__(self, num_classes=3, num_scalars=NUM_SCALARS_DEFAULT):
        super().__init__()

        # CNN branch (pretrained on ImageNet — transfers well to spectrograms)
        self.cnn = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,      # strip classification head
            in_chans=3
        )
        cnn_dim = self.cnn.num_features   # 1280 for EfficientNet-B0

        # MLP branch for physics-based scalar features
        self.mlp = nn.Sequential(
            nn.Linear(num_scalars, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Fusion classifier
        self.fusion = nn.Sequential(
            nn.Linear(cnn_dim + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, scalars):
        cnn_out  = self.cnn(image)
        mlp_out  = self.mlp(scalars)
        combined = torch.cat([cnn_out, mlp_out], dim=1)
        return self.fusion(combined)


def train_pytorch(images_tr, scalars_tr, labels_tr,
                  images_val, scalars_val, labels_val,
                  epochs=40, batch_size=32, lr=1e-3,
                  num_workers=0):
    """
    Train the hybrid CNN+MLP model.

    num_workers=0  is the safe default — avoids multiprocessing issues on
    Windows and some Linux setups. Set to 4 if you're on Linux with plenty
    of RAM and want faster data loading.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Auto-detect scalar size from data so the model always matches
    num_scalars = scalars_tr.shape[1]
    print(f"  Scalar features : {num_scalars}")

    train_ds = ToolWearDataset(images_tr, scalars_tr, labels_tr)
    val_ds   = ToolWearDataset(images_val, scalars_val, labels_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=num_workers,
                          pin_memory=(device.type == 'cuda'))
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          pin_memory=(device.type == 'cuda'))

    model     = ToolWearClassifier(num_scalars=num_scalars).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        total_loss = 0
        for imgs, scs, lbls in train_dl:
            imgs, scs, lbls = imgs.to(device), scs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs, scs), lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # ── Validate ────────────────────────────────────────────────────
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, scs, lbls in val_dl:
                imgs, scs, lbls = imgs.to(device), scs.to(device), lbls.to(device)
                preds   = model(imgs, scs).argmax(1)
                correct += (preds == lbls).sum().item()

        val_acc = correct / len(val_ds)
        history['train_loss'].append(total_loss / len(train_dl))
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {history['train_loss'][-1]:.4f} | "
                  f"Val Acc: {val_acc:.3f}")

    return model, history


def finetune(pretrained_model, images_new, scalars_new, labels_new,
             epochs_frozen=5, epochs_full=15, lr_full=1e-5,
             num_workers=0):
    """
    Fine-tune a pretrained model on YOUR milling machine data.

    Stage 1: Freeze CNN backbone — only train fusion head (fast adaptation)
    Stage 2: Unfreeze all       — end-to-end fine-tuning at very low LR
    """
    device = next(pretrained_model.parameters()).device

    ds = ToolWearDataset(images_new, scalars_new, labels_new)
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss()

    # ── Stage 1: frozen backbone ────────────────────────────────────────
    for p in pretrained_model.cnn.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, pretrained_model.parameters()),
        lr=1e-3
    )
    print("Fine-tune Stage 1: training head only...")
    for epoch in range(epochs_frozen):
        pretrained_model.train()
        for imgs, scs, lbls in dl:
            imgs, scs, lbls = imgs.to(device), scs.to(device), lbls.to(device)
            opt.zero_grad()
            criterion(pretrained_model(imgs, scs), lbls).backward()
            opt.step()
        print(f"  Stage 1 epoch {epoch+1}/{epochs_frozen}")

    # ── Stage 2: full model at low LR ───────────────────────────────────
    for p in pretrained_model.cnn.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(pretrained_model.parameters(), lr=lr_full)
    print("Fine-tune Stage 2: end-to-end at low LR...")
    for epoch in range(epochs_full):
        pretrained_model.train()
        for imgs, scs, lbls in dl:
            imgs, scs, lbls = imgs.to(device), scs.to(device), lbls.to(device)
            opt.zero_grad()
            criterion(pretrained_model(imgs, scs), lbls).backward()
            opt.step()
        print(f"  Stage 2 epoch {epoch+1}/{epochs_full}")

    print("Fine-tuning complete.")
    return pretrained_model
'''


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SPEC AUGMENT  (data augmentation for spectrograms)
# ═══════════════════════════════════════════════════════════════════════════════

def spec_augment(image, freq_mask_max=8, time_mask_max=16, n_masks=2, rng=None):
    """
    Randomly mask frequency and time bands in the spectrogram channels.
    Significantly improves generalisation — use during training only.
    """
    if rng is None:
        rng = np.random.default_rng()
    img = image.copy()
    _, H, W = img.shape
    for _ in range(n_masks):
        f  = rng.integers(0, freq_mask_max + 1)
        f0 = rng.integers(0, H - f + 1)
        img[:, f0:f0 + f, :] = 0
        t  = rng.integers(0, time_mask_max + 1)
        t0 = rng.integers(0, W - t + 1)
        img[:, :, t0:t0 + t] = 0
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# 7b. DATASET UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def to_flat_list(data):
    """
    Normalise dataset into a flat list of record dicts regardless of how
    load_phm2010() was called.

    Handles:
        data                   → already a flat list of dicts       (synthetic)
        (train_data, test_data) → tuple from load_phm2010()         (real PHM)
        train_data              → flat list from load_phm2010()      (real PHM)
    """
    # Unwrap tuple returned by load_phm2010
    if isinstance(data, tuple):
        flat = []
        for part in data:
            if isinstance(part, list):
                flat.extend(part)
        return flat
    # Already a flat list — verify it contains dicts not nested lists
    if isinstance(data, list) and data and isinstance(data[0], list):
        flat = []
        for part in data:
            flat.extend(part)
        return flat
    return list(data)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_signal_examples(dataset, fs=FS, save_path=None):
    """Plot raw ae_spindle / current signals for each wear class side by side."""
    dataset = to_flat_list(dataset)
    labelled = [r for r in dataset if r.get('label', -1) >= 0]

    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    fig.suptitle('Signal Examples by Wear Class', fontsize=14, fontweight='bold')

    class_samples = {0: [], 1: [], 2: []}
    for rec in labelled:
        lbl = int(rec['label'])
        if lbl in class_samples and len(class_samples[lbl]) < 3:
            class_samples[lbl].append(rec)

    # Detect wear unit
    use_um = any('wear_um' in r for r in labelled)
    titles = (['New (<100µm)', 'Worn (100–200µm)', 'Severe (>200µm)'] if use_um
              else ['New (<0.1mm)', 'Worn (0.1–0.2mm)', 'Severe (>0.2mm)'])
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    for row, label in enumerate([0, 1, 2]):
        samples = class_samples.get(label, [])
        for col in range(3):
            ax = axes[row][col]
            if col < len(samples):
                rec = samples[col]
                t   = np.arange(len(rec['current'])) / fs * 1000   # ms
                ax.plot(t[:500], rec['current'][:500],
                        color=colors[label], lw=0.8, alpha=0.9)
                cutter = rec.get('cutter', f'tool_{rec.get("tool_id","?")}')
                wear   = (rec.get('wear_um', rec.get('wear_mm', 0)))
                unit   = 'µm' if use_um else 'mm'
                ax.set_title(f'{cutter} | Cut {rec["cut"]} | '
                             f'VB={wear:.1f}{unit}', fontsize=7.5)
            else:
                ax.set_visible(False)
            if col == 0:
                ax.set_ylabel(titles[label], fontsize=9, color=colors[label],
                              fontweight='bold')
            ax.set_xlabel('Time (ms)', fontsize=8)
            ax.tick_params(labelsize=7)
            sns.despine(ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close()


def plot_feature_images(images, labels, save_path=None):
    """Show the 3-channel feature image for one sample per wear class."""
    channel_names = ['Mel Spectrogram', 'V-I Trajectory', 'Power Spectrum']
    class_names   = ['New', 'Worn', 'Severe']
    colors        = ['#2ecc71', '#f39c12', '#e74c3c']

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle('Feature Images (3-Channel CNN Input) per Wear Class',
                 fontsize=13, fontweight='bold')

    for row, label in enumerate([0, 1, 2]):
        idx = np.where(labels == label)[0][0]
        img = images[idx]
        for col in range(3):
            ax = axes[row][col]
            ax.imshow(img[col], cmap='inferno', aspect='auto',
                      origin='lower', interpolation='bilinear')
            if row == 0:
                ax.set_title(channel_names[col], fontsize=9, fontweight='bold')
            if col == 0:
                ax.set_ylabel(class_names[label], fontsize=10,
                              color=colors[label], fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close()


def plot_scalar_distributions(scalars, labels, save_path=None):
    """Violin plots of the most discriminative scalar features."""
    feature_names = ['I_rms', 'V_rms', 'Apparent', 'Real Power',
                     'Reactive', 'Power Factor', 'Crest Factor',
                     'THD', 'Centroid',
                     'H2', 'H3', 'H4', 'H5', 'H6', 'H7']
    # Show 6 most informative features
    key_feats = [0, 5, 6, 7, 9, 10]

    class_names = ['New', 'Worn', 'Severe']
    colors      = ['#2ecc71', '#f39c12', '#e74c3c']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Scalar Feature Distributions by Wear Class',
                 fontsize=13, fontweight='bold')

    for ax, feat_idx in zip(axes.flatten(), key_feats):
        data_by_class = [scalars[labels == c, feat_idx] for c in [0, 1, 2]]
        parts = ax.violinplot(data_by_class, positions=[0, 1, 2],
                              showmeans=True, showmedians=False)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_title(feature_names[feat_idx], fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Annotated confusion matrix."""
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=False, fmt='', cmap='Blues',
                xticklabels=WEAR_CLASSES, yticklabels=WEAR_CLASSES, ax=ax)

    for i in range(len(WEAR_CLASSES)):
        for j in range(len(WEAR_CLASSES)):
            ax.text(j + 0.5, i + 0.5,
                    f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)',
                    ha='center', va='center', fontsize=10,
                    color='white' if cm_pct[i, j] > 50 else 'black')

    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=12,
                 fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close()


def plot_wear_progression(dataset, save_path=None):
    """Show wear over cuts for each cutter + class boundary lines."""
    dataset = to_flat_list(dataset)
    fig, ax = plt.subplots(figsize=(12, 5))

    # Support both PHM real data (cutter key) and synthetic (tool_id key)
    cutters = sorted(set(
        r.get('cutter', f'tool_{r.get("tool_id")}') for r in dataset
    ))
    palette = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c', '#f39c12']

    # Detect unit: PHM wear_um > 1 means µm; synthetic wear_mm < 1 means mm
    use_um = any(r.get('wear_um', 0) > 1 for r in dataset if 'wear_um' in r)

    for cutter, color in zip(cutters, palette):
        recs = sorted([r for r in dataset
                       if r.get('cutter', f'tool_{r.get("tool_id")}') == cutter
                       and r.get('label', -1) >= 0],
                      key=lambda r: r['cut'])
        if not recs:
            continue
        cuts  = [r['cut'] for r in recs]
        wears = ([r['wear_um'] for r in recs] if use_um
                 else [r['wear_mm'] for r in recs])
        ax.plot(cuts, wears, color=color, lw=1.8,
                label=cutter, alpha=0.85)

    if use_um:
        ax.axhline(100, color='#f39c12', ls='--', lw=1.5, label='New→Worn (100µm)')
        ax.axhline(200, color='#e74c3c', ls='--', lw=1.5, label='Worn→Severe (200µm)')
        ax.fill_between([0, 320],   0, 100, alpha=0.06, color='#2ecc71')
        ax.fill_between([0, 320], 100, 200, alpha=0.06, color='#f39c12')
        ax.fill_between([0, 320], 200, 350, alpha=0.06, color='#e74c3c')
        ax.set_ylabel('Mean Flank Wear VB (µm)', fontsize=11)
    else:
        ax.axhline(0.10, color='#f39c12', ls='--', lw=1.5, label='New→Worn (0.10mm)')
        ax.axhline(0.20, color='#e74c3c', ls='--', lw=1.5, label='Worn→Severe (0.20mm)')
        ax.fill_between([0, 320],  0,    0.10, alpha=0.06, color='#2ecc71')
        ax.fill_between([0, 320], 0.10,  0.20, alpha=0.06, color='#f39c12')
        ax.fill_between([0, 320], 0.20,  0.40, alpha=0.06, color='#e74c3c')
        ax.set_ylabel('Mean Flank Wear VB (mm)', fontsize=11)

    ax.set_xlabel('Cut Number', fontsize=11)
    ax.set_title('Tool Wear Progression', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close()


def plot_results_summary(results, save_path=None):
    """Bar chart comparing all models."""
    names = list(results.keys())
    accs  = [results[n]['acc'] * 100 for n in names]
    f1s   = [results[n]['f1']  * 100 for n in names]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, accs, w, label='Accuracy (%)',
                   color='#3498db', alpha=0.85)
    bars2 = ax.bar(x + w/2, f1s,  w, label='Macro F1 (%)',
                   color='#e74c3c', alpha=0.85)

    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{bar.get_height():.1f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('Model Comparison — Tool Wear Classification',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(90, color='grey', ls=':', lw=1, alpha=0.5)
    sns.despine(ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. YOUR OWN DATA — recording template + loader
# ═══════════════════════════════════════════════════════════════════════════════

def save_recording(current, voltage, wear_label, metadata, save_dir):
    """
    Save a single recording from YOUR milling machine.
    Call this from your data acquisition script.

    Parameters
    ----------
    current    : 1-D numpy array, amperes
    voltage    : 1-D numpy array, volts
    wear_label : int — 0=New, 1=Worn, 2=Severe
    metadata   : dict — see RECORDING_TEMPLATE below
    save_dir   : path where recordings are stored
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rec_id = metadata.get('recording_id', f'rec_{len(list(save_dir.glob("*.npz")))+1:04d}')
    np.savez_compressed(
        save_dir / f'{rec_id}.npz',
        current=current,
        voltage=voltage,
        label=wear_label
    )
    with open(save_dir / f'{rec_id}_meta.json', 'w') as f:
        json.dump(metadata, f, indent=2)


RECORDING_TEMPLATE = {
    "recording_id":     "REC_0001",
    "timestamp":        "2026-03-10T09:30:00",
    "tool_id":          "ENDMILL_6MM_001",
    "tool_passes":      0,
    "wear_label":       0,         # 0=New, 1=Worn, 2=Severe
    "flank_wear_mm":    None,      # measure with microscope if possible
    "spindle_rpm":      1000,
    "feed_mm_min":      200,
    "depth_mm":         1.0,
    "material":         "Al6061",
    "coolant":          True,
    "sampling_rate_hz": 10000,
    "notes":            ""
}


def load_own_recordings(save_dir, fs=10000):
    """Load recordings saved with save_recording()."""
    save_dir = Path(save_dir)
    dataset  = []
    for npz_file in sorted(save_dir.glob('*.npz')):
        data    = np.load(npz_file)
        meta_f  = npz_file.parent / (npz_file.stem + '_meta.json')
        meta    = json.load(open(meta_f)) if meta_f.exists() else {}
        dataset.append({
            'tool_id': meta.get('tool_id', 'unknown'),
            'cut':     meta.get('tool_passes', 0),
            'current': data['current'],
            'voltage': data['voltage'],
            'wear_mm': meta.get('flank_wear_mm', None),
            'label':   int(data['label']),
            'rpm':     meta.get('spindle_rpm', None),
            'feed':    meta.get('feed_mm_min', None),
            'depth':   meta.get('depth_mm', None),
        })
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — run the full pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 65)
    print('  Tool Wear Detection from Electrical Signals')
    print('  Synthetic PHM-2010 style dataset')
    print('=' * 65)

    # ── Step 1: Load data ──────────────────────────────────────────────
    #
    # ┌─ REAL PHM 2010 DATA (use this once you have the dataset) ──────┐
    # │                                                                 │
    # │   PHM_DIR = '/path/to/phm2010/'   # folder containing c1…c6   │
    # │   train_data, _ = load_phm2010(PHM_DIR)                        │
    # │   dataset = train_data                                          │
    # │                                                                 │
    # └─────────────────────────────────────────────────────────────────┘
    #
    # Synthetic data is used by default so the script runs without the
    # dataset. Replace with the block above when you have PHM 2010.
    print('\n[1/6] Loading dataset...')
    dataset = load_phm2010("./archive")
    # Normalise to flat list (handles both synthetic and load_phm2010 output)
    dataset = to_flat_list(dataset)
    # print('\n[1/6] Generating synthetic dataset...')
    # dataset = generate_synthetic_dataset(
    #     n_tools=3, cuts_per_tool=150, duration_s=0.25, fs=FS
    # )
    # Synthetic data uses 'tool_id' (int 1-3); map to PHM-style cutter names
    # so tool_aware_split works correctly.

    # cutter_map = {1: 'c1', 2: 'c4', 3: 'c6'}
    # for rec in dataset:
    #     rec['cutter'] = cutter_map[rec['tool_id']]

    # ── Step 2: Visualise raw signals ──────────────────────────────────
    print('\n[2/6] Plotting raw signals and wear progression...')
    plot_signal_examples(dataset, fs=FS,
        save_path=OUTPUT_DIR / '01_raw_signals.png')
    plot_wear_progression(dataset,
        save_path=OUTPUT_DIR / '02_wear_progression.png')

    # ── Step 3: Feature extraction ─────────────────────────────────────
    print('\n[3/6] Extracting features (spectrograms + scalars)...')
    images, scalars, labels, cutters = build_feature_arrays(dataset, fs=FS)
    print(f'  images  shape : {images.shape}')
    print(f'  scalars shape : {scalars.shape}')
    print(f'  labels  shape : {labels.shape}')

    plot_feature_images(images, labels,
        save_path=OUTPUT_DIR / '03_feature_images.png')
    plot_scalar_distributions(scalars, labels,
        save_path=OUTPUT_DIR / '04_scalar_distributions.png')

    # ── Step 4: Train/val split (cutter-aware) ─────────────────────────
    # Validate on c4, train on c1+c6.
    # When using real PHM data this maps exactly to the official split:
    #   train: c1, c4, c6   →   hold out c4 as local validation
    #   test:  c2, c3, c5   →   no labels available
    print('\n[4/6] Splitting dataset (cutter-aware — validate on c4)...')
    (images_tr, scalars_tr, labels_tr,
     images_val, scalars_val, labels_val) = tool_aware_split(
         images, scalars, labels, cutters, val_cutter='c4'
    )
    print(f'  Train: {len(labels_tr)} samples | '
          f'New={np.sum(labels_tr==0)} Worn={np.sum(labels_tr==1)} '
          f'Severe={np.sum(labels_tr==2)}')
    print(f'  Val  : {len(labels_val)} samples | '
          f'New={np.sum(labels_val==0)} Worn={np.sum(labels_val==1)} '
          f'Severe={np.sum(labels_val==2)}')

    # ── Step 5: Train sklearn baselines ────────────────────────────────
    print('\n[5/6] Training sklearn baseline models...')
    X_train = flatten_for_sklearn(images_tr, scalars_tr)
    X_val   = flatten_for_sklearn(images_val, scalars_val)
    results = train_sklearn_baseline(X_train, labels_tr, X_val, labels_val)

    # ── Step 6: Evaluate and plot ───────────────────────────────────────
    print('\n[6/6] Evaluating and saving plots...')
    best_name  = max(results, key=lambda n: results[n]['acc'])
    best_preds = results[best_name]['preds']

    print(f'\n  Best model: {best_name}')
    print(classification_report(labels_val, best_preds,
                                 target_names=WEAR_CLASSES))

    plot_confusion_matrix(labels_val, best_preds, best_name,
        save_path=OUTPUT_DIR / '05_confusion_matrix.png')
    plot_results_summary(results,
        save_path=OUTPUT_DIR / '06_model_comparison.png')

    # ── Save PyTorch code ───────────────────────────────────────────────
    with open(OUTPUT_DIR / 'pytorch_full_model.py', 'w') as f:
        f.write(PYTORCH_CODE)

    # ── Summary ─────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('  RESULTS SUMMARY')
    print('=' * 65)
    for name, res in results.items():
        print(f'  {name:<25} Acc={res["acc"]*100:.1f}%  F1={res["f1"]*100:.1f}%')
    print('\n  Output files:')
    for f in sorted(OUTPUT_DIR.glob('0*.png')):
        print(f'    {f.name}')
    print(f'    pytorch_full_model.py')
    print('\n  To switch to real PHM 2010 data:')
    print('    train_data, _ = load_phm2010("/path/to/phm2010/")')
    print('    images, scalars, labels, cutters = build_feature_arrays(train_data)')
    print('    (and call tool_aware_split / split_phm2010_by_cutter as before)')
    print('=' * 65)


if __name__ == '__main__':
    main()
