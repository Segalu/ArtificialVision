"""
inference.py
============
Use a saved tool_wear_model.pth to predict wear class on new cuts.

Three usage patterns:
    A) Predict from a raw CSV file (same format as PHM 2010)
    B) Predict from numpy arrays   (from your smart meter / DAQ)
    C) Batch predict over a folder of CSV files

Run from command line:
    python inference.py --model tool_wear_model.pth --csv /path/to/c_1_042.csv
    python inference.py --model tool_wear_model.pth --folder /path/to/c2/c2/
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# ── Imports from your project ─────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from tool_wear_detection import (
    FS,
    CUT_COLUMNS,
    preprocess_signal,
    make_spectrogram_image,
    compute_scalar_features,
    _synthetic_voltage,
)
from pytorch_full_model import ToolWearClassifier

# ─────────────────────────────────────────────────────────────────────────────
WEAR_CLASSES = ['New (<100µm)', 'Worn (100–200µm)', 'Severe (>200µm)']
NUM_SCALARS  = 23   # match whatever your training used


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(pth_path, num_scalars=NUM_SCALARS, device=None):
    """
    Load a saved ToolWearClassifier from a .pth file.

    Parameters
    ----------
    pth_path    : str or Path — path to tool_wear_model.pth
    num_scalars : int — must match what was used during training (default 23)
    device      : 'cuda' | 'cpu' | None (auto-detect)

    Returns
    -------
    model  : ToolWearClassifier in eval mode
    device : torch.device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model = ToolWearClassifier(num_classes=3, num_scalars=num_scalars)
    state = torch.load(pth_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f'Model loaded from {pth_path}')
    print(f'  Device      : {device}')
    print(f'  Num scalars : {num_scalars}')
    return model, device


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT — SINGLE CUT
# ═══════════════════════════════════════════════════════════════════════════════

def predict(model, device, current, voltage=None, fs=FS,
            force=None, vib=None, verbose=True):
    """
    Predict wear class for a single cut.

    Parameters
    ----------
    model   : loaded ToolWearClassifier
    device  : torch.device
    current : 1-D numpy array
                - PHM data     → ae_spindle column
                - Your data    → current from smart meter (amps)
    voltage : 1-D numpy array or None
                - PHM data     → pass None, a proxy will be synthesised
                - Your data    → voltage from smart meter (volts)
    fs      : int — sampling rate
                - PHM 2010     → 50_000 (default)
                - Your meter   → whatever your meter uses, e.g. 10_000
    force   : (N,3) numpy array or None — force_x/y/z if available
    vib     : (N,3) numpy array or None — vib_x/y/z if available

    Returns
    -------
    pred    : int         — 0=New  1=Worn  2=Severe
    probs   : np.ndarray  — softmax probabilities, shape (3,)
    label   : str         — human-readable class name
    """
    # Synthesise voltage proxy if not provided (PHM case)
    if voltage is None:
        voltage = _synthetic_voltage(current.astype(np.float32), fs=fs)

    # Preprocess
    current, voltage = preprocess_signal(
        current.astype(np.float32),
        voltage.astype(np.float32),
        fs
    )

    # Feature extraction
    img     = make_spectrogram_image(current, voltage, fs)
    scalars = compute_scalar_features(current, voltage, fs,
                                       force=force, vib=vib)

    # Pad or trim scalars to match model's expected size
    expected = NUM_SCALARS
    if len(scalars) < expected:
        scalars = np.pad(scalars, (0, expected - len(scalars)))
    elif len(scalars) > expected:
        scalars = scalars[:expected]

    # Forward pass
    img_t = torch.FloatTensor(img).unsqueeze(0).to(device)       # (1,3,64,64)
    sc_t  = torch.FloatTensor(scalars).unsqueeze(0).to(device)   # (1, NUM_SCALARS)

    with torch.no_grad():
        logits = model(img_t, sc_t)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred   = int(probs.argmax())

    if verbose:
        _print_result(pred, probs)

    return pred, probs, WEAR_CLASSES[pred]


def _print_result(pred, probs):
    print(f'\n  ┌─ Prediction ──────────────────────────┐')
    print(f'  │  Result     : {WEAR_CLASSES[pred]:<25}│')
    print(f'  │  Confidence : {probs[pred]*100:.1f}%{" "*(24 - len(f"{probs[pred]*100:.1f}%"))}│')
    print(f'  ├───────────────────────────────────────┤')
    for i, cls in enumerate(WEAR_CLASSES):
        bar   = '█' * int(probs[i] * 24)
        blank = '░' * (24 - len(bar))
        marker = ' ◄' if i == pred else '  '
        print(f'  │  {cls:<22} {probs[i]*100:5.1f}%{marker}│')
        print(f'  │  {bar}{blank}         │')
    print(f'  └───────────────────────────────────────┘')


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT — FROM PHM-STYLE CSV FILE
# ═══════════════════════════════════════════════════════════════════════════════

def predict_from_csv(model, device, csv_path, fs=FS, verbose=True):
    """
    Predict wear class from a single PHM-style cut CSV file.
    Expects 7 columns: force_x/y/z, vib_x/y/z, ae_spindle (no header).

    Example:
        pred, probs, label = predict_from_csv(model, device, 'c_1_042.csv')
    """
    import pandas as pd

    df = pd.read_csv(csv_path, header=None)

    if df.shape[1] == len(CUT_COLUMNS):
        df.columns = CUT_COLUMNS
        current = df['ae_spindle'].values.astype(np.float32)
        force   = df[['force_x', 'force_y', 'force_z']].values.astype(np.float32)
        vib     = df[['vib_x',   'vib_y',   'vib_z']].values.astype(np.float32)
    elif df.shape[1] == 1:
        # Single-column file — treat as raw current signal
        current = df.iloc[:, 0].values.astype(np.float32)
        force, vib = None, None
    else:
        # Assume last column is the signal
        current = df.iloc[:, -1].values.astype(np.float32)
        force, vib = None, None

    if verbose:
        print(f'\nPredicting: {Path(csv_path).name}  ({len(current):,} samples @ {fs} Hz)')

    return predict(model, device, current, voltage=None, fs=fs,
                   force=force, vib=vib, verbose=verbose)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT — BATCH OVER A FOLDER
# ═══════════════════════════════════════════════════════════════════════════════

def predict_folder(model, device, folder_path, fs=FS, pattern='*.csv'):
    """
    Predict wear class for every CSV in a folder.
    Prints a summary table and returns a list of results.

    Example:
        results = predict_folder(model, device, '/path/to/c2/c2/')
    """
    folder  = Path(folder_path)
    files   = sorted(folder.glob(pattern))

    if not files:
        print(f'No files matching {pattern} in {folder}')
        return []

    print(f'\nBatch prediction: {len(files)} files in {folder}')
    print(f'{"File":<20} {"Prediction":<25} {"New":>7} {"Worn":>7} {"Severe":>8}')
    print('─' * 72)

    results = []
    for f in files:
        pred, probs, label = predict_from_csv(model, device, f,
                                               fs=fs, verbose=False)
        marker = ['○', '◑', '●'][pred]   # visual wear indicator
        print(f'{f.name:<20} {marker} {label:<23} '
              f'{probs[0]*100:6.1f}% {probs[1]*100:6.1f}% {probs[2]*100:7.1f}%')
        results.append({
            'file':  f.name,
            'pred':  pred,
            'label': label,
            'probs': probs,
        })

    # Summary
    preds = [r['pred'] for r in results]
    print('─' * 72)
    print(f'{"TOTAL":<20} New={preds.count(0)}  '
          f'Worn={preds.count(1)}  Severe={preds.count(2)}')

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT — FROM YOUR SMART METER (numpy arrays)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_from_arrays(model, device, current, voltage, fs=10_000):
    """
    Predict from raw numpy arrays — use this inside your DAQ loop.

    Parameters
    ----------
    current : numpy array — current signal from your smart meter (amps)
    voltage : numpy array — voltage signal from your smart meter (volts)
    fs      : int         — your meter's sampling rate (Hz)

    Example:
        # Inside your data acquisition loop:
        current = meter.read_current(duration=2.0)   # your meter SDK
        voltage = meter.read_voltage(duration=2.0)
        pred, probs, label = predict_from_arrays(model, device,
                                                  current, voltage,
                                                  fs=10_000)
        print(f'Cut wear status: {label}')
    """
    print(f'\nPredicting from arrays  ({len(current):,} samples @ {fs} Hz)')
    return predict(model, device, current, voltage=voltage, fs=fs,
                   force=None, vib=None, verbose=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — command line interface
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Tool wear inference')
    p.add_argument('--model',  required=True,
                   help='Path to tool_wear_model.pth')
    p.add_argument('--csv',    default=None,
                   help='Predict a single CSV cut file')
    p.add_argument('--folder', default=None,
                   help='Batch predict all CSVs in a folder')
    p.add_argument('--fs',     default=50_000, type=int,
                   help='Sampling rate (default: 50000 for PHM, '
                        'use your meter rate for your own data)')
    p.add_argument('--scalars', default=NUM_SCALARS, type=int,
                   help=f'Number of scalar features (default: {NUM_SCALARS})')
    return p.parse_args()


def main():
    # HOW TO RUN
    # python inference.py --model tool_wear_model.pth --folder / path / to / c2 / c2 /
    args = parse_args()

    # Load model
    model, device = load_model(args.model, num_scalars=args.scalars)

    if args.csv:
        predict_from_csv(model, device, args.csv, fs=args.fs)

    elif args.folder:
        predict_folder(model, device, args.folder, fs=args.fs)

    else:
        # No input given — run a quick self-test with synthetic data
        print('\nNo --csv or --folder given. Running self-test...')
        from tool_wear_detection import _tool_current, _synthetic_voltage

        for name, wear_mm in [('New', 0.05), ('Worn', 0.15), ('Severe', 0.28)]:
            current = _tool_current(0.25, FS, wear_mm, 1000, 200, 1.0, noise_seed=42)
            voltage = _synthetic_voltage(current, FS)
            print(f'\n── {name} tool (simulated {wear_mm*1000:.0f}µm wear) ──')
            predict(model, device, current, voltage, fs=FS)


if __name__ == '__main__':
    main()
