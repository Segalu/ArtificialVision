"""
run_pytorch_model.py
====================
End-to-end example: load data → train PyTorch model → fine-tune → inference.

Usage
-----
    # 1. With synthetic data (no PHM download needed — great for testing):
    python run_pytorch_model.py --mode synthetic

    # 2. With real PHM 2010 data:
    python run_pytorch_model.py --mode phm --data /path/to/phm2010/

    # 3. Fine-tune on your own recordings + run inference on a new cut:
    python run_pytorch_model.py --mode finetune --data /path/to/phm2010/ \
                                --recordings /path/to/your/recordings/

Requirements
------------
    pip install torch torchvision timm
    (numpy, scipy, librosa, scikit-image already needed by tool_wear_detection.py)
"""

import argparse
import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Ensure tool_wear_detection.py is importable ──────────────────────────────
# Place this script in the same folder as tool_wear_detection.py, or adjust:
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from tool_wear_detection import (
    FS,
    generate_synthetic_dataset,
    load_phm2010,
    load_own_recordings,
    save_recording,
    RECORDING_TEMPLATE,
    build_feature_arrays,
    tool_aware_split,
    to_flat_list,
    preprocess_signal,
    make_spectrogram_image,
    compute_scalar_features,
)

# ── Try importing PyTorch model (graceful error if torch not installed) ───────
try:
    from pytorch_full_model import (
        ToolWearClassifier,
        ToolWearDataset,
        train_pytorch,
        finetune,
    )
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f'[ERROR] Could not import pytorch_full_model: {e}')
    print('        Run: pip install torch torchvision timm')
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
WEAR_CLASSES  = ['New (<100µm)', 'Worn (100–200µm)', 'Severe (>200µm)']
MODEL_PATH    = SCRIPT_DIR / 'tool_wear_model.pth'
OUTPUT_DIR    = SCRIPT_DIR / 'pytorch_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_SCALARS   = 22     # 15 signal + 7 force/vib (zeros when not available)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(mode, phm_dir=None):
    """
    Returns a flat list of record dicts ready for build_feature_arrays().

    mode='synthetic' — generates PHM-like data, no download needed
    mode='phm'       — loads real PHM 2010 training cutters (c1, c4, c6)
    """
    if mode == 'synthetic':
        print('[1/5] Generating synthetic dataset...')
        dataset = generate_synthetic_dataset(
            n_tools=3, cuts_per_tool=150, duration_s=0.25, fs=FS, verbose=True
        )
        cutter_map = {1: 'c1', 2: 'c4', 3: 'c6'}
        for rec in dataset:
            rec['cutter'] = cutter_map[rec['tool_id']]
        return to_flat_list(dataset)

    elif mode == 'phm':
        if phm_dir is None:
            raise ValueError('--data path required for --mode phm')
        print(f'[1/5] Loading PHM 2010 from {phm_dir}...')
        train_data, _ = load_phm2010(phm_dir, verbose=True)
        return to_flat_list(train_data)

    else:
        raise ValueError(f'Unknown mode: {mode}')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — EXTRACT FEATURES + SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def extract_and_split(dataset, val_cutter='c4'):
    """
    Feature extraction → cutter-aware train/val split.

    val_cutter: which training cutter to hold out as validation
                PHM options: 'c1', 'c4', 'c6'
                synthetic:   'c1', 'c4', 'c6' (mapped above)
    """
    print('\n[2/5] Extracting features...')
    images, scalars, labels, cutters = build_feature_arrays(dataset, fs=FS)

    print(f'  images  : {images.shape}')
    print(f'  scalars : {scalars.shape}')
    print(f'  labels  : {labels.shape}  '
          f'(New={np.sum(labels==0)}  '
          f'Worn={np.sum(labels==1)}  '
          f'Severe={np.sum(labels==2)})')

    print(f'\n  Splitting — train on all except {val_cutter}, '
          f'validate on {val_cutter}')
    (img_tr, sc_tr, lbl_tr,
     img_val, sc_val, lbl_val) = tool_aware_split(
         images, scalars, labels, cutters, val_cutter=val_cutter
    )
    print(f'  Train : {len(lbl_tr):4d} samples')
    print(f'  Val   : {len(lbl_val):4d} samples')

    return (img_tr, sc_tr, lbl_tr), (img_val, sc_val, lbl_val)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_num_scalars(num_scalars):
    """
    Monkey-patch ToolWearClassifier to use the correct scalar size.
    This makes run_pytorch_model.py work regardless of which version
    of pytorch_full_model.py is installed (old hardcoded 15 or new auto).
    """
    import pytorch_full_model as _m
    import torch.nn as nn

    original_init = _m.ToolWearClassifier.__init__

    def patched_init(self, num_classes=3, num_scalars=num_scalars):
        # Call with the correct scalar count
        original_init(self, num_classes=num_classes, num_scalars=num_scalars)

    _m.ToolWearClassifier.__init__ = patched_init


def train(train_split, val_split, epochs=40):
    """
    Train the hybrid EfficientNet-B0 + MLP model and save weights.
    """
    img_tr, sc_tr, lbl_tr   = train_split
    img_val, sc_val, lbl_val = val_split

    # Detect scalar size from data and patch the model class before training.
    # This is the robust fix for the num_scalars mismatch between versions.
    num_scalars = sc_tr.shape[1]
    print(f'\n[3/5] Training PyTorch model ({epochs} epochs)...')
    print(f'  Scalar features detected: {num_scalars}')
    _patch_num_scalars(num_scalars)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')

    model, history = train_pytorch(
        img_tr, sc_tr, lbl_tr,
        img_val, sc_val, lbl_val,
        epochs=epochs,
        batch_size=32,
        lr=1e-3,
    )

    # Save weights
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'\n  Model saved → {MODEL_PATH}')

    # Plot training history
    _plot_history(history)

    return model


def _plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Training History', fontweight='bold')

    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], '#3498db', lw=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')

    ax2.plot(epochs, [v * 100 for v in history['val_acc']], '#e74c3c', lw=2)
    ax2.axhline(90, color='grey', ls=':', lw=1, alpha=0.6)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    path = OUTPUT_DIR / 'training_history.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  History plot saved → {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 (OPTIONAL) — FINE-TUNE ON YOUR OWN MACHINE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def run_finetune(model, recordings_dir):
    """
    Fine-tune a pretrained model on recordings from your own milling machine.

    recordings_dir should contain .npz + _meta.json files saved with
    save_recording() from tool_wear_detection.py.
    """
    print(f'\n[4/5] Fine-tuning on your recordings from {recordings_dir}...')

    my_data = load_own_recordings(recordings_dir, fs=10_000)
    if not my_data:
        print('  No recordings found — skipping fine-tune.')
        return model

    my_images, my_scalars, my_labels, _ = build_feature_arrays(my_data)
    print(f'  Loaded {len(my_labels)} recordings  '
          f'(New={np.sum(my_labels==0)}  '
          f'Worn={np.sum(my_labels==1)}  '
          f'Severe={np.sum(my_labels==2)})')

    model = finetune(
        model,
        my_images, my_scalars, my_labels,
        epochs_frozen=5,    # Stage 1: train head only
        epochs_full=15,     # Stage 2: end-to-end at low LR
        lr_full=1e-5,
    )

    ft_path = SCRIPT_DIR / 'tool_wear_finetuned.pth'
    torch.save(model.state_dict(), ft_path)
    print(f'  Fine-tuned model saved → {ft_path}')
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — INFERENCE ON A SINGLE NEW CUT
# ═══════════════════════════════════════════════════════════════════════════════

def predict_cut(model, current, voltage, fs=FS, force=None, vib=None):
    """
    Predict wear class for a single cut recording.

    Parameters
    ----------
    model   : trained ToolWearClassifier
    current : 1-D numpy array — from your smart meter (amps) or ae_spindle
    voltage : 1-D numpy array — from your smart meter (volts)
    fs      : sampling rate of YOUR meter (e.g. 10_000 for a 10kHz meter)
    force   : (N,3) array or None — if you also have a force sensor
    vib     : (N,3) array or None — if you also have an accelerometer

    Returns
    -------
    pred  : int   — 0=New, 1=Worn, 2=Severe
    probs : array — softmax probabilities for each class
    """
    model.eval()
    device = next(model.parameters()).device

    # Preprocess
    current, voltage = preprocess_signal(current, voltage, fs)

    # Build image + scalar features (same pipeline as training)
    img     = make_spectrogram_image(current, voltage, fs)
    scalars = compute_scalar_features(current, voltage, fs,
                                       force=force, vib=vib)

    img_t = torch.FloatTensor(img).unsqueeze(0).to(device)      # (1,3,64,64)
    sc_t  = torch.FloatTensor(scalars).unsqueeze(0).to(device)  # (1,22)

    with torch.no_grad():
        logits = model(img_t, sc_t)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred   = int(probs.argmax())

    return pred, probs


def demo_inference(model):
    """
    Run predict_cut() on three synthetic signals (one per wear class)
    to verify the model works end-to-end.
    """
    print('\n[5/5] Demo inference on synthetic cuts...')
    from tool_wear_detection import _tool_current, _synthetic_voltage

    test_cases = [
        ('New tool',    0.05,  1000, 200, 1.0),
        ('Worn tool',   0.15,  1000, 200, 1.0),
        ('Severe tool', 0.28,  1000, 200, 1.0),
    ]

    for name, wear_mm, rpm, feed, depth in test_cases:
        current = _tool_current(0.25, FS, wear_mm, rpm, feed, depth,
                                 noise_seed=999)
        voltage = _synthetic_voltage(current, FS)

        pred, probs = predict_cut(model, current, voltage, fs=FS)

        print(f'\n  {name} (true wear ~{wear_mm*1000:.0f}µm)')
        print(f'    Prediction : {WEAR_CLASSES[pred]}')
        print(f'    Confidence : {probs[pred]*100:.1f}%')
        for i, cls in enumerate(WEAR_CLASSES):
            bar = '█' * int(probs[i] * 20)
            print(f'    {cls:<22} {probs[i]*100:5.1f}%  {bar}')


# ═══════════════════════════════════════════════════════════════════════════════
# HOW TO SAVE YOUR OWN RECORDINGS (reference example)
# ═══════════════════════════════════════════════════════════════════════════════

def example_save_recording(save_dir='/tmp/my_recordings/'):
    """
    Shows how to save a single recording from your smart meter.
    Call this inside your DAQ loop after each cut.

    In practice:
        - current/voltage come from your smart meter SDK or DAQ card
        - wear_label is set by you (0=New, 1=Worn, 2=Severe)
        - Update metadata with the actual machining parameters
    """
    # Simulate a reading from your smart meter
    duration_s   = 2.0
    fs_meter     = 10_000          # your meter's sampling rate (Hz)
    n            = int(duration_s * fs_meter)
    t            = np.arange(n) / fs_meter

    # Replace these with real readings from your meter:
    current = 8.5 * np.sin(2 * np.pi * 60 * t) + np.random.normal(0, 0.1, n)
    voltage = 120 * np.sqrt(2) * np.sin(2 * np.pi * 60 * t)

    metadata = {
        **RECORDING_TEMPLATE,
        'recording_id':     'REC_0001',
        'timestamp':        '2026-03-10T09:30:00',
        'tool_id':          'ENDMILL_6MM_001',
        'tool_passes':      42,
        'wear_label':       0,           # 0=New  1=Worn  2=Severe
        'flank_wear_mm':    None,        # fill in if you measured it
        'spindle_rpm':      1000,
        'feed_mm_min':      200,
        'depth_mm':         1.0,
        'material':         'Al6061',
        'coolant':          True,
        'sampling_rate_hz': fs_meter,
        'notes':            'First cut with new endmill',
    }

    save_recording(
        current    = current.astype(np.float32),
        voltage    = voltage.astype(np.float32),
        wear_label = 0,
        metadata   = metadata,
        save_dir   = save_dir,
    )
    print(f'  Saved example recording to {save_dir}')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Tool wear PyTorch training pipeline')
    p.add_argument('--mode',
                   choices=['synthetic', 'phm', 'finetune'],
                   default='synthetic',
                   help='synthetic: no data needed | '
                        'phm: use PHM 2010 | '
                        'finetune: PHM pretraining + your recordings')
    p.add_argument('--data',       default=None,
                   help='Path to PHM 2010 root folder (required for phm/finetune)')
    p.add_argument('--recordings', default=None,
                   help='Path to your own recordings folder (for finetune mode)')
    p.add_argument('--val-cutter', default='c4',
                   choices=['c1', 'c4', 'c6'],
                   help='Which training cutter to use as validation set')
    p.add_argument('--epochs',     default=40, type=int,
                   help='Training epochs (default: 40)')
    p.add_argument('--load-model', default=None,
                   help='Path to saved .pth — skip training, go straight to inference')
    return p.parse_args()


def main():
    args = parse_args()

    print('=' * 65)
    print('  Tool Wear — PyTorch Training Pipeline')
    print(f'  Mode    : {args.mode}')
    print(f'  Device  : {"CUDA" if torch.cuda.is_available() else "CPU"}')
    print('=' * 65)

    # ── Load or skip training ─────────────────────────────────────────
    if args.load_model:
        # Skip training entirely — load existing weights
        print(f'\nLoading saved model from {args.load_model}...')
        _patch_num_scalars(NUM_SCALARS)
        model = ToolWearClassifier(num_classes=3, num_scalars=NUM_SCALARS)
        model.load_state_dict(torch.load(args.load_model, map_location='cpu'))
        model.eval()
        print('  Model loaded.')
    else:
        # ── Step 1: Data ──────────────────────────────────────────────
        phm_mode = 'phm' if args.mode in ('phm', 'finetune') else 'synthetic'
        dataset  = load_data(phm_mode, phm_dir=args.data)

        # ── Step 2: Features + split ──────────────────────────────────
        train_split, val_split = extract_and_split(
            dataset, val_cutter=args.val_cutter
        )

        # ── Step 3: Train ─────────────────────────────────────────────
        model = train(train_split, val_split, epochs=args.epochs)

    # ── Step 4: Fine-tune (finetune mode only) ────────────────────────
    if args.mode == 'finetune':
        if args.recordings is None:
            print('\n[4/5] --recordings not provided — skipping fine-tune.')
            print('      Collect recordings with save_recording() first,')
            print('      then re-run with --recordings /path/to/recordings/')
        else:
            model = run_finetune(model, args.recordings)

    # ── Step 5: Demo inference ────────────────────────────────────────
    demo_inference(model)

    print('\n' + '=' * 65)
    print('  Done.')
    print(f'  Model weights : {MODEL_PATH}')
    print(f'  Output plots  : {OUTPUT_DIR}/')
    print()
    print('  To run inference on your own cut:')
    print('    from run_pytorch_model import predict_cut')
    print('    from pytorch_full_model import ToolWearClassifier')
    print('    import torch')
    print()
    print('    model = ToolWearClassifier(num_classes=3, num_scalars=22)')
    print('    model.load_state_dict(torch.load("tool_wear_model.pth"))')
    print('    pred, probs = predict_cut(model, current, voltage, fs=10_000)')
    print('=' * 65)


if __name__ == '__main__':
    main()