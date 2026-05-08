# V-JEPA vs. Wavelet-CNN: A Comparison for Tool Wear Detection

This document analyses the V-JEPA 2.1 linear probe experiment (`jepa_probe.ipynb`) against the project's existing Wavelet + CNN hybrid approach, covering architectural trade-offs, empirical results, and practical guidance on when each method is appropriate.

---

## 1. What Each Approach Does

### Wavelet-CNN Hybrid (original method)

The production model in `pytorch_full_model.py` is a purpose-built, multi-branch classifier:

```
Raw signal (current + voltage, 50 kHz)
    │
    ├─ [CNN branch]      make_spectrogram_image()  →  64×64 Mel spectrogram  →  EfficientNet-B0  →  1280-dim
    ├─ [Wavelet branch]  VoltageCurrentWaveletExtractor  →  64×64 CWT heatmap  →  WaveletCNN  →  N-dim
    └─ [MLP branch]      compute_scalar_features()  →  22 scalars  →  2-layer MLP
                                                                              │
                                                                    Fusion head  →  3-class softmax
```

All three branches are trained end-to-end on PHM 2010 data. The wavelet branch uses a **Continuous Wavelet Transform (Morlet)** which gives adaptive time-frequency resolution — high time resolution at high frequencies, high frequency resolution at low frequencies. This is better suited to transient wear events than the fixed-window STFT used in the spectrogram branch.

### V-JEPA 2.1 Linear Probe (new experiment)

The notebook encodes entire cut sequences as short MP4 videos and passes them through a **frozen** Facebook V-JEPA 2.1 (ViT-L, 326M parameters). A single logistic regression head is then trained on top:

```
Cut signal  →  Mel spectrogram frame (224×224)  →  MP4 video (315 frames)
                                                         │
                               V-JEPA 2.1 frozen encoder (ViT-L, 326M params)
                                                         │
                                   mean-pool patch tokens  →  1024-dim embedding
                                                         │
                                      LogisticRegression  →  3-class prediction
```

No gradient flows through V-JEPA — it is purely a feature extractor. This is intentional: a **linear probe** tests whether the pretrained representations are already useful without any task-specific adaptation.

---

## 2. Empirical Results

### Wavelet-CNN Hybrid (from `comparission/` and README)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| New (<100 µm) | 0.94 | ~0.94 | ~0.94 |
| Worn (100–200 µm) | 0.91 | ~0.91 | ~0.91 |
| Severe (>200 µm) | 0.92 | ~0.92 | ~0.92 |
| **Overall accuracy** | | | **~93%** |

### V-JEPA 2.1 Linear Probe (from `jepa_probe.ipynb`, test cutter: c6)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| New (<100 µm) | 0.447 | 1.000 | 0.618 | 63 |
| Worn (100–200 µm) | 0.856 | 0.656 | 0.743 | 227 |
| Severe (>200 µm) | 0.000 | 0.000 | 0.000 | 25 |
| **Overall accuracy** | | | | **67.3%** |

The 26-percentage-point gap is large and instructive — see Section 4 for a root-cause breakdown.

---

## 3. Advantages of V-JEPA

### 3.1 Zero task-specific training data required
The encoder is frozen. You need only enough labelled cuts to train the logistic regression head — potentially dozens of cuts rather than hundreds. This is valuable when:
- A new machine or cutter type is introduced with limited historical data.
- You want a quick baseline before committing to full model training.

### 3.2 Temporal context is built-in
V-JEPA is a **video foundation model** trained with masked predictive coding over space and time. Its ViT-L backbone has joint spatio-temporal attention across frame patches. In principle, it can detect gradual embedding drift across a wear progression without explicit temporal feature engineering. The embedding drift plot (`jepa_outputs/embedding_drift.png`) shows this works qualitatively — L2 distances between consecutive frames increase near wear-class transitions.

### 3.3 Representations may generalise across machine types
The Wavelet-CNN is trained specifically on PHM 2010 c1/c4/c6 cutter data. Its weights encode assumptions about that signal distribution. V-JEPA's frozen representations are independent of any machining dataset — a model trained on its embeddings may transfer more gracefully to a different mill, spindle speed, or material without retraining the backbone.

### 3.4 Simple to iterate on the head
Swapping logistic regression for a small MLP, an SVM, or a gradient-boosted tree requires no changes to the embedding pipeline (Cells 1–6). The 1024-dim features are cached as numpy arrays and can be reused across many experiments.

### 3.5 No domain-specific feature engineering
The Wavelet-CNN pipeline requires hand-tuned preprocessing: bandpass filter parameters, CWT wavelet type and scale range, scalar feature definitions (RMS, THD, harmonics). V-JEPA takes raw spectrogram frames and figures out its own features.

---

## 4. Disadvantages of V-JEPA

### 4.1 Large domain gap — the single biggest problem
V-JEPA was pretrained on **natural video** (humans, objects, scenes). Machining spectrogram videos are:
- Visually periodic and nearly static (texture changes, not motion).
- Low semantic content by natural-video standards — no object boundaries, no motion flow.
- High-information in frequency patterns that natural video models ignore.

The model's learned priors are trained on entirely different statistics. This is the primary reason for the 26-point accuracy gap — the 1024-dim features encode semantics that are simply less relevant to wear state.

### 4.2 Complete failure on the Severe class
The Severe class (>200 µm) achieves F1=0.00. This is partly a **training distribution problem** — cutter c1 never reaches the Severe class at all (classes {0,1} only), and c4 has very few Severe cuts. The logistic regression never learns a Severe decision boundary. The Wavelet-CNN handles this through class-weighted loss and sees Severe examples from both c4 and c6 during cross-validation.

This is not an inherent V-JEPA limitation — it would be fixed by including Severe examples in training or using class weights — but it highlights that foundation model probes are sensitive to class imbalance in the labelled head dataset.

### 4.3 Token-to-frame mapping is a heuristic
Cell 6 divides `last_hidden_state` tokens evenly across frames in a chunk:

```python
tokens_per_frame = hidden.shape[0] // n_frames_in_chunk
```

V-JEPA processes the full temporal sequence with joint attention — there is no guarantee that the first `T` tokens correspond to frame 1. The correct approach is to pass a single clip and use the full pooled representation, not to slice tokens per frame. This introduces noise into every single embedding.

### 4.4 Heavy compute for a frozen model
V-JEPA ViT-L has 326M parameters. Inference on a 315-frame video requires multiple forward passes (CHUNK_SIZE=32). The Wavelet-CNN achieves 93% accuracy with ~30M parameters trained specifically on this task. The foundation model is approximately 10× larger for substantially worse performance.

### 4.5 Input mismatch: frame-level vs. clip-level
V-JEPA is designed to receive **clips** (multiple frames with temporal context) and produce a single clip-level representation. The notebook treats each frame independently as a single-frame clip, which discards the model's key capability — predicting masked future patches from past context. Running sliding window clips of 8–16 frames centred on each cut would be a more faithful use of the architecture.

### 4.6 Spectrogram-as-video is a round-trip encoding
The pipeline converts a 1D signal → Mel spectrogram → MP4 video → V-JEPA embedding. Each step introduces lossy compression. The Wavelet-CNN branch works directly on the CWT of the raw signal — there is no intermediate video encoding, no codec compression, and no colour-space quantisation.

---

## 5. Head-to-Head Comparison

| Dimension | Wavelet-CNN Hybrid | V-JEPA Linear Probe |
|-----------|-------------------|---------------------|
| Overall accuracy (c6 test) | ~93% | 67.3% |
| Severe class F1 | ~0.92 | 0.00 |
| Model parameters | ~30M (trained) | 326M (frozen) + probe |
| Task-specific training needed | Yes — ~630 labelled cuts | No (probe only needs ~10s of labels) |
| Feature engineering required | Bandpass filter, CWT config, scalar definitions | None |
| Handles class imbalance | Yes (weighted loss) | Depends on probe setup |
| Generalises to new machine types | Requires retraining backbone | Backbone is machine-agnostic |
| Temporal modelling | Explicit (per-cut features) | Latent (joint attention, underused) |
| Inference speed | Fast (small model) | Slow (326M forward pass) |
| Correct use of architecture | Yes — trained end-to-end for this domain | Partial — temporal capability underutilised |

---

## 6. When to Use Each Method

**Use the Wavelet-CNN hybrid when:**
- You have labelled PHM-style data for at least two cutters.
- Accuracy on all three wear classes (especially Severe) is critical.
- You need fast, production-ready inference.
- The signal domain (50 kHz electrical, milling) matches your deployment environment.

**Use V-JEPA (or similar foundation model) when:**
- You are exploring a new machine type with < 50 labelled cuts.
- You want a fast, low-effort baseline before committing to training infrastructure.
- Cross-machine generalisation is more important than peak per-machine accuracy.
- You have GPU budget for 326M-parameter inference and want to fine-tune the full model end-to-end (not just the linear probe — which is the unexplored upgrade path here).

---

## 7. Potential Improvements to the V-JEPA Approach

These changes would close the gap before concluding V-JEPA cannot work for this task:

1. **Fix the token pooling** — pass a single clip and use the full pooled output, not per-frame token slicing.
2. **Use sliding clips** — process 8-frame windows centred on each cut instead of individual frames.
3. **Balance training classes** — add class weights to the logistic regression (`class_weight='balanced'`) to fix the Severe class failure.
4. **Upgrade the probe head** — a 2-layer MLP (1024→256→64→3) with dropout will substantially outperform logistic regression on 1024-dim features.
5. **Fine-tune the encoder** — unfreeze the last 2–4 transformer blocks and fine-tune on PHM data with a small learning rate (~1e-5). This is the most impactful single change.

---

## 8. Conclusion

The Wavelet-CNN hybrid is the correct choice for this dataset: it achieves ~93% accuracy, handles all three wear classes including Severe, and was designed for this exact signal domain. The V-JEPA linear probe experiment is a useful **diagnostic** — it tells us that V-JEPA's frozen representations partially encode wear-relevant information (67.3% is far above the 33% random baseline), but the domain gap between natural video and machining spectrograms prevents it from competing without fine-tuning.

The clearest path forward for V-JEPA is to move from a frozen linear probe to a lightly fine-tuned model — a change that preserves the generalisation benefits of the large pretrained backbone while adapting its representations to the specific statistics of machining signals.