# ─────────────────────────────────────────────────────────────────────────────
# Full PyTorch implementation — run on your own machine with a GPU
# pip install torch torchvision timm
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
try:
    import timm  # noqa: F401  — imported for downstream use; safe if missing
except ImportError:
    timm = None
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


class HybridToolWearDataset(Dataset):
    """
    Dataset class combining spectrogram images with wavelet-based features.
    """
    def __init__(self, spectrogram_images, wavelet_images=None, scalars=None, labels=None):
        """
        Initialize the dataset

        Parameters:
        ----------
        spectrogram_images : torch.Tensor
            Spectrogram images from existing pipeline
        wavelet_images : torch.Tensor
            Voltage/current wavelet features
        scalars : torch.Tensor
            Scalar features from existing pipeline
        labels : torch.Tensor
            Class labels
        """
        self.spectrogram_images = spectrogram_images
        self.wavelet_images = wavelet_images
        self.scalars = scalars
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Get item for index"""
        if self.wavelet_images is not None:
            # Return both spectrogram and wavelet images
            return (
                self.spectrogram_images[idx],
                self.wavelet_images[idx],
                self.scalars[idx],
                self.labels[idx]
            )
        else:
            # Fallback to only spectrogram
            return (
                self.spectrogram_images[idx],
                self.scalars[idx],
                self.labels[idx]
            )


class ToolWearClassifier(nn.Module):
    """
    Hybrid model with multiple analysis branches:
      1. CNN branch — spectrogram images (existing)
      2. Wavelet branch — voltage/current wavelet features (new)
      3. MLP branch — physics-based scalar features (existing)
      4. Fusion head — combines all branches
    """
    def __init__(self, num_classes=3, num_scalars=NUM_SCALARS_DEFAULT):
        super().__init__()

        # Standard CNN branch (existing, using spectrogram images)
        self.cnn_branch = nn.ModuleDict()
        self.cnn_branch['spectrogram'] = nn.Sequential(
            # Extract features from spectrogram images
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        cnn_dim = 128  # Output size after spectrogram CNN

        # New CNN branch for wavelet features
        self.cnn_branch['wavelet'] = nn.ModuleDict()
        self.cnn_branch['wavelet']['conv1'] = nn.Sequential(
            # Adaptive layers for wavelet coefficients
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        wavelet_dim = 32  # Output size after wavelet CNN

        # MLP branch for scalar features (existing)
        self.mlp = nn.Sequential(
            nn.Linear(num_scalars, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Fusion classifier - combines all three branches
        combined_features = cnn_dim + wavelet_dim + 64  # Combined feature size
        self.fusion = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, spectrogram_img, wavelet_img=None, scalars=None):
        """Forward pass through the hybrid model.

        Parameters:
        -----------
        spectrogram_img : torch.Tensor
            Spectrogram image from the original pipeline
        wavelet_img : torch.Tensor (optional)
            Voltage/current wavelet features from the new branch
        scalars : torch.Tensor (optional)
            Scalar features

        Returns:
        --------
        torch.Tensor
            Classification output logits
        """
        # Process spectrogram branch (always required)
        spectrogram_features = self.cnn_branch['spectrogram'](spectrogram_img)

        # Process wavelet branch (optional)
        if wavelet_img is not None:
            wavelet_features = self.cnn_branch['wavelet']['conv1'](wavelet_img)
        else:
            # If no wavelet image, create zero tensor of the right shape
            batch_size = spectrogram_features.shape[0]
            wavelet_features = torch.zeros(batch_size, 32, device=spectrogram_features.device)

        # Process scalar features (always required)
        if scalars is not None:
            mlp_features = self.mlp(scalars)
        else:
            # Create zero tensor for scalars if not provided
            batch_size = spectrogram_features.shape[0]
            mlp_features = torch.zeros(batch_size, 64, device=spectrogram_features.device)

        # Combine features from all branches
        combined = torch.cat([
            spectrogram_features,
            wavelet_features,
            mlp_features
        ], dim=1)

        return self.fusion(combined)


def train_pytorch_with_wavelets(images_tr, wavelet_images_tr, scalars_tr, labels_tr,
                                 images_val, wavelet_images_val, scalars_val, labels_val,
                                 epochs=40, batch_size=32, lr=1e-3,
                                 num_workers=0):
    """
    Train the hybrid CNN+MLP+Wavelet model.

    Parameters:
    ----------
    images_tr : numpy array
        Spectrogram images for training
    wavelet_images_tr : numpy array
        Voltage/current wavelet images for training (optional)
    scalars_tr : numpy array
        Scalar features for training
    labels_tr : numpy array
        Labels for training
    images_val : numpy array
        Spectrogram images for validation
    wavelet_images_val : numpy array
        Voltage/current wavelet images for validation (optional)
    scalars_val : numpy array
        Scalar features for validation
    labels_val : numpy array
        Labels for validation
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    lr : float
    Learning rate
    num_workers : int
        Number of data loading workers
    """
    # Train the hybrid CNN+MLP model.
    #
    # num_workers=0  is the safe default — avoids multiprocessing issues on
    # Windows and some Linux setups. Set to 4 if you're on Linux with plenty
    # of RAM and want faster data loading.
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
            loss = criterion(model(imgs, None, scs), lbls)
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
                preds   = model(imgs, None, scs).argmax(1)
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
    # Fine-tune a pretrained model on YOUR milling machine data.
    #
    # Stage 1: Freeze CNN backbone — only train fusion head (fast adaptation)
    # Stage 2: Unfreeze all       — end-to-end fine-tuning at very low LR
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