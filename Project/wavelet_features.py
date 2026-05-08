"""
Voltage/Current Wavelet Feature Extraction for Tool Wear Detection
============================================================

This module provides wavelet-based time-frequency feature extraction for
the voltage/current signals in tool wear detection. The wavelet branch
complements the existing spectrogram-based approach by providing
better time-localization for transient wear signals.

Usage:
------
- Use the `VoltageCurrentWaveletExtractor` to extract features from voltage/current
  signals alongside the existing pipeline
- Use the `VoltageCurrentWaveletCNN` class as a drop-in replacement for
the current spectrogram CNN branch

Author: Claude Code Assistant
"""
import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize

# Set up for wavelet transforms
FS = 50_000  # PHM 2010 sampling rate (Hz)
WAVELET_TYPE = "morl"  # Default Morlet wavelet for time-frequency analysis

class VoltageCurrentWaveletExtractor:
    """
    Extract wavelet-based time-frequency features from voltage/current signals.

    Parameters:
    -----------
    fs : int
        Sampling frequency (Hz)
    wavelet_type : str
        Type of wavelet to use (default: "morl" - Morlet)
    wavelet_freq : int
        Frequency of the central wavelet (Hz) - typically a fraction of FS
    n_cycles : int
        Number of cycles of the central wavelet
    img_size : int
        Target image size for CNN input
    """

    def __init__(self, fs=FS, wavelet_type=WAVELET_TYPE, wavelet_freq=None,
                 n_cycles=10, img_size=64):
        self.fs = fs
        self.wavelet_type = wavelet_type
        self.wavelet_freq = wavelet_freq or (fs // 60)  # Default to 60 Hz
        self.n_cycles = n_cycles
        self.img_size = img_size

        # Pre-compute wavelet parameters
        self.wavelet_kwargs = {
            'wavelet': wavelet_type,
            'sampling': self.fs,
            'bounds': ['maximum'],
            'centered': True
        }

        # Compute wavelet coefficients for a test signal to get expected ranges
        self._test_coeffs = None

    def _compute_wavelet_coefficients(self, signal):
        """Compute Continuous Wavelet Transform coefficients for a signal"""
        # Extract time segments for wavelet analysis
        samples_per_cycle = int(self.fs / 60)  # Using 60Hz as base cycle
        n_samples = samples_per_cycle * self.n_cycles

        # Apply wavelet transform
        wavelet_result = signal.cwt(wt=signal[:n_samples], **self.wavelet_kwargs)

        # Convert wavelet scales to time-frequency domain (in Hz)
        scales = wavelet_result['scales']
        # Scales to frequencies
        freq_scale = self.fs / scales

        return wavelet_result, freq_scale

    def _convert_to_image(self, wavelet_result, freq_scale):
        """Convert wavelet coefficients to a 2D image for CNN processing"""
        # Get the time and frequency grids
        times = wavelet_result['positions'] * (self.fs / self.n_cycles)  # Convert to time
        freqs = freq_scale

        # Create a 2D representation
        # Interpolate coefficients to create an image-like representation

        # Pad time and freq with zeros to create image-like dimensions
        n_time = len(times)
        n_freq = len(freqs)

        # For Morlet wavelet, the coefficients are in the 'coeffs' field
        coeffs = wavelet_result['coeffs']

        # Create a time-frequency heatmap
        # Convert to 2D matrix of shape (n_time, n_freq)
        time_freq_coeffs = np.zeros((n_time, n_freq))

        # Normalize time and frequency to image bounds
        # Reshape coefficients to match time and frequency bins
        for i, coeff in enumerate(coeffs):
            if i < len(times) and i < len(freqs):
                time_idx = min(i, n_time - 1)
                freq_idx = min(len(freqs) - 1 - i, n_freq - 1)  # Invert to match freq order
                time_freq_coeffs[time_idx, freq_idx] = np.abs(coeff)

        # Normalize and rescale to match spectrogram dimensions
        mn, mx = time_freq_coeffs.min(), time_freq_coeffs.max()
        if mx - mn > 0:  # Avoid division by zero
            norm_coeffs = (time_freq_coeffs - mn) / (mx - mn)
        else:
            norm_coeffs = np.zeros_like(time_freq_coeffs)

        # Resize to target image size
        if self.img_size != time_freq_coeffs.shape[0] or self.img_size != time_freq_coeffs.shape[1]:
            norm_coeffs = resize(norm_coeffs, (self.img_size, self.img_size),
                                anti_aliasing=True)

        return norm_coeffs

    def extract_features(self, voltage, current):
        """
        Extract wavelet features from voltage/current signals

        Parameters:
        -----------
        voltage : numpy array
            Voltage signal time series
        current : numpy array
            Current signal time series

        Returns:
        --------
        wavelet_image : numpy array
            2D array of wavelet coefficients (size: img_size x img_size)
        """
        # Normalize signals to [-1, 1]
        voltage_n = voltage / (np.max(np.abs(voltage)) + 1e-8)
        current_n = current / (np.max(np.abs(current)) + 1e-8)

        # Compute wavelet coefficients for both voltage and current
        v_wavelet, v_freq = self._compute_wavelet_coefficients(voltage_n)
        c_wavelet, c_freq = self._compute_wavelet_coefficients(current_n)

        # Convert to images
        v_image = self._convert_to_image(v_wavelet, v_freq)
        c_image = self._convert_to_image(c_wavelet, c_freq)

        # Stack them to create 2-channel image (voltage first, then current)
        # Add another channel combining them (mean of both)
        combined_image = np.mean([v_image, c_image], axis=0)
        wavelet_image = np.stack([v_image, c_image, combined_image], axis=0)

        return wavelet_image


class VoltageCurrentWaveletCNN(nn.Module):
    """
    CNN module for processing voltage/current wavelet features

    Parameters:
    -----------
    num_classes : int
        Number of output classes
    in_channels : int
        Number of input channels (2 for voltage/current or 1 for single signal)
    """

    def __init__(self, num_classes=3, in_channels=3):
        super(VoltageCurrentWaveletCNN, self).__init__()

        # Input layers - use small CNN architecture suitable for wavelet images
        self.cnn = nn.Sequential(
            # First convolution to reduce dimensions
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Second convolution
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Third convolution with larger kernel
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Flatten and dense layers
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Classifier layer
        self.classifier = nn.Linear(64, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for the CNN"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Forward pass through the CNN"""
        x = self.cnn(x)
        return self.classifier(x)


# Helper function to convert to tensor format expected by DataLoader
class VoltageCurrentWaveletDataset:
    """
    Dataset class that combines voltage/current wavelet features with other features
    """
    def __init__(self, images, scalars, labels, wavelet_images=None):
        """
        Initialize the dataset

        Parameters:
        -----------
        images : numpy array
            Spectrogram images from existing pipeline
        scalars : numpy array
            Scalar features from existing pipeline
        labels : numpy array
            Class labels
        wavelet_images : numpy array
            Voltage/current wavelet features. If None, use only spectrograms.
        """
        self.spectrogram_images = torch.FloatTensor(images)
        self.scalars = torch.FloatTensor(scalars)
        self.labels = torch.LongTensor(labels)

        if wavelet_images is not None:
            self.wavelet_images = torch.FloatTensor(wavelet_images)
        else:
            self.wavelet_images = None

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
            # Fallback to only spectrogram if no wavelet images
            return (
                self.spectrogram_images[idx],
                self.scalars[idx],
                self.labels[idx]
            )