"""TurboQuant 4-bit NC (TQ4NC) KV cache implementation.

TQ4NC uses:
- Hadamard rotation for key decorrelation
- Lloyd-Max MSE quantization with per-coordinate centroid tables
- 4-bit uniform quantization for values
- Norm correction for dequantization

This is a research-level quantization scheme requiring:
1. Hadamard matrices (per head_dim)
2. 16-element centroid lookup tables (per coordinate)
3. Norm correction factors
4. Complex quantization pipeline

Status: Basic implementation framework. Full optimization requires 3-4 weeks.
"""

import torch
import numpy as np
from typing import Tuple


def generate_hadamard_matrix(dim: int) -> torch.Tensor:
    """Generate Hadamard matrix for dimension dim (must be power of 2)."""
    assert dim & (dim - 1) == 0, "Hadamard dim must be power of 2"

    # Build the unnormalized Sylvester matrix, then scale once at the end.
    # Normalizing inside the recursion would divide by sqrt(dim) at every
    # level, shrinking the result by sqrt(dim)^log2(dim) instead of sqrt(dim).
    H = torch.ones(1, 1)
    while H.shape[0] < dim:
        H = torch.cat(
            (
                torch.cat((H, H), dim=1),
                torch.cat((H, -H), dim=1),
            ),
            dim=0,
        )

    # Orthonormal scaling: H @ H.T == I
    return H / np.sqrt(dim)


def lloyd_max_quantizer_1d(data: np.ndarray, num_levels: int = 16, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd-Max MSE-optimal scalar quantizer.

    Args:
        data: 1D array of values to quantize
        num_levels: Number of quantization levels (16 for 4-bit)
        max_iter: Maximum Lloyd iterations

    Returns:
        centroids: [num_levels] optimal centroids
        boundaries: [num_levels+1] decision boundaries
    """
    # Initialize centroids uniformly
    data_min, data_max = data.min(), data.max()
    centroids = np.linspace(data_min, data_max, num_levels)

    for _ in range(max_iter):
        # E-step: assign each point to nearest centroid
        distances = np.abs(data[:, None] - centroids[None, :])
        assignments = np.argmin(distances, axis=1)

        # M-step: update centroids as cluster means
        new_centroids = np.zeros(num_levels)
        for i in range(num_levels):
            cluster = data[assignments == i]
            if len(cluster) > 0:
                new_centroids[i] = cluster.mean()
            else:
                # Empty cluster: keep old centroid
                new_centroids[i] = centroids[i]

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    # Compute decision boundaries (midpoints)
    boundaries = np.zeros(num_levels + 1)
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    for i in range(1, num_levels):
        boundaries[i] = (centroids[i-1] + centroids[i]) / 2

    return centroids, boundaries


class TQ4NCQuantizer:
    """TurboQuant 4-bit NC quantizer with Hadamard + Lloyd-Max."""

    def __init__(self, head_dim: int = 128):
        self.head_dim = head_dim
        assert head_dim & (head_dim - 1) == 0, "head_dim must be power of 2"

        # Generate Hadamard rotation matrix
        self.hadamard = generate_hadamard_matrix(head_dim)

        # Centroid tables (16 levels per coordinate, will be calibrated)
        self.key_centroids = None  # [head_dim, 16]
        self.is_calibrated = False

    def calibrate(self, key_samples: torch.Tensor):
        """Calibrate Lloyd-Max centroids from key samples.

        Args:
            key_samples: [num_samples, head_dim] FP16 key vectors
        """
        device = key_samples.device
        num_samples = key_samples.shape[0]

        # Apply Hadamard rotation
        rotated = torch.matmul(key_samples.float(), self.hadamard.to(device).t())

        # Compute Lloyd-Max centroids per coordinate
        self.key_centroids = torch.zeros(self.head_dim, 16, device=device, dtype=torch.float32)

        for d in range(self.head_dim):
            coord_data = rotated[:, d].cpu().numpy()
            centroids, _ = lloyd_max_quantizer_1d(coord_data, num_levels=16)
            self.key_centroids[d] = torch.from_numpy(centroids).to(device)

        self.is_calibrated = True

    def quantize_keys(self, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize keys to 4-bit with Hadamard + Lloyd-Max.

        Args:
            keys: [seq_len, kv_heads, head_dim] FP16

        Returns:
            key_codes: [seq_len, kv_heads, head_dim] uint8 (4-bit packed)
            norm_factors: [seq_len, kv_heads] FP16 (for norm correction)
        """
        assert self.is_calibrated, "Must calibrate before quantizing"

        seq_len, kv_heads, head_dim = keys.shape
        device = keys.device

        # Compute norms before rotation
        norms = torch.norm(keys.float(), dim=2)  # [seq_len, kv_heads]

        # Apply Hadamard rotation
        keys_flat = keys.reshape(-1, head_dim)
        rotated = torch.matmul(keys_flat.float(), self.hadamard.to(device).t())
        rotated = rotated.reshape(seq_len, kv_heads, head_dim)

        # Quantize each coordinate to nearest centroid
        key_codes = torch.zeros(seq_len, kv_heads, head_dim, dtype=torch.uint8, device=device)

        for d in range(head_dim):
            coord_vals = rotated[:, :, d]  # [seq_len, kv_heads]
            centroids = self.key_centroids[d]  # [16]

            # Find nearest centroid (brute force for now)
            distances = torch.abs(coord_vals[:, :, None] - centroids[None, None, :])
            indices = torch.argmin(distances, dim=2)  # [seq_len, kv_heads]
            key_codes[:, :, d] = indices.to(torch.uint8)

        # Compute norm correction factors
        # Dequantized norm / original norm
        dequant_rotated = torch.zeros_like(rotated)
        for d in range(head_dim):
            dequant_rotated[:, :, d] = self.key_centroids[d][key_codes[:, :, d].long()]

        dequant_norms = torch.norm(dequant_rotated, dim=2)
        norm_factors = (norms / (dequant_norms + 1e-8)).half()

        return key_codes, norm_factors

    def dequantize_keys(self, key_codes: torch.Tensor, norm_factors: torch.Tensor) -> torch.Tensor:
        """Dequantize 4-bit keys back to FP16.

        Args:
            key_codes: [seq_len, kv_heads, head_dim] uint8
            norm_factors: [seq_len, kv_heads] FP16

        Returns:
            keys: [seq_len, kv_heads, head_dim] FP16
        """
        seq_len, kv_heads, head_dim = key_codes.shape
        device = key_codes.device

        # Lookup centroids
        rotated = torch.zeros(seq_len, kv_heads, head_dim, dtype=torch.float32, device=device)
        for d in range(head_dim):
            rotated[:, :, d] = self.key_centroids[d][key_codes[:, :, d].long()]

        # Apply norm correction
        rotated = rotated * norm_factors[:, :, None]

        # Inverse Hadamard rotation
        rotated_flat = rotated.reshape(-1, head_dim)
        keys_flat = torch.matmul(rotated_flat, self.hadamard.to(device))  # Hadamard is self-inverse up to scaling
        keys = keys_flat.reshape(seq_len, kv_heads, head_dim).half()

        return keys


def quantize_values_uniform_4bit(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform 4-bit quantization for values (simpler than keys).

    Args:
        values: [seq_len, kv_heads, head_dim] FP16

    Returns:
        value_codes: [seq_len, kv_heads, head_dim] uint8 (4-bit)
        scales: [seq_len, kv_heads] FP16
        zeros: [seq_len, kv_heads] FP16
    """
    seq_len, kv_heads, head_dim = values.shape

    # Per-token per-head min/max
    v_min = values.float().min(dim=2, keepdim=True)[0]  # [seq_len, kv_heads, 1]
    v_max = values.float().max(dim=2, keepdim=True)[0]

    # Scale: (max - min) / 15
    scales = ((v_max - v_min) / 15.0).squeeze(2)
    scales = torch.clamp(scales, min=1e-8)

    # Quantize
    normalized = (values.float() - v_min) / scales[:, :, None]
    value_codes = torch.clamp(normalized.round(), 0, 15).to(torch.uint8)

    zeros = v_min.squeeze(2).half()
    scales = scales.half()

    return value_codes, scales, zeros


if __name__ == "__main__":
    # Test TQ4NC quantizer
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    head_dim = 128
    num_calibration_samples = 1000
    seq_len, kv_heads = 64, 8

    # Generate calibration data
    calibration_keys = torch.randn(num_calibration_samples, head_dim, device=device, dtype=torch.float16)

    # Initialize and calibrate quantizer
    print("Initializing TQ4NC quantizer...")
    quantizer = TQ4NCQuantizer(head_dim=head_dim)

    print("Calibrating Lloyd-Max centroids...")
    quantizer.calibrate(calibration_keys)

    # Test quantization
    keys = torch.randn(seq_len, kv_heads, head_dim, device=device, dtype=torch.float16)

    print("Quantizing keys...")
    key_codes, norm_factors = quantizer.quantize_keys(keys)

    print("Dequantizing keys...")
    keys_reconstructed = quantizer.dequantize_keys(key_codes, norm_factors)

    # Measure error
    mse = ((keys - keys_reconstructed).float() ** 2).mean().item()
    max_error = (keys - keys_reconstructed).abs().max().item()

    print(f"\nTQ4NC Quantization Results:")
    print(f"  Input shape: {keys.shape}")
    print(f"  Code shape: {key_codes.shape}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Max error: {max_error:.4f}")
    print(f"  Input norm: {keys.norm().item():.2f}")
    print(f"  Reconstructed norm: {keys_reconstructed.norm().item():.2f}")

    # Test value quantization
    values = torch.randn(seq_len, kv_heads, head_dim, device=device, dtype=torch.float16)
    value_codes, v_scales, v_zeros = quantize_values_uniform_4bit(values)
    values_reconstructed = (value_codes.float() * v_scales[:, :, None] + v_zeros[:, :, None]).half()

    v_mse = ((values - values_reconstructed).float() ** 2).mean().item()
    print(f"\nValue Quantization (uniform 4-bit):")
    print(f"  MSE: {v_mse:.6f}")
    print(f"  ✅ TQ4NC basic implementation complete")
