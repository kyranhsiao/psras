from typing import Tuple
import torch
import math

def calculate_ber(ori_bits: str, get_bits: str) -> Tuple[float, int]:
    assert len(ori_bits) == len(get_bits)
    err = sum(o != g for o, g in zip(ori_bits, get_bits))
    ber = err / len(ori_bits)
    return ber, err

def avg_distance_to_diag(pairs: torch.Tensor) -> torch.Tensor:
    assert pairs.ndim == 2
    assert pairs.shape[1] % 2 == 0
    
    B, L = pairs.shape
    points = pairs.view(B, -1, 2)
    distances = torch.abs(points[:, :, 0] - points[:, :, 1]) / math.sqrt(2)
    
    return distances.mean(dim=1)