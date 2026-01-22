import random
from typing import List

import torch

def derive_seeds(init_seed: int, num_seeds: int) -> List[int]:
    generator = random.Random(init_seed)
    return [generator.randrange(0, 2**32) for _ in range(num_seeds)]


def generate_uniform_distribution(
    size: int,
    seed: int, 
    eps: float = 1e-12) -> List[float]:
    generator = random.Random(seed)
    return [generator.uniform(eps, 1 - eps) for _ in range(size)]