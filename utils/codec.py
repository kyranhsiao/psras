from typing import Tuple, List
import torch
import numpy as np
import subprocess
import reedsolo

from utils.gen import generate_uniform_distribution

def map_uniform2gaussian(
    latent_shape: Tuple,
    sigma: float,
    seed: int = 54,
    bin_str: str = ''):
    latent_size = int(np.prod(latent_shape))
    uniform_X, uniform_paired, uniform_unpaired, num_pairs, bin_str = cross_pair(bin_str, latent_size, sigma, seed)
    gaussian_paired = convert_uniform2gaussian(uniform_paired)
    gaussian_unpaired = convert_uniform2gaussian(uniform_unpaired)
    gaussian_X = convert_uniform2gaussian(uniform_X)
    gaussian_X_shuffled = shuffle1d(gaussian_X.unsqueeze(0), seed)
    gaussian_data = gaussian_X_shuffled.view(latent_shape)
    
    return gaussian_data, gaussian_X, gaussian_paired, gaussian_unpaired, num_pairs, bin_str
    
def convert_uniform2gaussian(uniform_X: torch.tensor) -> torch.Tensor:
    gaussian_X = torch.erfinv(2 * uniform_X - 1) * torch.sqrt(torch.tensor(2.0))
    return gaussian_X
    
def shuffle1d(X: torch.Tensor, seed: int) -> torch.Tensor:
    assert X.ndim == 2
    generator = torch.Generator().manual_seed(seed)
    B, L = X.shape
    random_indices = torch.randperm(L, generator=generator, device=X.device)
    X_shuffled = torch.gather(X, 1, random_indices.repeat(B, 1))
    return X_shuffled

def unshuffle1d(X: torch.Tensor, seed: int) -> torch.Tensor:
    assert X.ndim == 2
    generator = torch.Generator().manual_seed(seed)
    B, L = X.shape
    random_indices = torch.randperm(L, generator=generator, device=X.device)
    
    inv_indices = torch.empty_like(random_indices)
    inv_indices[random_indices] = torch.arange(L, device=X.device)

    X_unshuffled = torch.gather(X, 1, inv_indices.repeat(B, 1))
    return X_unshuffled

def cross_pair(
    bin_str: str,
    latent_size: int,
    sigma: float,
    seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str]:
    generator = torch.Generator().manual_seed(seed) # ensure only affecting the randomness using this generator
    # generate a [0, 1] uniform distribution 
    X = generate_uniform_distribution(latent_size, seed)
    X = torch.tensor(X, dtype=torch.float32)
    X_sorted, _ = torch.sort(X)
    X_le0_5 = X_sorted[X_sorted <= 0.5]
    X_gt0_5 = X_sorted[X_sorted > 0.5]
    
    # find all possible pairs based on `sigma`
    used_idx = set()
    pairs = []
    i, j = 0, 0
    while i < len(X_le0_5) and j < len(X_gt0_5):
        if torch.abs(X_le0_5[i] - X_gt0_5[j]) >= sigma:
            pairs.append([X_le0_5[i], X_gt0_5[j]])
            used_idx.add(i)
            used_idx.add(len(X_le0_5) + j)
            i += 1
            j += 1
        else:
            j += 1
    num_pairs = len(bin_str) if bin_str else len(pairs)
    assert len(bin_str) <= len(pairs), f'The generated gaussian distribution cannot embed {len(bin_str)} bits >= {num_pairs}!'
    if not bin_str:
        bits = torch.randint(0, 2, (num_pairs,), generator=generator)
        bin_str = ''.join(map(str, bits.tolist()))
        
    # direction encoding
    k = 0
    for bit in bin_str:
        if bit == '1':
            pairs[k][0], pairs[k][1] = pairs[k][1], pairs[k][0]
        k += 1
    
    # convert points list to tensor
    mask = torch.ones(len(X_sorted), dtype=torch.bool)
    mask[list(used_idx)] = False
    unpaired = X_sorted[mask]
    paired = torch.tensor(pairs, dtype=X.dtype).flatten()
    
    X_out = torch.cat([paired, unpaired]) # concatenate `paired` and `unpaired` sampling points
    return X_out, paired, unpaired, num_pairs, bin_str
    
def decode_from_gaussian(X: torch.Tensor) -> List[str]:
    B, L = X.shape
    assert L % 2 == 0
    first_in_pair = X[:, 0::2]
    second_in_pair = X[:, 1::2]
    bits = (first_in_pair <= second_in_pair)
    bin_chars = torch.where(bits, 0, 1)
    
    bin_str = [''.join(map(str, row.tolist())) for row in bin_chars]
    return bin_str

def convert_ndarray2adfile(
    audio: np.ndarray,
    sr: int,
    bit_rate: str,
    save_path: str) -> None:
    assert audio.dtype == np.float32
    audio_s16 = (audio * 32767).astype(np.int16) # convert f32 to s16
    if audio_s16.ndim > 1:
        audio_s16 = audio_s16[:, 0]
        
    cmd = [
        'ffmpeg',
        '-y', # overwrite the file if exists
        '-f', 's16le', # PCM16 
        '-ar', str(sr), # sample rate
        '-ac', '1', # number of channels
        '-i', 'pipe:0', # read from stdin
        '-b:a', bit_rate, # audio bit rate
        save_path # file path
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    process.communicate(audio_s16.tobytes())
    if process.returncode != 0:
        raise RuntimeError('Fail to run ffmpeg!')
    
def RS_encode(
    plain_text: str,
    nsym: int = 10) -> str:
    rs = reedsolo.RSCodec(nsym=nsym)
    bytes_rep = plain_text.encode('utf-8')
    encoded = rs.encode(bytes_rep)
    enc_bin_str = ''.join(f'{byte:08b}' for byte in encoded)
    return enc_bin_str

def RS_decode(
    bin_str: str,
    nsym: int = 10) -> Tuple[str, str]:
    try:
        rs = reedsolo.RSCodec(nsym=nsym)
        enc = bytes(int(bin_str[i: i+8], 2) for i in range(0, len(bin_str), 8))
        dec, _, _ = rs.decode(enc)
        dec_text = bytes(dec).decode('utf-8')
        dec_bin_str = ''.join(f'{b:08b}' for b in dec)
        return dec_text, dec_bin_str
    except:
        null_bin_str = '0' * (len(bin_str) - 8 * nsym)
        print(f'Warning: Too many errors to correct, fail to decode!')
        return '', null_bin_str