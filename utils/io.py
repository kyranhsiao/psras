from api.ezaudio import EzAudio

import os
import subprocess
from typing import List, Dict, Optional
import random

import torch
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

from utils.codec import convert_ndarray2adfile

def load_model(
    model_name: str,
    ckpt_path: str,
    vae_path: str,
    device: str) -> EzAudio:
    model = EzAudio(
        model_name=model_name,
        ckpt_path=ckpt_path,
        vae_path=vae_path,
        device=device)
    return model
    
def load_csvfiles_prompts(
    csvfile_path: str,
    num_prompts: int,
    seed: int = 54) -> List[str]:
    df = pd.read_csv(csvfile_path)
    prompts = df['caption'].tolist()
    generator = random.Random(seed)
    prompts = generator.sample(prompts, num_prompts)
    return prompts

def save_audio_bundle(
    audio: np.ndarray,
    sr: int,
    br_list: List[str],
    # prompt: str,
    base_dir: str,
    idx: str) -> str:
    save_dir = os.path.join(base_dir, idx)
    os.makedirs(save_dir, exist_ok=True)
    
    sf.write(os.path.join(save_dir, f'{idx}_f32.wav'), audio, sr, subtype='FLOAT')
    sf.write(os.path.join(save_dir, f'{idx}_s16.wav'), audio, sr, subtype='PCM_16')
    for bit_rate in br_list:
        convert_ndarray2adfile(audio, sr, bit_rate, os.path.join(save_dir, f'{idx}_{bit_rate}.m4a'))
        convert_ndarray2adfile(audio, sr, bit_rate, os.path.join(save_dir, f'{idx}_{bit_rate}.mp3'))

    return save_dir

def ffmpeg_load(file_path: str, sr: int = None):
    if sr is not None:
        cmd = [
            'ffmpeg',
            '-i', file_path,
            '-f', 's16le',
            '-ac', '1',
            '-ar', sr,
            'pipe:1'
        ]
    else:
        cmd = [
            'ffmpeg',
            '-i', file_path,
            '-f', 's16le',
            '-ac', '1',
            'pipe:1'
        ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out, _ = process.communicate()
    audio = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
    return audio

def load_audio_bundle(
    br_list: List[str],
    base_dir: str,
    idx: str) -> List[torch.Tensor]:
    bundle = []
    save_dir = os.path.join(base_dir, idx)
    
    wav_f32, _ = librosa.load(os.path.join(save_dir, f'{idx}_f32.wav'), sr=None)
    wav_s16, _ = librosa.load(os.path.join(save_dir, f'{idx}_s16.wav'), sr=None)
    bundle.extend([wav_f32, wav_s16])
 
    # The order must be consistent with the format list
    for bit_rate in br_list:
        aac = ffmpeg_load(os.path.join(save_dir, f'{idx}_{bit_rate}.m4a'), sr=None)
        mp3 = ffmpeg_load(os.path.join(save_dir, f'{idx}_{bit_rate}.mp3'), sr=None)
        bundle.extend([aac, mp3])
    bundle = [torch.from_numpy(x) for x in bundle]
    return bundle
        
def create_metrics_record(
    idx: int,
    seed: int,
    prompt: str,
    sigma: float,
    num_pairs: int,
    formats: List[str],
    ber: List[float],
    err_cnt: List[int],
    max_diff: List[float],
    min_diff: List[float],
    avg_diff: List[float],
    std_diff: List[float],
    ori_avg_dist: List[float],
    get_avg_dist: List[float],
    message: Optional[List[str]] = None,
    success: Optional[List[bool]] = None,
) -> Dict:
    metrics_list = []
    for i, fmt in enumerate(formats):
        metric = {
            'format': fmt,
            'ber': ber[i],
            'err_cnt': err_cnt[i],
            'max_diff': max_diff[i],
            'min_diff': min_diff[i],
            'avg_diff': avg_diff[i],
            'std_diff': std_diff[i],
            'ori_avg_dist': ori_avg_dist[i],
            'get_avg_dist': get_avg_dist[i],
        }

        # Only add if provided
        if message is not None:
            metric['message'] = message[i]
        if success is not None:
            metric['success'] = success[i]

        metrics_list.append(metric)

    metrics_data = {
        'id': idx,
        'seed': seed,
        'prompt': prompt,
        'sigma': sigma,
        'num_pairs': num_pairs,
        'metrics': metrics_list
    }

    return metrics_data
   
    
    