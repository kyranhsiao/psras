import os
import argparse
import json

import torch
import librosa

from src.inference import sample
from src import belm
from utils.io import load_model, ffmpeg_load
from utils.codec import unshuffle1d, decode_from_gaussian, RS_decode

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generation configurations.')
    '''Configuration for audio generation'''
    parser.add_argument('--guidance_scale', type=float, default=5.0)
    parser.add_argument('--guidance_rescale', type=float, default=0.75)
    parser.add_argument('--ddim_steps', type=int, default=100)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--model_name', type=str, default='s3_xl')
    parser.add_argument('--ckpt_path', type=str, default='Ezaudio/ckpts/s3/ezaudio_s3_xl.pt')
    parser.add_argument('--vae_path', type=str, default='Ezaudio/ckpts/vae/1m.pt')

    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    return args

def main(args):
    ezaudio = load_model(
        model_name=args.model_name,
        ckpt_path=args.ckpt_path,
        vae_path=args.vae_path,
        device=args.device
    )
    sr = ezaudio.params['autoencoder']['sr']
    latent_dim = 10 * ezaudio.params['autoencoder']['latent_sr']
    codec_dim = ezaudio.params['model']['out_chans']
    latent_shape = (1, codec_dim, latent_dim)
    
    base_dir = os.path.dirname(args.audio_path)
    metrics_path = os.path.join(base_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f'No such file {metrics_path}!')
    with open(metrics_path, mode='r', encoding='utf-8') as f:
        metrics_data = json.load(f)
    # sigma is not needed for message extraction
    sample_seed = metrics_data['seed']
    prompt = metrics_data['prompt']
    num_pairs = metrics_data['num_pairs']
    
    _, ext = os.path.splitext(args.audio_path)
    if ext.lower() == 'wav':
        audio, _ = librosa.load(args.audio_path, sr=None)
    else:
        audio = ffmpeg_load(args.audio_path, sr=None)
    
    audio_list = [torch.from_numpy(audio)]
    batched_audio = torch.stack(audio_list, dim=0).to(args.device)
    batched_prompts = [prompt] * len(audio_list)
    
    # recover the initial noise from lossy audios
    z0 = belm.wav2latent(ezaudio.autoencoder, batched_audio, ezaudio.params).clone().float()
    _, noise, rec_zi = sample(
        ezaudio.autoencoder, ezaudio.unet, ezaudio.tokenizer,
        ezaudio.text_encoder, ezaudio.params, ezaudio.inverse_scheduler,
        batched_prompts, [''] * len(audio_list), latent_dim,
        args.guidance_scale, args.guidance_rescale, args.ddim_steps,
        args.eta, sample_seed, args.device,
        inverse=True, latents=z0
    )
    noise_flatten = noise.view(len(audio_list), -1).cpu()
    
    get_gaussian_unshuffled = unshuffle1d(noise_flatten, sample_seed)
    get_pairs = get_gaussian_unshuffled[:, :num_pairs * 2]
    get_bin_str = decode_from_gaussian(get_pairs)
    
    dec_text = [RS_decode(b)[0] for b in get_bin_str]
    if dec_text == '':
        print(f'Fail to fully extract the embedded message!')
    else:
        print(f'Extracted message: {dec_text}')
        
if __name__ == '__main__':
    args = parse_arguments()
    main(args)