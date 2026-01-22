import os
import json
import argparse

import torch

from src.inference import sample
from src import belm

from utils.codec import map_uniform2gaussian, unshuffle1d, decode_from_gaussian
from utils.io import load_model, load_csvfiles_prompts, save_audio_bundle, load_audio_bundle, create_metrics_record  
from utils.gen import derive_seeds
from utils.metrics import calculate_ber, avg_distance_to_diag



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
    
    parser.add_argument('--sigma', type=float, default=0.85)
    
    parser.add_argument('--save_dir', type=str, default='/data1/kyranhsiao/data/diffstega-audio/res')
    parser.add_argument('--csvfiles_path', type=str, default='csvfiles/train.csv')
    parser.add_argument('--num_samples', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
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
    prompts = load_csvfiles_prompts(args.csvfiles_path, num_prompts=args.num_samples, seed=args.seed)
    seed_list = derive_seeds(args.seed, args.num_samples)
    bit_rates = ['160k', '128k', '64k', '48k', '32k', '16k']
    formats = ['WAV_f32', 'WAV_s16']
    for bit_rate in bit_rates:
        formats.extend([f'AAC_{bit_rate}', f'MP3_{bit_rate}']) # The order must be consistent with the loading function
    
    for idx, (prompt, sample_seed) in enumerate(zip(prompts, seed_list)):
        (gaussian_data, X,
         paired, unpaired,
         num_pairs, bin_str) = map_uniform2gaussian(latent_shape, args.sigma, sample_seed)
        gaussian_data = gaussian_data.to(args.device)
        # audio diffusion
        output, z0, zi_list = sample(
            ezaudio.autoencoder, ezaudio.unet, ezaudio.tokenizer,
            ezaudio.text_encoder, ezaudio.params, ezaudio.noise_scheduler,
            prompt, '', latent_dim,
            args.guidance_scale, args.guidance_rescale, args.ddim_steps,
            args.eta, sample_seed, args.device,
            latents=gaussian_data
        )
        save_audio_bundle(output, sr, bit_rates, args.save_dir, str(idx))
        audio_list = load_audio_bundle(bit_rates, args.save_dir, str(idx))
        audio_length = min(x.shape[0] for x in audio_list)
        audio_list = [x[:audio_length] for x in audio_list]
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
        
        batched_ori_gaussian_flatten = gaussian_data.expand(len(audio_list), *gaussian_data.shape).view(len(audio_list), -1).cpu()
        noise_flatten = noise.view(len(audio_list), -1).cpu()
        
        ori_gaussian_unshuffled = unshuffle1d(batched_ori_gaussian_flatten, sample_seed)
        get_gaussian_unshuffled = unshuffle1d(noise_flatten, sample_seed)
        ori_pairs = ori_gaussian_unshuffled[:, :num_pairs * 2]
        get_pairs = get_gaussian_unshuffled[:, :num_pairs * 2]
        
        ori_bin_str = [bin_str] * len(audio_list)
        get_bin_str = decode_from_gaussian(get_pairs)
        
        '''Metrics to save'''
        ber, err_cnt = map(list, zip(*(calculate_ber(ori_bits, get_bits) 
                                    for ori_bits, get_bits in zip(ori_bin_str, get_bin_str))))
        diff = torch.abs(ori_pairs - get_pairs)
        max_diff = diff.max(dim=1).values.tolist()
        min_diff = diff.min(dim=1).values.tolist()
        avg_diff = diff.mean(dim=1).tolist()
        std_diff = diff.std(dim=1).tolist()
        ori_avg_dist = avg_distance_to_diag(ori_pairs).tolist()
        get_avg_dist = avg_distance_to_diag(get_pairs).tolist()

        metrics_data = create_metrics_record(
            idx, sample_seed,
            prompt, args.sigma, num_pairs,
            formats,
            ber,
            err_cnt,
            max_diff, min_diff, avg_diff, std_diff,
            ori_avg_dist, get_avg_dist
        ) 
        metrics_path = os.path.join(args.save_dir, str(idx), 'metrics.json')
        with open(metrics_path, mode='w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=4)       

if __name__ == '__main__':
    args = parse_arguments()
    main(args)