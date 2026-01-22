import os
import argparse
import secrets
import json
import torch

from api.ezaudio import EzAudio
from src.inference import sample
from src import belm

from utils.codec import *
from utils.gen import *
from utils.io import *
from utils.metrics import *

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
    
    parser.add_argument('--prompt', type=str, default='a bird is singing')
    parser.add_argument('--message', type=str, default='AaZz10!@#$%^&*()_+[]|;:~你好！')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default='/data1/kyranhsiao/data/diffstega-audio/output')
    parser.add_argument('--sub_dir', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    if args.seed is None:
        args.seed = secrets.randbelow(2**31)
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
    bit_rates = ['160k', '128k', '64k', '48k', '32k', '16k']
    formats = ['WAV_f32', 'WAV_s16']
    for bit_rate in bit_rates:
        formats.extend([f'AAC_{bit_rate}', f'MP3_{bit_rate}']) # The order must be consistent with the loading function

    idx = args.sub_dir
    bin_str = RS_encode(args.message)
    (gaussian_data, X,
     paired, unpaired,
     num_pairs, bin_str) = map_uniform2gaussian(latent_shape, args.sigma, args.seed, bin_str)
    # audio diffusion
    gaussian_data = gaussian_data.to(args.device)
    output, z0, zi_list = sample(
        ezaudio.autoencoder, ezaudio.unet, ezaudio.tokenizer,
        ezaudio.text_encoder, ezaudio.params, ezaudio.noise_scheduler,
        args.prompt, '', latent_dim,
        args.guidance_scale, args.guidance_rescale, args.ddim_steps,
        args.eta, args.seed, args.device,
        latents=gaussian_data
    )
    save_audio_bundle(output, sr, bit_rates, args.save_dir, str(idx))
    audio_list = load_audio_bundle(bit_rates, args.save_dir, str(idx))
    audio_length = min(x.shape[0] for x in audio_list)
    audio_list = [x[:audio_length] for x in audio_list]
    batched_audio = torch.stack(audio_list, dim=0).to(args.device)
    batched_prompts = [args.prompt] * len(audio_list)
    # recover the initial noise from lossy audios
    z0 = belm.wav2latent(ezaudio.autoencoder, batched_audio, ezaudio.params).clone().float()
    _, noise, rec_zi = sample(
        ezaudio.autoencoder, ezaudio.unet, ezaudio.tokenizer,
        ezaudio.text_encoder, ezaudio.params, ezaudio.inverse_scheduler,
        batched_prompts, [''] * len(audio_list), latent_dim,
        args.guidance_scale, args.guidance_rescale, args.ddim_steps,
        args.eta, args.seed, args.device,
        inverse=True, latents=z0
    )
    
    batched_ori_gaussian_flatten = gaussian_data.expand(len(audio_list), *gaussian_data.shape).view(len(audio_list), -1).cpu()
    noise_flatten = noise.view(len(audio_list), -1).cpu()
    
    ori_gaussian_unshuffled = unshuffle1d(batched_ori_gaussian_flatten, args.seed)
    get_gaussian_unshuffled = unshuffle1d(noise_flatten, args.seed)
    ori_pairs = ori_gaussian_unshuffled[:, :num_pairs * 2]
    get_pairs = get_gaussian_unshuffled[:, :num_pairs * 2]

    ori_bin_str = [bin_str] * len(audio_list)
    get_bin_str = decode_from_gaussian(get_pairs)
    dec_text = [RS_decode(b)[0] for b in get_bin_str]
    success = [bool(dec_txt) for dec_txt in dec_text]
    
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
        idx, args.seed,
        args.prompt, args.sigma, num_pairs,
        formats,
        ber,
        err_cnt,
        max_diff, min_diff, avg_diff, std_diff,
        ori_avg_dist, get_avg_dist,
        dec_text, success,
    )        
    metrics_path = os.path.join(args.save_dir, str(idx), 'metrics.json')
    with open(metrics_path, mode='w', encoding='utf-8') as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=4)

    print('*' * 100)
    print(f'Generated audios have been saved to "{os.path.join(args.save_dir, str(idx))}" !!!')
    print(f'Metrics have been saved to "{metrics_path}" !!!')
    print(f'Extracted messages go below: ')
    for fmt, dec_txt in zip(formats, dec_text):
        print(f'{fmt:10}: {dec_txt}')
    print('*' * 100)
    
if __name__ == '__main__':
    args = parse_arguments()
    main(args)

