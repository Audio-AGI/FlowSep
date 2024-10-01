# Author: Haohe Liu
# Email: haoheliu@gmail.com
# Date: 11 Feb 2023


import sys

sys.path.append("src")
import shutil
import os
# please modify the following settings to use wandb or setup the cache folder
# os.environ["HF_HOME"] = ""
# os.environ["WANDB_API_KEY"] = ""
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HUGGINGFACE_HUB_CACHE"] = ""
# os.environ["TORCH_HOME"] = ""

# import os

from tqdm import tqdm
import argparse
import yaml
import torch
import ipdb
from utilities.data.dataset import AudioDataset

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
import wandb
from latent_diffusion.util import instantiate_from_config


def print_on_rank0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)
        
def main(configs, exp_group_name, exp_name,text,wav):
    seed_everything(0)

    log_path = configs["log_directory"]

    val_dataset = AudioDataset(configs, split="test", add_ons=[])

    latent_diffusion = instantiate_from_config(configs["model"]).to("cuda")

    resume_from_checkpoint = args.load_checkpoint
    if resume_from_checkpoint is not None:
        ckpt = torch.load(resume_from_checkpoint)["state_dict"]

    latent_diffusion.load_state_dict(ckpt, strict=True)

    print("the checkpoint is",resume_from_checkpoint)

    count = 0
    for cur_text in text:
        cur_wav = wav[count]

        batch = {}

        batch["fname"] = [cur_wav]
        batch["text"] = [cur_text]
        batch["caption"] = [cur_text]
        batch["waveform"] = torch.rand(1,1,163840).cuda()
        batch["log_mel_spec"] = torch.rand(1,1024,64).cuda()
        batch["sampling_rate"] = torch.tensor([16000]).cuda()
        batch["label_vector"] = torch.rand(1,527).cuda()
        batch["stft"] = torch.rand(1,1024,512).cuda()
        noise_waveform, random_start = val_dataset.read_wav_file(cur_wav)

        noise_waveform = noise_waveform[0][:163840]

        mixed_mel, stft = val_dataset.wav_feature_extraction(noise_waveform.reshape(1,-1))
        
        batch["mixed_waveform"] = torch.from_numpy(noise_waveform.reshape(1,1,163840))
        batch["mixed_mel"] = mixed_mel.reshape(1,mixed_mel.shape[0],mixed_mel.shape[1])

        waveform = latent_diffusion.generate_sample([batch],name="lass_result",unconditional_guidance_scale=1.0,ddim_steps=args.infer_step,n_gen=1,save_mixed = args.no_mixed)
        count+=1


def set_yaml_config(config_yaml, cfg_scale, ddim, n_cand):
    config_yaml["model"]["params"]["evaluation_params"]["unconditional_guidance_scale"] = cfg_scale
    config_yaml["model"]["params"]["evaluation_params"]["ddim_sampling_steps"] = ddim
    config_yaml["model"]["params"]["evaluation_params"]["n_candidates_per_samples"] = n_cand
    return config_yaml

if __name__ == "__main__":

    # python3 ~/submit.py -i config/4_audiomae_cond/2023_05_20_audiomae_crossattn_audiocaps_pool_rand.sh --cpu 32 --gpu 4 --mem 96 --group_id 122 --cluster_id 17 --note audiomae_pool_rand_audiocaps --elas
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        default="lass_config/2channel_flow.yaml",
        help="path to config .yaml file",
    )

    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default="A rocket flies by followed by a loud explosion and fire crackling as a truck engine runs idle_mixture",
        help="text of the prompt",
    )

    parser.add_argument(
        "-a",
        "--audio",
        type=str,
        default="metadata-master/mixed/exp1_A rocket flies by followed by a loud explosion and fire crackling as a truck engine runs idle_mixture.wav",
        help="path to config .yaml file",
    )
    parser.add_argument(
        "-l",
        "--load_checkpoint",
        type=str,
        default="model_logs/pretrained/v2_100k.ckpt",
        help="path to the checkpoint",
    )

    parser.add_argument(
        "-s",
        "--infer_step",
        type=int,
        default="20",
        help="the steps for inference",
    )
    

    parser.add_argument(
        "-m",
        "--no_mixed",
        type=bool,
        action='store_false',
        help="saving the mixed waveform as well",
    )
    
    args = parser.parse_args()
    # torch._dynamo.config.suppress_errors = True

    assert torch.cuda.is_available(), "CUDA is not available"

    config_yaml = args.config_yaml

    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml_path = os.path.join(config_yaml)
    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)
    
    text = [args.text]

    wav = [args.audio]

    # text = ["A rocket flies by followed by a loud explosion and fire crackling as a truck engine runs idle",
    # "A series of burping and farting",
    # "A ticktock sound playing at the same rhythm with piano notes",
    # "A man speaks then a small bird chirps",
    # "Footsteps and scuffing occur, after which a door grinds, squeaks and clicks, an adult male speaks, and the door grinds, squeaks and clicks shut with a thump",
    # ]

    # wav = ["mixed/exp1_A rocket flies by followed by a loud explosion and fire crackling as a truck engine runs idle_mixture.wav",
    # "mixed/exp367_A series of burping and farting_mixture.wav",
    # "mixed/exp996_A ticktock sound playing at the same rhythm with piano notes_mixture.wav",
    # "mixed/exp1338_A man speaks then a small bird chirps_mixture.wav",
    # "mixed/exp1718_Footsteps and scuffing occur, after which a door grinds, squeaks and clicks, an adult male speaks, and the door grinds, squeaks and clicks shut with a thump_mixture.wav",
    # ]
    
    main(config_yaml,exp_group_name, exp_name,text,wav)
