# Code modefied from AudioLDM Haohe Liu details can be found from https://github.com/haoheliu/AudioLDM-training-finetunin

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

import argparse
import yaml
import torch
import ipdb

from pytorch_lightning.strategies.ddp import DDPStrategy
# from latent_diffusion.models.ddpm import LatentDiffusion
from utilities.data.dataset import AudioDataset
from audioldm_eval import EvaluationHelper

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from utilities.tools import get_restore_step
# import wandb
from latent_diffusion.util import instantiate_from_config
import logging
import pdb
from tqdm import tqdm
# logging.basicConfig(level=logging.WARNING)
logging.getLogger('fsspec').setLevel(logging.ERROR)
# import logging
logging.getLogger('numba').setLevel(logging.WARNING)


# wandb.login()

def print_on_rank0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)
        
def main(configs, config_yaml_path, exp_group_name, exp_name):
    if("seed" in configs.keys()):
        seed_everything(configs["seed"])
    else:
        seed_everything(0)
    if("precision" in configs.keys()):
        torch.set_float32_matmul_precision(configs["precision"])


    log_path = configs["log_directory"]
    exp_group_name = configs["exp_group"]
    exp_name = configs["exp_name"]

    batch_size = configs["model"]["params"]["batchsize"]

    # device = torch.device(f"cuda:{0}")
    # evaluator = EvaluationHelper(16000, device)
    evaluator = None

    if("dataloader_add_ons" in configs["data"].keys()):
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    dataset = AudioDataset(configs, split="train", add_ons=dataloader_add_ons)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=batch_size,
        # num_workers=0,
        pin_memory=True,
        shuffle=True,
    )

    
    # one=next(it)
    # pdb.set_trace()

    print(
        "The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s"
        % (len(dataset), len(loader), batch_size)
    )
    # it = iter(loader)
    # for i in tqdm(range(len(loader))):
    #     one=next(it)


    val_dataset = AudioDataset(configs, split="test", add_ons=dataloader_add_ons)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
    )

    print(
        "The length of the test_dataset is %s, the length of the dataloader is %s, the batchsize is %s"
        % (len(val_dataset), len(val_loader), batch_size)
    )

    # Copy test data
    test_data_subset_folder = os.path.join(
        os.path.dirname(configs["log_directory"]), "testset_data", val_dataset.dataset_name
    )
    os.makedirs(test_data_subset_folder, exist_ok=True)
    val_len = len(os.listdir(test_data_subset_folder))
    # if val_len<100:
    #     print("the length of val is",val_len)
    #     copy_test_subset_data(
    #         val_dataset.data, test_data_subset_folder
    #     )

    device_count = torch.cuda.device_count() 

    try:
        config_reload_from_ckpt = configs["reload_from_ckpt"]
    except:
        config_reload_from_ckpt = None

    try:
        limit_val_batches = configs["step"]["limit_val_batches"]
    except:
        limit_val_batches = None

    # validation_every_n_epochs = configs["step"]["validation_every_n_epochs"]
    # validation_every_n_epochs = 2/436370
    # try:
    #     validation_every_n_steps= configs["step"]["validation_every_n_steps"]
    #     validation_every_n_epochs = int(validation_every_n_steps/len(loader)/device_count)
    # except:
    validation_every_n_steps = (configs["step"]["validation_every_n_epochs"]) * len(loader)
    validation_every_n_epochs = configs["step"]["validation_every_n_epochs"]


    validation_every_n_steps = validation_every_n_epochs * len(loader)/device_count

    # if validation_every_n_epochs >= 1:
    #     validation_every_n_steps = None
    
    # ipdb.set_trace()

    

    save_checkpoint_every_n_steps = configs["step"]["save_checkpoint_every_n_steps"]
    max_steps = configs["step"]["max_steps"]
    save_top_k = configs["step"]["save_top_k"]

    checkpoint_path = os.path.join(
        log_path,
        exp_group_name, 
        exp_name,
        "checkpoints"
    )

    wandb_path = os.path.join(
        log_path,
        exp_group_name, 
        exp_name
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="global_step",
        mode="max",
        filename="checkpoint-fad-{val/frechet_inception_distance:.2f}-global_step={global_step:.0f}",
        every_n_train_steps=save_checkpoint_every_n_steps,
        save_top_k=save_top_k,
        auto_insert_metric_name=False,
        save_last=True,
    )

    os.makedirs(checkpoint_path, exist_ok=True)

    # shutil.copy(config_yaml_path, wandb_path)
    # # os.system("cp %s %s" % (config_yaml_path, wandb_path))

    if len(os.listdir(checkpoint_path)) > 0:
        print("Load checkpoint from path: %s" % checkpoint_path)
        restore_step, n_step = get_restore_step(checkpoint_path)
        resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
        print("Resume from checkpoint", resume_from_checkpoint)
    elif config_reload_from_ckpt is not None:
        resume_from_checkpoint = config_reload_from_ckpt
        print("Reload ckpt specified in the config file %s" % resume_from_checkpoint)
    else:
        print("Train from scratch")
        resume_from_checkpoint = None

    devices = torch.cuda.device_count()





    latent_diffusion = instantiate_from_config(configs["model"])
    latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)

    wandb_logger = WandbLogger(
        project=exp_group_name,
        name= exp_name,
        save_dir=wandb_path,
        # project=configs["project"],
        config=configs,
    )

    latent_diffusion.test_data_subset_path = test_data_subset_folder
    
    print("==> Save checkpoint every %s steps" % save_checkpoint_every_n_steps)
    print("==> Perform validation every %s epoch" % validation_every_n_epochs)

    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        # precision="16-mixed",
        # profiler=profiler,
        logger=wandb_logger,
        max_steps = max_steps,
        num_sanity_val_steps=0, 
        limit_val_batches=limit_val_batches,
        check_val_every_n_epoch = validation_every_n_epochs,
        # val_check_interval = validation_every_n_steps,
        # check_val_every_n_step=validation_every_n_steps,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback],
    )


    trainer.fit(latent_diffusion, loader, val_loader, ckpt_path=resume_from_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        default = "lass_config/2channel_flow.yaml",
        required=False,
        help="path to config .yaml file",
    )
    
    args = parser.parse_args()
    # torch._dynamo.config.suppress_errors = True

    assert torch.cuda.is_available(), "CUDA is not available"

    config_yaml = args.config_yaml

    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml_path = os.path.join(config_yaml)
    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)
    main(config_yaml, config_yaml_path, exp_group_name, exp_name)

