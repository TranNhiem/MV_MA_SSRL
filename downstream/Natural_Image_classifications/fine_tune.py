# SimClr
# SGD, batch_size = 4096, 1024
# Learning rate = 0.8(1 percent)
# Random cropping + flipping
# 60 epochs(1 percent), 30 epochs(10 percent)
# BYOL learning rate = [0.01, 0.02, 0.05, 0.1, 0.005] with epochs = [30, 50]

from argparse import Namespace
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
from torchvision.models import resnet50

from solo.methods.linear import LinearModel  # imports the linear eval class
from solo.utils.classification_dataloader import prepare_data
from solo.utils.checkpointer import Checkpointer

# basic parameters for offline linear evaluation
# some parameters for extra functionally are missing, but don't mind this for now.

DATASET_NAME = 'ImageNet1per'
METHOD_NAME = 'Baseline'

kwargs = {
    "num_classes": 1000,
    "cifar": False,
    "max_epochs": 200,
    "optimizer": "sgd",
    "precision": 16,
    "lars": False,
    "lr": 0.8,
    "exclude_bias_n_norm": False,
    "gpus": [1, 2],    # Number of GPUs
    "weight_decay": 0.0003,
    "extra_optimizer_args": {"momentum": 0.9},
    "scheduler": "step",
    "warmup_epochs": None,
    "min_lr": None,
    "warmup_start_lr": None,
    "lr_decay_steps": [24, 48, 72],
    "replica_batch_size": 4096,
    "batch_size": 4096,
    "num_workers": 4,
    "worker_init_fn": None,
    "metric": 'accuracy',
    "task": 'semi_supervised',
    "pretrained_feature_extractor": "/home/rick/MNCRL_Solo/MNCRL-resnet50-imagenet-300ep-1f2tdknm-ep=290.ckpt",
}

# create the backbone network
backbone = resnet50()
backbone.fc = torch.nn.Identity()

# load pretrained feature extractor
state = torch.load(kwargs["pretrained_feature_extractor"])["state_dict"]
for k in list(state.keys()):
    if "backbone" in k:
        state[k.replace("backbone.", "")] = state[k]
    del state[k]
backbone.load_state_dict(state, strict=False)

model = LinearModel(backbone, **kwargs)


train_loader, val_loader = prepare_data(
    "custom",
    data_dir="/data1/1K_New",
    train_dir="/data1/1K_New/train_1per",
    val_dir="/data1/1K_New/val",
    batch_size=kwargs["replica_batch_size"],
    num_workers=kwargs["num_workers"],
    worker_init_fn=kwargs["worker_init_fn"]
)

wandb_logger = WandbLogger(
    # name of the experiment
    name=f'{METHOD_NAME}_linear_eval_{DATASET_NAME}_lr={kwargs["lr"]}_wd={kwargs["weight_decay"]}',
    project="MNCRL_downstream_tasks",  # name of the wandb project
    entity='mlbrl',
    offline=False,
)
wandb_logger.watch(model, log="all", log_freq=50)

callbacks = []

# automatically log our learning rate
# lr_monitor = LearningRateMonitor(logging_interval="epoch")
# callbacks.append(lr_monitor)

# checkpointer can automatically log your parameters,
# but we need to wrap them in a Namespace object
args = Namespace(**kwargs)
# saves the checkout after every epoch
ckpt = Checkpointer(
    args,
    logdir="checkpoints/linear",
    frequency=1,
)
callbacks.append(ckpt)
args = Namespace(**kwargs)

trainer = Trainer(accelerator='gpu', gpus=kwargs["gpus"],
                  logger=wandb_logger, max_epochs=kwargs["max_epochs"], strategy="ddp")

if __name__ == '__main__':
    print("start training")
    # trainer.validate(model, val_loader)
    trainer.fit(model, train_loader, val_loader)
    print("end training")
