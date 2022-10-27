import copy 
import torch 
from torch import nn 
import pytorch_lightning as pl
from functools import partial
import torchvision
from typing import List, Tuple, Optional, Dict, Any
from MVAR_Dino.utils.update_momentum import update_momentum, normalize_weight, deactivate_requires_grad
from MVAR_Dino.utils.modules import DINOProjectionHead, static_lr
from MVAR_Dino.utils.dino_loss import DINOLoss
from MVAR_Dino.ViTs.vision_transformer import vit_tiny, vit_base, vit_small
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from MV_MA_SSL.utils.knn import WeightedKNNClassifier
from MV_MA_SSL.utils.lars import LARSWrapper
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from MV_MA_SSL.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform_mv_ma,
    prepare_transform,
)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument( "--loadertype", type=str, default="mv_ma", help="dataloader type")
parser.add_argument( "--num_augment_trategy", type=str, default="SimCLR_RA", help="Strategy augmentation type")
parser.add_argument( "--num_crops_per_aug", type=list, default=[1, 1], help="Strategy augmentation type")
parser.add_argument( "--num_crop_glob", type=str, default=2, help="Strategy augmentation type")
parser.add_argument( "--crop_size_glob", type=int, default=224, help="")
parser.add_argument( "--num_crop_loc", type=str, default=4, help="")
parser.add_argument( "--crop_size_loc", type=int, default=96, help="")
parser.add_argument( "--crop_type", type=str, default="inception_style", help="")
parser.add_argument( "--min_loc", type=float, default=0.1, help="")
parser.add_argument( "--max_loc", type=float, default=0.3, help="")
parser.add_argument( "--min_glob", type=float, default=0.3, help="")
parser.add_argument( "--max_loc", type=float, default=1.0, help="")

parser.add_argument( "--data_dir", type=str, default="data1/1K_New/", help="")
parser.add_argument( "--train_dir", type=str, default="train", help="")
parser.add_argument( "--val_dir", type=str, default="val", help="")
parser.add_argument( "--subset_classes", type=int, default=10, help="number of classes to use from training set")
parser.add_argument( "--ckpt_path", type=str, default=None, help="path to checkpoint")

args = parser.parse_args()

class DINO(pl.LightningModule): 
    def __init__(self, backbone,input_dim, num_augment,
        optimizer_type, scheduler_type="warmup_cosine", lr=0.03, weight_decay=0.0001, warmup_epochs=10, warmup_start_lr=0.0, min_lr=0.0, max_epochs=100, 
        classifier_lr: float =1e-4,num_classes: int=1000,knn_k: int = 20,
         num_global_views=2): 

        super().__init__()
        self.num_augment= num_augment
        self.num_global_views= num_global_views
        self.optimizer= optimizer_type
        self.scheduler= scheduler_type
        self.lr=lr
        self.warmup_start_lr= warmup_start_lr
        self.min_lr= min_lr
        self.max_epochs= max_epochs
        self.weight_decay= weight_decay
        self.warmup_epochs= warmup_epochs
        ## Linear Classifier and KNN 
        self.classifier_lr = classifier_lr
        self.num_classes= num_classes

        self.student_backbone= backbone
        self.teacher_backbone= copy.deepcopy(backbone)

        self.student_head= DINOProjectionHead(input_dim, 512, 64, 2048,freeze_last_layer=1)
        self.teacher_head= DINOProjectionHead(input_dim, 512,64, 2048,)

        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)
        
        self.criterion= DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)
        self.classifier = nn.Linear(self.features_dim, num_classes)

        if self.knn_eval:
            self.knn = WeightedKNNClassifier(k=self.knn_k, distance_fx="euclidean")


    def forward(self, x):
        student_output= self.student_backbone(x).flatten(start_dim=1)
        student_output= self.student_head(student_output)
        return student_output
    
    def teacher_forward(self, x):
        teacher_output= self.teacher_backbone(x).flatten(start_dim=1)
        teacher_output= self.teacher_head(teacher_output)
        return teacher_output
    
    def training_steps(self, batch, batch_idx):
        update_momentum(self.student_backbone, self.teacher_backbone, m=0.996)
        update_momentum(self.student_head, self.teacher_head, m=0.996)

        all_views, _, _ = batch
        all_views= [views.to(self.deivce) for views in all_views]
        global_views= all_views[:self.num_global_views*self.num_augment]

        student_output=[self.forward(x) for x in all_views]
        teacher_output=[self.teacher_forward(x) for x in global_views]
        loss= self.criterion( teacher_output, student_output, epoch=self.current_epoch)
        self.log("train_loss", loss)
        return loss
        
    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self) -> Tuple[List, List]:
            """Collects learnable parameters and configures the optimizer and learning rate scheduler.

            Returns:
                Tuple[List, List]: two lists containing the optimizer and the scheduler.
            """

            # collect learnable parameters
            idxs_no_scheduler = [
                i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
            ]

            # select optimizer
            if self.optimizer == "sgd":
                optimizer = torch.optim.SGD
            elif self.optimizer == "adam":
                optimizer = torch.optim.Adam
            elif self.optimizer == "adamw":
                optimizer = torch.optim.AdamW
            else:
                raise ValueError(f"{self.optimizer} not in (sgd, adam, adamw)")

            # create optimizer
            optimizer = optimizer(
                self.learnable_params,
                #lr=self.lr,
                lr=self.lr,
                weight_decay=self.weight_decay,
                ## Continue adding parameters here
                #**self.extra_optimizer_args,
            )
            # optionally wrap with lars
            # if self.lars:
            #     assert self.optimizer == "sgd", "LARS is only compatible with SGD."
            #     optimizer = LARSWrapper(
            #         optimizer,
            #         eta=self.eta_lars,
            #         clip=self.grad_clip_lars,
            #         exclude_bias_n_norm=self.exclude_bias_n_norm,
            #     )

            if self.scheduler == "none":
                return optimizer

            if self.scheduler == "warmup_cosine":
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.warmup_epochs,
                    max_epochs=self.max_epochs,
                    warmup_start_lr=self.warmup_start_lr,
                    eta_min=self.min_lr,
                )
            elif self.scheduler == "cosine":
                scheduler = CosineAnnealingLR(optimizer, self.max_epochs, eta_min=self.min_lr)
            elif self.scheduler == "step":
                scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
            else:
                raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

            # if idxs_no_scheduler:
            #     partial_fn = partial(
            #         static_lr,
            #         get_lr=scheduler.get_lr,
            #         param_group_indexes=idxs_no_scheduler,
            #         lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
            #     )
            #     scheduler.get_lr = partial_fn

            # return [optimizer], [scheduler]

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "backbone", "params": self.backbone.parameters()},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
            },
        ]

seed_everything(5)
transform_kwargs={
        "brightness": [0.4, 0.4],
        "contrast":  0.4,
        "saturation": 0.2,
        "hue": 0.1,
        "color_jitter_prob": 0.8,
        "gray_scale_prob":   0.2,
        "horizontal_flip_prob":  0.5,
        "gaussian_prob":0.5,
        "solarization_prob": 0.1,
    }
mulda_kwargs={
    "rda_num_ops": 2, 
    "rda_magnitude": 9,
    "ada_policy": "imagenet",
    "fda_policy":"imagenet",
}

transform = prepare_transform(args.loadertype, args.num_augment_trategy,transform_kwargs, mulda_kwargs) 
transform = prepare_n_crop_transform_mv_ma(transform,  num_crops_per_aug=args.num_crops_per_aug,num_crop_glob=args.num_crop_glob, crop_size_glob=args.crop_size_glob,
                                               num_crop_loc=args.num_crop_loc, crop_size_loc=args.crop_size_loc, #crop_type=args.crop_type,
                                               min_loc=args.min_scale_loc, max_loc=args.max_scale_loc,  min_glob=args.min_scale_glob, max_glob=args.max_scale_glob, #shuffle_crop_transform=args.shuffle_transforms_crops
                                               )
train_dataset = prepare_datasets(
            args.loadertype,
            transform,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            no_labels= None, #args.no_labels,
            subset_class_num=args.subset_classes, 
        )
train_loader = prepare_dataloader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )

callbacks=[] 
# wandb logging

model= DINO( backbone= torchvision.models.resnet18(),input_dim=512, num_augment=2,
        optimizer_type="sgd")
if args.wandb:
    wandb_logger = WandbLogger(
        name=args.name,
        project=args.project,
        entity=args.entity,
        offline= False, #args.offline,
        group = args.experiment_type,
        job_type = args.job_name,
    )
    wandb_logger.watch(model, log="gradients", log_freq=100)
    wandb_logger.log_hyperparams(args)

    # lr logging
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

trainer = Trainer.from_argparse_args(
        args,
        #fast_dev_run= True,
        # gradient_clip_val=0.6, 
        # gradient_clip_algorithm="value",
        gpus=[6, 7],
        # num_nodes=1,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        enable_checkpointing=False,)


trainer.fit(model, train_loader, ckpt_path=args.ckpt_path)