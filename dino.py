import copy 
import torch 
from torch import nn 
import pytorch_lightning as pl
import torch.nn.functional as F
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
from MV_MA_SSL.utils.metrics import accuracy_at_k, weighted_mean
## Dataloader Import  Logging import
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb
## Dataloader Import 
from MV_MA_SSL.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform_mv_ma,
    prepare_transform,
)
from MV_MA_SSL.utils.classification_dataloader import prepare_data as prepare_data_classification

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( "--loadertype", type=str, default="mv_ma", help="dataloader type")
parser.add_argument( "--Dataaugment_strategy", type=dict, default={"strategy": "SimCLR_RA","num_strategy": 2}, help="Type of augmentation Strategy ")
parser.add_argument( "--num_crops_per_aug", type=list, default=[1, 1], help="Strategy augmentation type")
parser.add_argument( "--num_crop_glob", type=int, default=2, help="")
parser.add_argument( "--crop_size_glob", type=int, default=224, help="")
parser.add_argument( "--num_crop_loc", type=int, default=4, help="")
parser.add_argument( "--crop_size_loc", type=int, default=96, help="")
parser.add_argument( "--crop_type", type=str, default="random_uniform", help="")
parser.add_argument( "--min_loc", type=float, default=0.1, help="")
parser.add_argument( "--max_loc", type=float, default=0.3, help="")
parser.add_argument( "--min_glob", type=float, default=0.3, help="")
parser.add_argument( "--max_glob", type=float, default=1.0, help="")

parser.add_argument( "--data_dir", type=str, default="/data1/1K_New/", help="")
parser.add_argument( "--train_dir", type=str, default="train", help="")
parser.add_argument( "--val_dir", type=str, default="val", help="")
parser.add_argument( "--subset_classes", type=int, default=10, help="number of classes to use from training set")
parser.add_argument( "--ckpt_path", type=str, default=None, help="path to checkpoint")
parser.add_argument( "--batch_size", type=int, default=100, help="Training batch_size")
parser.add_argument( "--num_workers", type=int, default=20, help="Training batch_size")


parser.add_argument( "--name", type=str, default="test new method", help="Training batch_size")
parser.add_argument( "--project", type=str, default="MVAR_SSRL", help="Training batch_size")
parser.add_argument( "--entity", type=str, default="mlbrl", help="Training batch_size")
parser.add_argument( "--experiment_type", type=str, default="ablation", help="Training batch_size")
parser.add_argument( "--job_name", type=str, default="vit_full_imagenet", help="Training batch_size")
parser.add_argument( "--wandb_logs", type=bool, default=True, help="Using Wanbd loging experiment")

args = parser.parse_args()


class MVAR_DINO(pl.LightningModule):
    def __init__(self, 
        
        ## Network Achitecture parameters
        backbone, input_dim: int, output_dim: int, hidden_dim: int,
        bottleneck_dim: int, use_bn: bool,freeze_last_layer: int,   
        pretrain_epochs: int, 
        
        ## Loss function compute correlate DataAugmentation Policy for Global Views
        num_augmentation_strategy: int, num_glob_views: int,
        
        ## Optimizer parameters 
        accumulate_grad_batches: int= None, 
        optim_type: str = "adamw",scheduler_type: str="warmup_cosine", 
        lr: float = 0.001, weight_decay: float = 0.0001, warmup_epochs: int = 10,
        warmup_start_lr: float = 0.001, min_lr: float = 0.00001, lr_decay_steps: List[int] = None,
        eta_lars: float = 0.001, grad_clip_lars: float = 1.0, 
        exclude_bias_n_norm: bool = True, lars_optim: bool = False,
        ## Loss parameters 
        tau_momentum: float= 0.99,   warmup_teacher_temp: float = 0.04, 
        teacher_temp: float = 0.04, warmup_teacher_temp_epochs: int = 30, 
        student_temp: float = 0.1, center_momentum: float = 0.9,

        ## Parameters for Online Downstream tasks 
        use_ConvNet: bool= True, 
        num_classes: int=10,
        linear_classifier_lr: float = 0.0002,
        use_knn: bool = False,
        knn_k: int=20, distance_fx: str="euclidean", 
        
        
        **kwargs):
        
   
        super().__init__()
       
        #backbone = nn.Sequential(*list(backbone.children())[:-1])
        # instead of a resnet you can also use a vision transformer backbone as in the
        # original paper (you might have to reduce the batch size in this case):
        # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        # input_dim = backbone.embed_dim
        if use_ConvNet: 
            backbone.fc = nn.Identity()
        else: 
            backbone= nn.Sequential(*list(backbone.children())[:-1])
        self.accumulate_grad_batches=accumulate_grad_batches
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(input_dim, hidden_dim, bottleneck_dim, output_dim,use_bn, freeze_last_layer)
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, hidden_dim, bottleneck_dim, output_dim,use_bn,  )
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

      
        # Optimizer Configuration Parameters 
        self.optimizer_type = optim_type 
        self.scheduler_type = scheduler_type
        self.pretrain_epochs=pretrain_epochs
        self.warmup_start_lr= warmup_start_lr 
        self.lr= lr
        self.min_lr=min_lr
        self.warmup_epochs= warmup_epochs
        self.weight_decay= weight_decay
        ## For Lars Optimizer 
        self.lars_optimizer=lars_optim 
        self.eta_lars= eta_lars
        self.exclude_bias_n_norm= exclude_bias_n_norm 
        self.grad_clip_lars= grad_clip_lars

        ## Compute loss with given N_glob_views*Num_augmentation_strategy
        self.num_glob_views = num_glob_views
        self.num_augmentation_strategy = num_augmentation_strategy

        ### Loss parameters  
        self.tau_momentum=tau_momentum
        self.criterion = DINOLoss(output_dim, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs,student_temp, center_momentum )


        ### Online Downstream task Linear evaluation and KNN 
        self.use_knn= use_knn
        if use_ConvNet: 
            self.feature_dim= self.student_backbone.inplanes
        else: 
            self.feature_dim= self.student_backbone.num_features
         
        self.classifier= nn.Linear(self.feature_dim, num_classes)
        self.classifier_lr= linear_classifier_lr
        if use_knn:
            self.knn = WeightedKNNClassifier(k=self.knn_k, distance_fx="euclidean")

        if self.accumulate_grad_batches:
            self.lr = self.lr * self.accumulate_grad_batches
            self.classifier_lr = self.classifier_lr * self.accumulate_grad_batches
            self.min_lr = self.min_lr * self.accumulate_grad_batches
            self.warmup_start_lr = self.warmup_start_lr * self.accumulate_grad_batches
    
    def base_forward(self, X: torch.Tensor) -> Dict:
        """Basic forward that allows children classes to override forward().

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """

        feats = self.student_backbone(X)
        print(feats.shape)
        logits = self.classifier(feats.detach())


        return {
            "logits": logits,
            "feats": feats,
        }
    
    def _base_shared_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        out = self.base_forward(X)
        logits = out["logits"]

        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        # handle when the number of classes is smaller than 5
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))

        return {**out, "loss": loss, "acc1": acc1, "acc5": acc5}


    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        update_momentum(self.student_backbone, self.teacher_backbone, m=self.tau_momentum)
        update_momentum(self.student_head, self.teacher_head, m=self.tau_momentum)
        _, views, _ = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:self.num_glob_views*self.num_augmentation_strategy]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        metrics={"cross_entropy_loss": loss}
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = None
    ) -> Dict[str, Any]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.

        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y].
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the batch_size (used for averaging), the classification loss
                and accuracies.
        """

        X, targets = batch
        batch_size = targets.size(0)

        out = self._base_shared_step(X, targets)

        if self.use_knn and not self.trainer.sanity_checking:
            self.knn(test_features=out.pop("feats").detach(), test_targets=targets.detach())

        metrics = {
            "batch_size": batch_size,
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
        }
        return metrics

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}

        if self.use_knn and not self.trainer.sanity_checking:
            val_knn_acc1, val_knn_acc5 = self.knn.compute()
            log.update({"val_knn_acc1": val_knn_acc1, "val_knn_acc5": val_knn_acc5})

        self.log_dict(log, sync_dist=True)


    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):

        ### ----------select optimizer type --------------
        if self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer_type == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW
        else:
            raise ValueError(f"{self.optimizer_type} not in (sgd, adam, adamw)")

        # create optimizer
        optimizer = optimizer(
            #self.parameters(),
            self.learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            #**self.extra_optimizer_args,
        )
        if self.lars_optimizer:
            assert self.optimizer_type == "sgd", "LARS is only compatible with SGD."
            optimizer = LARSWrapper(
                optimizer,
                eta=self.eta_lars,
                clip=self.grad_clip_lars,
                exclude_bias_n_norm=self.exclude_bias_n_norm,
            )

        ### --------------------- Select for scheduler the Optimizer ------------ 
        if self.scheduler_type == "none":
                return optimizer
        if self.scheduler_type == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.pretrain_epochs,
                warmup_start_lr=self.warmup_start_lr,
                eta_min=self.min_lr,
            )
        elif self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.pretrain_epochs, eta_min=self.min_lr)
        elif self.scheduler_type == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
        else:
            raise ValueError(f"{self.scheduler_type} not in (warmup_cosine, cosine, step)")
        #optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return [optimizer], [scheduler]

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "backbone", "params": self.student_backbone.parameters()},
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

transform = prepare_transform(args.loadertype, args.Dataaugment_strategy["strategy"],transform_kwargs, mulda_kwargs) 
transform = prepare_n_crop_transform_mv_ma(transform,  num_crops_per_aug=args.num_crops_per_aug,num_crop_glob=args.num_crop_glob, crop_size_glob=args.crop_size_glob,
                                               num_crop_loc=args.num_crop_loc, crop_size_loc=args.crop_size_loc, crop_type=args.crop_type,
                                               min_loc=args.min_loc, max_loc=args.max_loc,  min_glob=args.min_glob, max_glob=args.max_glob, #shuffle_crop_transform=args.shuffle_transforms_crops
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
if args.val_dir is not None:
    _, val_loader = prepare_data_classification(
        args.loadertype,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        subset_class_num=args.subset_classes, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
else: 
    val_loader=None
### ------------------ Model Hyperparameter setting SSL Pretraining ---------------------

kwargs={
    "backbone": torchvision.models.resnet50(),
    "use_ConvNet": True, # if backbone is ConvNet
    "input_dim": 2048,  
    "hidden_dim": 2048,
    "bottleneck_dim": 256,
    "output_dim": 65536,# for ViTs
    "use_bn": False, # if True, use BN in the head
    "freeze_last_layer": -1, # Number epochs keep the ProjectHead's output layer fixed.  
    "tau_momentum": 0.996,
    "accumulate_grad_batches": None, # for larger batch forward N times
    "pretrain_epochs": 10,
    
    ## Loss Computinng Hyperparameters for SSL Pretraining 
    "warmup_teacher_temp": 0.04, 
    "teacher_temp":  0.04,
    "warmup_teacher_temp_epochs": 5, 
    "student_temp": 0.1,
    "center_momentum": 0.9,

    ## Configure data Augmentation impact to compute the loss 
    "num_glob_views": args.num_crop_glob, 
    "num_augmentation_strategy": args.Dataaugment_strategy["num_strategy"],

    "optim_type": "sgd", ## if lars optim --> optim_type: "sgd"
    "lr": 0.01, # lr will overwrite the lr within the scheduler.
    "scheduler_type": "warmup_cosine",# ["cosine"(CosineAnnealingLR), "step",None]
    "min_lr": 0.001,
    "weight_decay": 0.0001,
    ## Adjust these paras if using warmup_cosine schedule
    # -----------------------#
    "warmup_start_lr": 0.0001, # base_lr
    "warmup_epochs": 2 , 
    # -----------------------#
    ## if using Lars Optimizer
    # -----------------------#
    "lars_optim": True, 
    "eta_lars": 0.001,
    "exclude_bias_n_norm": True, 
    "grad_clip_lars": False, 
    # -----------------------#

    ## Downstream Task Hyperparameters
    # KNN evaluation
    "use_knn": False, 
    "knn_k": 200,
    "distance_fx": "euclidean", 
    # Linear Evaluation
    "linear_classifier_lr": 0.1,
    "num_classes": args.subset_classes,

    }

model= MVAR_DINO(**kwargs)

### --------------------- Logging model during training --------------------- ###
callbacks=[] 
if args.wandb_logs:
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

## ---------------Debug Dataloader ---------------------##
# for x1, x2, x3 in train_loader:
#     #print(im.shape)
#     # unpack
#     #x1, x2, x3, x4 = im
#     print(len(x2))
# #     torch.save(x2, "visualize_tensor_1", pickle_module=pickle)
#     print("Rick Double Check Global Views shape", x2[args.num_crop_glob*2].shape )
#     #print("Rick Double Check  Local Views shape", x2[args.num_crop_local*2].shape)
# # #     #x1_=x2[7]
# # #     print(x1.shape)
# # #     #print(x1_.shape)
# # #     print(x3.shape)
#     break

### --------------------- Training --------------------- ###
trainer = Trainer(
       # args,
        #fast_dev_run= True,
        # gradient_clip_val=0.6, 
        # gradient_clip_algorithm="value",
        gpus= [7],
        max_epochs=kwargs["pretrain_epochs"],
        # num_nodes=1,max_epochs
        logger=wandb_logger if args.wandb_logs else None,
        callbacks=callbacks,
        enable_checkpointing=False,
        strategy="ddp",
        check_val_every_n_epoch=2,
        )
trainer.fit(model, train_loader,val_loader)