# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from pprint import pprint

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from MV_MA_SSL.args.setup import parse_args_pretrain
from MV_MA_SSL.methods import METHODS
from MV_MA_SSL.utils.auto_resumer import AutoResumer
import torch
import pickle
from torchvision import datasets

try:
    from MV_MA_SSL.methods.dali import PretrainABC
except ImportError as e:
    print(e)
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from MV_MA_SSL.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

import types
from MVAR_Dino.main_mvar_dino import DataAugmentationMVAR_DINO
from MV_MA_SSL.utils.checkpointer import Checkpointer
from MV_MA_SSL.utils.classification_dataloader import prepare_data as prepare_data_classification
from MV_MA_SSL.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform_mv_ma,
    prepare_transform,
)

from MV_MA_SSL.utils.value_schedule import (
    Alpha_schedule,
    WD_cosine_schedule,
    Beta_schedule,
    batch_size_schedule, 
    Stochastic_Weight_Avg
)

## For multiNode training bash 
#--network host 
#MASTER_ADDR=10.0.0.6 MASTER_PORT=3300 NODE_RANK=1 WORLD_SIZE=2 bash DINO_full_data.sh
def main():
    seed_everything(5)

    args = parse_args_pretrain()

    assert args.method in METHODS, f"Choose from {METHODS.keys()}"
    #args.method=="massl"
    print(args.num_large_crops)
    # if args.num_large_crops != 2:
    #     ## Adding for multi-Views
    #     assert args.method in METHODS#=="wmse"

    MethodClass = METHODS[args.method]
    if args.dali:
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with [dali]."
        MethodClass = types.new_class(f"Dali{MethodClass.__name__}", (PretrainABC, MethodClass))

    model = MethodClass(**args.__dict__)

    # pretrain dataloader
    # note that, to support plugin, i modify the prepare_transform func (i didn't unpack the dict!!)
    if not args.dali:
        # asymmetric augmentations
        if args.unique_augs >= 1:  # note : --brightness 0.4 0.4 0.4 0.4 \  # 4 params to bypass inner chk mechnaism in sh file
            # pluggin proposed multiple-DA
            if args.dataset == "mulda" or "mulda_v1" or "mv_ma":
                transform = prepare_transform(args.dataset, args.num_augment_trategy,args.transform_kwargs, args.mulda_kwargs) 
            else: # normal case, this way plz ~ ~
                transform = [
                    prepare_transform(args.dataset, kwargs) for kwargs in args.transform_kwargs
                ]
        else:
            transform = [prepare_transform(args.dataset, args.transform_kwargs)]
        ## My Goal is Get X--> Crop it --> Two crop --> Transform   
        


        transform = prepare_n_crop_transform_mv_ma(transform,  num_crops_per_aug=args.num_crops_per_aug,num_crop_glob=args.num_crop_glob, crop_size_glob=args.crop_size_glob,
                                               num_crop_loc=args.num_crop_loc, crop_size_loc=args.crop_size_loc, crop_type=args.crop_type,
                                               min_loc=args.min_scale_loc, max_loc=args.max_scale_loc,  min_glob=args.min_scale_glob, max_glob=args.max_scale_glob, shuffle_crop_transform=args.shuffle_transforms_crops
                                               )
        
        if args.debug_augmentations:
            print("Transforms:")
            pprint(transform)

        if args.subset_classes: 
            print(f"Using Subset Dataset with {args.subset_classes} Classes")

        train_dataset = prepare_datasets(
            args.dataset,
            transform,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            no_labels=args.no_labels,
            subset_class_num=args.subset_classes, 
        )
        train_loader = prepare_dataloader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )


    # normal dataloader for when it is available
    if args.dataset == "custom" and (args.no_labels or args.val_dir is None):
        val_loader = None

    ## pluggin proposed multiple-DA
    # i'm not sure about the below line, but i also add our ds into it!!
    elif args.dataset in ["imagenet100", "imagenet",   "mulda", "mulda_v1", "mv_ma"] and args.val_dir is None:
        val_loader = None
    else:
        _, val_loader = prepare_data_classification(
            args.dataset,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            subset_class_num=args.subset_classes, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    callbacks = []

    # wandb logging
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
        # lr_monitor = LearningRateMonitor(logging_interval="epoch")
        # callbacks.append(lr_monitor)

    if args.save_checkpoint:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, args.method),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    if args.auto_umap:
        assert (
            _umap_available
        ), "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            args,
            logdir=os.path.join(args.auto_umap_dir, args.method),
            frequency=args.auto_umap_frequency,)
        
        callbacks.append(auto_umap)

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    
    ckpt_path = None
    if args.auto_resume and args.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(args.checkpoint_dir, args.method),
            max_hours=args.auto_resumer_max_hours,)
        
        resume_from_checkpoint = auto_resumer.find_checkpoint(args)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",)
            ckpt_path = resume_from_checkpoint
    
    elif args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint

    alpha_schedule = Alpha_schedule(args,args.max_epochs, 0.6,args.alpha)
   
    Weight_Decay_schedule= WD_cosine_schedule(args, args.max_epochs, args.wd_init, args.wd_final, args.weight_decay_scheduler)
    callbacks.append(alpha_schedule)
    callbacks.append(Weight_Decay_schedule)

    trainer = Trainer.from_argparse_args(
        args,
        #fast_dev_run= True,
        # gradient_clip_val=0.6, 
        # gradient_clip_algorithm="value",
        num_nodes=1,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        enable_checkpointing=False,)

    print("\n\nI'm in here \n\n")
    # it's very not good... the issue occurs in train_loader, i'm not sure which da-method cause the img have invalid size
    # # while i will go deep into each trfs 'Composer'
    # for x1, x2, x3 in train_loader:
    #     #print(im.shape)
    #     # unpack
    #     #x1, x2, x3, x4 = im
    #     print(len(x2))
    #     torch.save(x2, "visualize_tensor_1", pickle_module=pickle)
    #     print("Rick Double Check Global Views shape", x2[args.num_large_crops-2].shape )
    #     print("Rick Double Check  Local Views shape", x2[args.num_large_crops+2].shape)
    # #     #x1_=x2[7]
    # #     print(x1.shape)
    # #     #print(x1_.shape)
    # #     print(x3.shape)
    #     break


    # for x1, x2, x3 in train_loader:
    #     ## X1, X3 will be the Index of image , X2 is Batch Tensors feeding duiring training.
    #     ## Saving the Tensor Image + Image Transform Tensors 

    #     ## Noted You must Uncomment the Visualization and Debug section in pretrain_dataloader
    #     ##--> Class [FullTransformPipeline_ma_mv]
    #     print("len of Batch tensor", len(x2))
    #     torch.save(x2, "visualize_tensor_image_Global_local_transform", pickle_module=pickle)
    #     print("Rick Double Check Global Views shape", x2[args.num_large_crops-2].shape )
    #     print("Rick Double Check  Local Views shape", x2[args.num_large_crops+3].shape)


    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
