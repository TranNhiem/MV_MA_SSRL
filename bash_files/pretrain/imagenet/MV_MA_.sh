python3 ../../../mv_ma_pretrain_edit.py \
    --dataset mv_ma \
    --mvar_training True \
    --experiment_type ablation \
    --job_name multiaugment_ablation_subset \
    --backbone resnet50 \
    --data_dir /data1/1K_New/ \
    --train_dir train \
    --val_dir val \
    --subset_classes 300 \
    --max_epochs 100 \
    --gpus 0,1,2,3,4,5,6,7 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --accumulate_grad_batches 2 \
    --classifier_lr 0.2 \
    --weight_decay 1e-6 \
    --batch_size 200 \
    --num_workers 4 \
    --num_augment_trategy RA_AA_FA \
    --num_augment_strategies 3\
    --brightness 0.4 0.4 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --color_jitter_prob 0.8 \
    --gray_scale_prob 0.2 \
    --horizontal_flip_prob 0.5 \
    --gaussian_prob 1.0  \
    --solarization_prob 0.2  \
    --crop_size 224  \
    --crop_size_glob 224 \
    --crop_size_loc 96 \
    --num_crop_glob 2 \
    --num_crop_loc 0 \
    --crop_type random_uniform \
    --min_scale_loc 0.13 \
    --max_scale_loc 0.34 \
    --min_scale_glob 0.2 \
    --max_scale_glob 1.0 \
    --rda_num_ops 2 \
    --rda_magnitude 9 \
    --ada_policy imagenet \
    --fda_policy imagenet \
    --num_crops_per_aug 1 1 1 \
    --shuffle_transforms_crops True\
    --name MV_MASSL_ablation_3_augment_RA_AA_FA_res50-Subset_300_imagenet-100ep_shuffle \
    --entity mlbrl \
    --project solo_MASSL_V2 \
    --wandb \
    --save_checkpoint \
    --method mvar \
    --proj_output_dim 512 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier \
    --alpha 0.5\
    --checkpoint_dir /data1/solo_MASSL_ckpt \
    --checkpoint_frequency 20 &&
python3 ../../../mv_ma_pretrain_edit.py \
    --dataset mv_ma \
    --mvar_training True \
    --experiment_type ablation \
    --job_name multiaugment_ablation_subset \
    --backbone resnet50 \
    --data_dir /data1/1K_New/ \
    --train_dir train \
    --val_dir val \
    --subset_classes 300 \
    --max_epochs 100 \
    --gpus 0,1,2,3,4,5,6,7 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --accumulate_grad_batches 2 \
    --classifier_lr 0.2 \
    --weight_decay 1e-6 \
    --batch_size 200 \
    --num_workers 4 \
    --num_augment_trategy RA_AA \
    --num_augment_strategies 2\
    --brightness 0.4 0.4\
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --color_jitter_prob 0.8 \
    --gray_scale_prob 0.2 \
    --horizontal_flip_prob 0.5 \
    --gaussian_prob 1.0  \
    --solarization_prob 0.2  \
    --crop_size 224  \
    --crop_size_glob 224 \
    --crop_size_loc 96 \
    --num_crop_glob 2 \
    --num_crop_loc 0 \
    --crop_type random_uniform \
    --min_scale_loc 0.13 \
    --max_scale_loc 0.34 \
    --min_scale_glob 0.2 \
    --max_scale_glob 1.0 \
    --rda_num_ops 2 \
    --rda_magnitude 9 \
    --ada_policy imagenet \
    --fda_policy imagenet \
    --num_crops_per_aug 1 1  \
    --shuffle_transforms_crops False\
    --name MV_MASSL_ablation_2_RA_AA_augment_res50-Subset_300_imagenet-100ep \
    --entity mlbrl \
    --project solo_MASSL_V2 \
    --wandb \
    --save_checkpoint \
    --method mvar \
    --proj_output_dim 512 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier \
    --alpha 0.5\
    --checkpoint_dir /data1/solo_MASSL_ckpt \
    --checkpoint_frequency 20 &&
python3 ../../../mv_ma_pretrain_edit.py \
    --dataset mv_ma \
    --mvar_training True \
    --experiment_type ablation \
    --job_name multiaugment_ablation_subset \
    --backbone resnet50 \
    --data_dir /data1/1K_New/ \
    --train_dir train \
    --val_dir val \
    --subset_classes 300 \
    --max_epochs 100 \
    --gpus 0,1,2,3,4,5,6,7 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --accumulate_grad_batches 2 \
    --classifier_lr 0.2 \
    --weight_decay 1e-6 \
    --batch_size 200 \
    --num_workers 4 \
    --num_augment_trategy AA_FA \
    --num_augment_strategies 2\
    --brightness 0.4 0.4\
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --color_jitter_prob 0.8 \
    --gray_scale_prob 0.2 \
    --horizontal_flip_prob 0.5 \
    --gaussian_prob 1.0  \
    --solarization_prob 0.2  \
    --crop_size 224  \
    --crop_size_glob 224 \
    --crop_size_loc 96 \
    --num_crop_glob 2 \
    --num_crop_loc 0 \
    --crop_type random_uniform \
    --min_scale_loc 0.13 \
    --max_scale_loc 0.34 \
    --min_scale_glob 0.2 \
    --max_scale_glob 1.0 \
    --rda_num_ops 2 \
    --rda_magnitude 9 \
    --ada_policy imagenet \
    --fda_policy imagenet \
    --num_crops_per_aug 1 1  \
    --shuffle_transforms_crops False\
    --name MV_MASSL_ablation_2_AA_FA_augment_res50-Subset_300_imagenet-100ep \
    --entity mlbrl \
    --project solo_MASSL_V2 \
    --wandb \
    --save_checkpoint \
    --method mvar \
    --proj_output_dim 512 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier \
    --alpha 0.5\
    --checkpoint_dir /data1/solo_MASSL_ckpt \
    --checkpoint_frequency 20 