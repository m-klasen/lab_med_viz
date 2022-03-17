#python main.py --aug_type "cutout" --folds 3
#python main_spottune.py --slice_sampling_interval 1 --nid 3 --baseline_exp_path "baseline_results/baseline_cutout" --aug_type "cutout" --folds 10 11 12 13 14
#python main_spottune.py --slice_sampling_interval 12 --nid 2 --baseline_exp_path "baseline_results/baseline_cutout" --aug_type "cutout" --folds 10 11 12 13 14
#python main_spottune.py --slice_sampling_interval 48 --nid 1 --baseline_exp_path "baseline_results/baseline_cutout" --aug_type "cutout" --folds 10 11 12 13 14

#python main.py --aug_type "brightness_contrast" --folds 3
#python main_spottune.py --slice_sampling_interval 1 --nid 3 --baseline_exp_path "baseline_results/baseline_brightness_contrast" --aug_type "brightness_contrast" --folds 10 11 12 13 14
#python main_spottune.py --slice_sampling_interval 12 --nid 2 --baseline_exp_path "baseline_results/baseline_brightness_contrast" --aug_type "brightness_contrast" --folds 10 11 12 13 14

CUDA_VISIBLE_DEVICES=0 python main_spottune.py --slice_sampling_interval 48 --nid 1 --baseline_exp_path "baseline_results/baseline_focal_lovasz_adam_default" \
                                                                                --aug_type "default" --folds 15 16 17 18 19

CUDA_VISIBLE_DEVICES=0 python main_spottune.py --slice_sampling_interval 12 --nid 2 --baseline_exp_path "baseline_results/baseline_focal_lovasz_adam_default" \
                                                                                --aug_type "default" --folds 15 16 17 18 19

CUDA_VISIBLE_DEVICES=0 python main_spottune.py --slice_sampling_interval 1 --nid 3 --baseline_exp_path "baseline_results/baseline_focal_lovasz_adam_default" \
                                                                                --aug_type "default" --folds 15 16 17 18 19

CUDA_VISIBLE_DEVICES=0 python main_spottune.py --slice_sampling_interval 48 --nid 1 --baseline_exp_path "baseline_results/baseline_focal_lovasz_adam_rand_aug_default_v1" \
                                                                                --aug_type "default" --folds 15 16 17 18 19 --rand_aug True

CUDA_VISIBLE_DEVICES=0 python main_spottune.py --slice_sampling_interval 12 --nid 2 --baseline_exp_path "baseline_results/baseline_focal_lovasz_adam_rand_aug_default_v1" \
                                                                                --aug_type "default" --folds 15 16 17 18 19 --rand_aug True

CUDA_VISIBLE_DEVICES=0 python main_spottune.py --slice_sampling_interval 1 --nid 3 --baseline_exp_path "baseline_results/baseline_focal_lovasz_adam_rand_aug_default_v1" \
                                                                                --aug_type "default" --folds 15 16 17 18 19 --rand_aug True

# CUDA_VISIBLE_DEVICES=0 python main_spottune.py --slice_sampling_interval 1 --nid 3 --aug_type "default" \
# --baseline_exp_path "baseline_results/baseline_focal_lovasz_adam_default_Posterize" --folds 15 16 17 18 19



# CUDA_VISIBLE_DEVICES=0 python main_spottune.py --slice_sampling_interval 48 --nid 1 --baseline_exp_path "baseline_results/baseline_focal_lovasz_SGD_default_None" \
#                                                                                 --aug_type "default" --folds 15 16 17 18 19

# CUDA_VISIBLE_DEVICES=0 python main_spottune.py --slice_sampling_interval 12 --nid 2 --baseline_exp_path "baseline_results/baseline_focal_lovasz_SGD_default_None" \
#                                                                                 --aug_type "default" --folds 15 16 17 18 19

# CUDA_VISIBLE_DEVICES=0 python main_spottune.py --slice_sampling_interval 1 --nid 3 --baseline_exp_path "baseline_results/baseline_focal_lovasz_SGD_default_None" \
#                                                                                 --aug_type "default" --folds 15 16 17 18 19

# CUDA_VISIBLE_DEVICES=0 python main_spottune.py --slice_sampling_interval 48 --nid 1 --baseline_exp_path "baseline_results/baseline_focal_lovasz_SGD_rand_aug_default" \
#                                                                                 --aug_type "default" --folds 15 16 17 18 19

# CUDA_VISIBLE_DEVICES=0 python main_spottune.py --slice_sampling_interval 12 --nid 2 --baseline_exp_path "baseline_results/baseline_focal_lovasz_SGD_rand_aug_default" \
#                                                                                 --aug_type "default" --folds 15 16 17 18 19

# CUDA_VISIBLE_DEVICES=0 python main_spottune.py --slice_sampling_interval 1 --nid 3 --baseline_exp_path "baseline_results/baseline_focal_lovasz_SGD_rand_aug_default" \
#                                                                                  --aug_type "default" --folds 15 16 17 18 19