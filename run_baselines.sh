#CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "rand_aug" --folds 0 1 2 4 5 --optim Adam --lr 1e-4 --rand_aug True


#CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "gamma" --folds 4 --optim Adam --lr 1e-4 



# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 4 --optim SGD --lr 1e-4 --PIL_tfms Sharpness
# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 4 --optim SGD --lr 1e-4 --PIL_tfms Brightness
# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 4 --optim SGD --lr 1e-4 --PIL_tfms Contrast
# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 4 --optim SGD --lr 1e-4 --PIL_tfms Solarize SolarizeAdd
# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 4 --optim SGD --lr 1e-4 --PIL_tfms Posterize
# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 4 --optim SGD --lr 1e-4 --PIL_tfms Rotate
# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 4 --optim SGD --lr 1e-4 --PIL_tfms TranslateXRel TranslateYRel
# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 4 --optim SGD --lr 1e-4 --PIL_tfms ShearX ShearY

#CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 3 0 1 2 4 5 --optim Adam --lr 1e-4 --rand_aug True

# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --rand_aug True --folds 5 --optim Adam --lr 1e-4



# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default"  --folds 0  --optim Adam --lr 1e-4

# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "gamma_fcm" --fcm_mask "wm" --folds 4 0  --optim Adam --lr 1e-4

# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "gamma_fcm" --fcm_mask "csf" --folds 4 0  --optim Adam --lr 1e-4

# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "gamma_fcm" --fcm_mask "all" --folds 4 0  --optim Adam --lr 1e-4

CUDA_VISIBLE_DEVICES=0 python main_oracle.py --aug_type "default" --optim Adam --lr 1e-4 \
     --folds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 --rand_aug True
#CUDA_VISIBLE_DEVICES=0 python main_oracle.py --aug_type "default" --rand_aug True --optim Adam --lr 1e-4 \
#     --folds 14 15 16 17


# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --rand_aug True --folds 5 --optim SGD --lr 1e-3


# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 3 --optim SGD --lr 1e-3 --PIL_tfms Sharpness
# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 3 --optim SGD --lr 1e-3 --PIL_tfms Brightness
# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 3 --optim SGD --lr 1e-3 --PIL_tfms Contrast
# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 3 --optim SGD --lr 1e-3 --PIL_tfms Solarize SolarizeAdd
# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 3 --optim SGD --lr 1e-3 --PIL_tfms Posterize
# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 3 --optim SGD --lr 1e-3 --PIL_tfms Rotate
# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 3 --optim SGD --lr 1e-3 --PIL_tfms TranslateXRel TranslateYRel
# CUDA_VISIBLE_DEVICES=0 python main.py --aug_type "default" --folds 3 --optim SGD --lr 1e-3 --PIL_tfms ShearX ShearY