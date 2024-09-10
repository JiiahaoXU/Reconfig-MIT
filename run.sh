CUDA_VISIBLE_DEVICES=7 python train.py --model Reconfig_MIT --bidirect --cuda --n_cpu 4 \
--data_ratio 0.1 --noise_level 5 --eva_epoch 1 --n_epochs 400 \
--regist \
--extended_loss --EL_lamda 10 \
--reconfig_mit --init-density 0.1 --final-density 0.7 --warmup_epoch 50 --final-grow-epoch 200 \
--update-frequency 1000 --method GMP --rm-first --regrow


# Hi there, here is a guide for how the run the experiments
# Before training, please put the dataset on ../data/.
# If you want to run the CycleGAN baseline, please run:
    # CUDA_VISIBLE_DEVICES=0 python train.py --model DeCycleGan --bidirect --cuda --n_cpu 4 \
    # --data_ratio 0.1 --noise_level 5 --eva_epoch 5 --n_epochs 400

# If you want to run the RegGAN, please run:
    # CUDA_VISIBLE_DEVICES=0 python train.py --model DeCycleGan --bidirect --cuda --n_cpu 4 \
    # --data_ratio 0.1 --noise_level 5 --eva_epoch 5 --n_epochs 400 \
    # --regist   # The only difference is here.

# If you want to run AdaptGAN, please run:
    # CUDA_VISIBLE_DEVICES=0 python train.py --model DeCycleGan --bidirect --cuda --n_cpu 4 \
    # --data_ratio 0.1 --noise_level 5 --eva_epoch 5 --n_epochs 400 \
    # --regist \
    # --extended_loss --EL_lamda 10 \
    # --sparse --init-density 0.1 --final-density 0.7 --warmup_epoch 50 --final-grow-epoch 200 \
    # --update-frequency 1000 --method GMP --rm-first --regrow