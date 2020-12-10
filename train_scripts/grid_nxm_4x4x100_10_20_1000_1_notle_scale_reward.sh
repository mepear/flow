python train/train.py \
    grid_nxm_4x4x100_10_20_1000_1_notle \
    --experiment-name=grid_nxm_4x4x100_10_20_1000_1_notle_scale_1000_popart \
    --num-env-steps=100000000 \
    --algo=ppo \
    --use-gae \
    --lr=2.5e-4 \
    --clip-param=0.1 \
    --value-loss-coef=0.5 \
    --num-processes=30 \
    --num-steps=1000 \
    --num-mini-batch=1 \
    --log-interval=1 \
    --save-interval=10 \
    --eval-interval=20 \
    --use-linear-lr-decay \
    --popart-reward \
    --reward-scale=1000 \
    --entropy-coef=0.01 
