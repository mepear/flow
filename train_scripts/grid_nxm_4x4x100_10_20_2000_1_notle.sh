python train/train.py \
    grid_nxm_4x4x100_10_20_2000_1_notle \
    --experiment-name=grid_nxm_4x4x100_10_20_2000_1_notle \
    --num-env-steps=100000000 \
    --algo=ppo \
    --use-gae \
    --lr=2.5e-4 \
    --clip-param=0.1 \
    --value-loss-coef=0.5 \
    --num-processes=30 \
    --num-steps=2000 \
    --num-mini-batch=4 \
    --log-interval=1 \
    --save-interval=10 \
    --eval-interval=20 \
    --use-linear-lr-decay \
    --entropy-coef=0.01 
