python train/train.py \
    grid_nxm_4x4x50_single_10_20_500_1_uniform_mode_5 \
    --experiment-name=grid_nxm_4x4x50_single_10_20_500_1_uniform_mode_5_v2 \
    --num-env-steps=50000000 \
    --algo=ppo \
    --use-gae \
    --lr=2.5e-4 \
    --clip-param=0.1 \
    --value-loss-coef=0.5 \
    --num-processes=100 \
    --num-steps=500 \
    --num-mini-batch=4 \
    --log-interval=1 \
    --save-interval=10 \
    --eval-interval=20 \
    --use-linear-lr-decay \
    --popart-reward \
    --entropy-coef=0.01 
