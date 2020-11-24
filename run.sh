python train/train.py \
    grid_nxm \
    --experiment-name=grid_nxm \
    --num-env-steps=10000000 \
    --algo=ppo \
    --use-gae \
    --lr=2.5e-4 \
    --clip-param=0.1 \
    --value-loss-coef=0.5 \
    --num-processes=1 \
    --num-steps=100 \
    --num-mini-batch=4 \
    --log-interval=1 \
    --use-linear-lr-decay \
    --entropy-coef=0.01 \
    --rl_trainer stable-baselines \
    --render_during_training