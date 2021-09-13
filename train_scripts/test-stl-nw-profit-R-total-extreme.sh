python train/train.py \
    test-stl-nw-profit-R-total-extreme \
    --experiment-name=test-stl-nw-profit-R-total-extreme \
    --num-env-steps=1600000000 \
    --algo=ppo \
    --use-gae \
    --lr=2.5e-4 \
    --clip-param=0.2 \
    --value-loss-coef=0.5 \
    --num-envs=800 \
    --num-actors=8 \
    --num-splits=2 \
    --eval-num-processes=50 \
    --num-steps=500 \
    --num-mini-batch=4 \
    --log-interval=1 \
    --save-interval=32 \
    --eval-interval=32 \
    --use-linear-lr-decay \
    --popart-reward \
    --entropy-coef=0.01 \
    --gamma=0.999 \
    --queue-size=5 \
    --reuse=4
