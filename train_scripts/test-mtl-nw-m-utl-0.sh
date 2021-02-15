python train/train.py \
    test-mtl-nw-m-utl-0 \
    --experiment-name=test-mtl-nw-m-utl-0 \
    --num-env-steps=100000000 \
    --algo=ppo \
    --use-gae \
    --lr=2.5e-4 \
    --clip-param=0.1 \
    --value-loss-coef=0.5 \
    --num-processes=1 \
    --eval-num-processes=1 \
    --num-steps=500 \
    --num-mini-batch=4 \
    --log-interval=1 \
    --save-interval=10 \
    --eval-interval=20 \
    --use-linear-lr-decay \
    --popart-reward \
    --entropy-coef=0.01 \
    --gamma=0.999
