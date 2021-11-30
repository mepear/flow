python train/eval_open_env.py test-stl-nw-profit-X1-open-for-test \
    --experiment-name=test-stl-nw-noprofit-X1-open \
    --algo=ppo \
    --eval-ckpt=100 \
    --verbose \
    --seed=0 \
    --num-processes=50 \
    --disable-render-during-eval