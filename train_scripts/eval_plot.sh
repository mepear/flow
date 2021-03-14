python train/eval_myppo.py test-stl-nw-profit-X1-large \
    --experiment-name=test-stl-nw-profit-X1-large \
    --algo=ppo \
    --num-processes=100 \
    --verbose \
    --eval-ckpt=307 \
    --plot-congestion \
    --disable-render-during-eval