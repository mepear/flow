python train/eval_myppo.py test-stl-nw-profit-large-out \
    --experiment-name=test-stl-nw-profit-large-out \
    --algo=ppo \
    --num-processes=100 \
    --verbose \
    --eval-ckpt=1 \
    --plot-congestion \
    --disable-render-during-eval