python train/eval_myppo.py test-stl-nw-profit-X1-1-large \
    --experiment-name=test-stl-nw-profit-X1-1-large \
    --algo=ppo \
    --num-processes=100 \
    --verbose \
    --eval-ckpt=198 \
    --disable-render-during-eval \
    --plot-congestion