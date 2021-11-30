python train/eval_myppo.py test-stl-nw-profit-X1-open \
    --experiment-name=test-stl-nw-profit-X1-open \
    --algo=ppo \
    --num-processes=50 \
    --verbose \
    --eval-ckpt=67 \
    --disable-render-during-eval \
    --plot-congestion