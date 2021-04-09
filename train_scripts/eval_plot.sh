python train/eval_myppo.py test-stl-nw-profit-X1-1-large \
    --experiment-name=test-stl-nw-profit-X1-1-large_test2 \
    --algo=ppo \
    --num-processes=100 \
    --verbose \
    --eval-ckpt=96 \
    --plot-congestion \
    --disable-render-during-eval