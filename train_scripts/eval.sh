python train/eval_myppo.py test-stl-nw-profit-X1-large \
    --experiment-name=test-stl-nw-profit-X1-large \
    --algo=ppo \
    --verbose \
    --num-processes=50 \
    --seed=237934 \
    --eval-ckpt=100 \
    --disable-render-during-eval \
#    --plot-congestion \
#    --random-rate=0 \