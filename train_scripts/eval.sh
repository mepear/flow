python train/eval_myppo.py test-stl-nw-profit-X1-open\
    --experiment-name=test-stl-nw-noprofit-X1-open \
    --algo=ppo \
    --verbose \
    --num-processes=50 \
    --seed=0 \
    --eval-ckpt=100 \
    --disable-render-during-eval