python train/eval_myppo.py test-stl-nw-X1-for-eval \
    --experiment-name=test-stl-nw-profit-X1-large \
    --algo=ppo \
    --num-processes=2 \
    --verbose \
    --eval-ckpt=50 \
    --disable-render-during-eval