python train/eval_myppo.py $1 \
    --experiment-name=$1 \
    --algo=ppo \
    --num-processes=1 \
    --verbose \
    --eval-ckpt=$2
