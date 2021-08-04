config_path=$1
model_path=$2
ckpt=$3

config=`basename ${config_path%.*}`
model=`basename $model_path`

python train/eval_myppo.py $config \
    --experiment-name=$model \
    --algo=ppo \
    --num-processes=1 \
    --verbose \
    --eval-ckpt=$ckpt
