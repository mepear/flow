config_path=$1

config=`basename ${config_path%.*}`

echo $config


python train/env_debug.py $config \
    --algo=ppo \
    --num-processes=1 \
    --verbose \
    --eval-ckpt=$ckpt \
    --plot-congestion
