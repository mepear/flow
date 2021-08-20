config_path=$1
model_path=$2

config=`basename ${config_path%.*}`
model=`basename $model_path`

echo $config
echo $model

for ckpt in {0..100..10}
# for ckpt in $3
do
    echo $ckpt
    python train/eval_myppo.py $config \
        --experiment-name=$model \
        --algo=ppo \
        --num-processes=100 \
        --eval-ckpt=$ckpt \
        --disable-render-during-eval \
        --plot-congestion 2>&1 | tee $model_path/$ckpt.log.txt
done