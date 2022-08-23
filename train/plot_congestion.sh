python train/plot_congestion.py test-stl-nw-noprofit-X1-RL+random_reposition \
    --random-rate=$1 \
    --algo=ppo \
    --experiment-name=test-stl-nw-noprofit-X1-nearest_dispatch+RL-new \
    --num-processes=500 \
    --eval-ckpt=100 \
    --disable-render-during-eval \
    --experiment-name_2=test-stl-nw-noprofit-X1-RL+random_reposition-new \
    --experiment-name_3=test-stl-nw-noprofit-X1-mid_edge \
#    --plot-congestion