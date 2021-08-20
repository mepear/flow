python train/plot_correlation.py test-stl-nw-X1-for-eval \
    --checkpoint=$1 \
    --experiment-name=test-stl-nw-profit-X1-large \
    --algo=ppo \
    --num-processes=50 \
    --disable-render-during-eval