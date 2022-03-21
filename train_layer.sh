python bin/train_experiment.py \
    --config configs/config_new_1.yaml \
    --cuda 1 \
    --add_parents_attention \
    --layer_parents \
    --layer_parents_ids 4 5 6 7 \
    --max_length 512 \
    --save_dir parents-attention-layer_4567_amr2-reca_0.00003_0.25_0.004_20
