python bin/train_experiment.py \
    --config configs/config_new_2.yaml \
    --cuda 2 \
    --add_parents_attention \
    --parents_attention_number 6 \
    --max_length 512 \
    --save_dir parents-attention-6_amr2-reca_0.00003_0.25_0.004_20
