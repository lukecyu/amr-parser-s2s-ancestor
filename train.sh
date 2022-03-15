python bin/train.py \
    --config configs/config.yaml \
    --direction amr \
    --cuda 0 \
    --add_parents_attention \
    --tune_attention \
    --attention_form add_1 \
    --parents_attention_number 0 \
    --max_length 512 \
    --save_dir parents-attention-0-new_amr2-reca_tune-constant-init-only-head-100_0.000003_0.25_0.004_20
