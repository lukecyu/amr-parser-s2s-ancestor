python bin/predict_amrs.py \
    --datasets data/amr_2_reca/test.txt.features.preproc \
    --gold-path data/tmp/amr_2.0/gold_parents-attention-0_amr2-reca_tune-constant-init-only-head-100_0.00003_0.25_0.004_20_exp.txt \
    --pred-path data/tmp/amr_2.0/pred_parents-attention-0_amr2-reca_tune-constant-init-only-head-100_0.00003_0.25_0.004_20_exp.txt \
    --checkpoint runs/parents-attention-0-new_amr2-reca_tune-constant-init-only-head-100_0.000003_0.25_0.004_20/best-smatch_checkpoint_31_0.8447.pt \
    --beam-size 5 \
    --batch-size 500 \
    --device cuda:5 \
    --parents_attention_number 0 \
    --tune_attention \
    --attention_form add_1 \
    --use-recategorization \
    --max_length 1024 \
    --penman-linearization --use-pointer-tokens --add_parents_attention \