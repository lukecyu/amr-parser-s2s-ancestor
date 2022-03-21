# amr-parser-s2s-ancestor

The code works fine with `python` 3.8.8 and `torch` 1.6.0.
torch=1.6.0

## Installation
```shell script
cd amr-parser-s2s-ancestor
pip install -r requirements.txt
pip install -e .
mkdir data/tmp/
```

## Train
Modify the dataset path in the configuration file `configs/config.yaml` before training (see explanation from the last few lines in the file). We also provide the hyperparaters shown in the paper for the experiment of AMR2 with recategorization. Modify it if you need.  

The `save_dir` in the file `train.sh` is the directory under `runs/` where the checkpoint files are saved. Modify it if you need.

We provide three different training bash scripts. `train_tune_parameter.sh` corresponds to the experiment of tuning the parameter. `train_layer` corresponds to the experiments that the ancestor matrix is on different layers (you can modify the `layer_parents_ids` to set on different layers). `train_head.sh` corresponds to the experiment that the ancestor matrix is on different number of heads (you can modify the `parents_attention_number` to set on different number of heads).

We also provide two different confiuration files under the directory `configs/`. `config_no_raca.yaml` is used when there is no racategorization preprocess steps while `config.yaml` is used with the recategorization preprocess.

Run the following script
```shell script
bash train.sh
```
and results are in `runs/`

## Evaluate
Modify `test.sh`

```shell script
bash test.sh
```
`gold.amr.txt` and `pred.amr.txt` will contain, respectively, the concatenated gold and the predictions.

To reproduce our paper's results, you will also need need to run the [BLINK](https://github.com/facebookresearch/BLINK) 
entity linking system on the prediction file (`data/tmp/amr2.0/pred.amr.txt` in the previous code snippet). 
To do so, you will need to install BLINK, and download their models:
```shell script
git clone https://github.com/facebookresearch/BLINK.git
cd BLINK
pip install -r requirements.txt
sh download_blink_models.sh
cd models
wget http://dl.fbaipublicfiles.com/BLINK//faiss_flat_index.pkl
cd ../..
```
Then, you will be able to launch the `blinkify.py` script:
```shell
python bin/blinkify.py \
    --datasets data/tmp/amr2.0/pred.amr.txt \
    --out data/tmp/amr2.0/pred.amr.blinkified.txt \
    --device cuda \
    --blink-models-dir BLINK/models/
```

To have comparable Smatch scores you will also need to use the scripts available at https://github.com/mdtux89/amr-evaluation

## Acknowledgements
We adopted the code framework and some modules or code snippets from [Spring](https://github.com/SapienzaNLP/spring). Thanks to this project!
