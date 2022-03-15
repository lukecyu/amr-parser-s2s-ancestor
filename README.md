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
Modify `config.yaml` in `configs`. 

```shell script
bash train.sh
```
Results in `runs/`

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
