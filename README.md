# Dependency Parser with Biaffine Attention and BERT Embeddings

This repository contains code for the dependency parser from the paper
[Parsers Know Best: German PP Attachment Revisited](https://aclanthology.org/2020.coling-main.185/)
published at COLING 2020.

## Usage

### Requirements
* Python 3.6
* Clone the `bert` repository:
  ```shell
  git clone https://github.com/google-research/bert.git
  ```
* Install dependencies in [requirements.txt](requirements.txt)

### Training
Run:
```shell
PYTHONPATH=`pwd` python nnparser/model/experiment.py train --help
```
to see possible arguments.

For example, to train a model on the sample dataset, run:

```shell
PYTHONPATH=`pwd` python nnparser/model/experiment.py train \
  --train_file data/sample/train.conll \
  --dev_file data/sample/dev.conll \
  --test_file data/sample/test.conll \
  --model_dir runs/sample \
  --word_emb_file data/sample/embs.txt \
  --word_dim 5 \
  --tag_dim 3 \
  --hidden_dim 7 \
  --num_lstms 2 \
  --interval 1 \
  --max_epoch 2 \
  --shuffle False \
  --num_train_buckets 2 \
  --num_dev_buckets 2 \
  --character_embeddings True \
  --char_dim 4 \
  --char_hidden_dim 7 \
  --working_batch 5000 \
  --bert_fine_tuning False
```

### Evaluation
Run:
```shell
PYTHONPATH=`pwd` python nnparser/model/experiment.py eval --help
```
to see possible arguments.

For example, to evaluate a trained model on the sample dataset, run:
```shell
PYTHONPATH=`pwd` python nnparser/model/experiment.py eval \
  --test_file data/sample/test.conll \
  --output_file runs/sample/results.conll \
  --model_dir runs/sample
```

### Tensorboard

```shell
tensorboard --logdir runs/sample/log
```


## Reproduction

All trained models contain:
* File `config.cfg` that records all parameters used to produce the model.
* Folder `log` records training and evaluation metrics, which can be viewed by `tensorboard`.
* See more information at [data](data) and [models](models).


## Citation

```bib
@inproceedings{do-rehbein-2020-parsers,
    title = "Parsers Know Best: {G}erman {PP} Attachment Revisited",
    author = "Do, Bich-Ngoc and Rehbein, Ines",
    editor = "Scott, Donia and Bel, Nuria and Zong, Chengqing",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2020.coling-main.185",
    doi = "10.18653/v1/2020.coling-main.185",
    pages = "2049--2061",
}
```