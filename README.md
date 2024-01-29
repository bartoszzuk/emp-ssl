#  Extreme Multi-Patch Self Supervised Learning (EMP-SSL)

This is a minimalistic and clean reimplementation of extreme **extreme multi-patch** self supervised learning in Pytorch Lightning.
EMP-SSL was introduced in paper [EMP-SSL: Towards Self-Supervised Learning in One Training Epoch](https://arxiv.org/abs/2304.03977).
This repository is just for fun/educational purposes. Please check out the repository with the official implementation [here](https://github.com/tsb0601/EMP-SSL). 
Currently only `resnet18` model and `cifar10` dataset are supported, but it should be rather easy to add
more models and datasets.

## Installation

You will need at least **Python 3.10**.

```
# Download repository
git clone git@github.com:bartoszzuk/emp-ssl.git
cd emp-ssl

# [Optional] Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Pretraining

Just use a `pretrain.py` script as shown below. The default argument value
were used to produce reported results.

```
python pretrain.py
```

Or if you want to play around with hyperparameters...

```
python pretrain.py \
    --dataset cifar10             # path to the dataset root
    --invariance-coefficient 200  # coefficient for invariance loss (cosine similarity)
    --batch-size 100              # number of images per step
    --train-patches 20            # number of augmented patches per image in training
    --valid-patches 128           # number of augmented patches per image in validation
    --num-workers 8               # number of workers in DataLoader's
    --max-epochs 1                # number of training epochs
    --learning-rate 0.03          # learning rate in LARS optimizer
    --weight-decay 0.0001         # weight decay in LARS optimizer
    --projection-dim 1024         # dimension for output of projection head
    --hidden-dim 4096             # hidden dimension in the projection head
    --num-neighbours 20           # number of neighbours to use for KNN evaluation
    --temperature 0.07            # temperature to use for KNN evaluation
    --seed 42                     # random seed
```

After finishing pretraining you should have a Pytorch Lighting module **checkpoint** as well as 
bag-of-features **embeddings** that you should use to train linear evaluation model.


## Linear Evaluation

Just use a `evaluate.py` script as shown below. The default argument value
were used to produce reported results.

```
python evaluate.py --dataset [YOUR_EMBEDDINGS_DIRECTORY]
```

Or if you want to play around with hyperparameters...

```
python evaluate.py \
    --dataset [YOUR_EMBEDDINGS_DIRECTORY]
    --batch-size 100        # number of images per step
    --max-epochs 100        # number of training epochs
    --num-workers 8         # number of workers in DataLoader's
    --learning-rate 0.03    # learning rate for SGD optimizer
    --weight-decay 0.00005  # weight decay in SGD optimizer
    --embedding-dim 4096    # dimension of the bag-of-features embeddings
    --seed 42               # random seed
```

## Results

Results for default setups in pretraining (evaluated with a weighted KNN) and linear evaluation.

| Stage                      | Dataset | Top1 Accuracy | Top5 Accuracy |
|----------------------------|---------|---------------|---------------|
| Pretrain KNN **(1 epoch)** | Cifar10 | 74.32         | 97.13         |
| Linear Evaluate            | Cifar10 | **82.11**     | 99.15         |
