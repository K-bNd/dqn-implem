# Deep QLearning Network Implementation

PyTorch implementation of [Human-Level Control through Deep Reinforcement Learning](https://daiwk.github.io/assets/dqn.pdf).

## Requirements

This project requires [Python 3.11](https://www.python.org/downloads/) to run.

You will also need to the python packages in requirements.txt using [pip](https://pip.pypa.io/en/stable/).

I am using [Weights & Biases](https://wandb.ai/) to monitor the training data so you will need an account if you plan on retraining the model.

```bash
pip install -r requirements.txt
```

## Inference

You can run the current model using this code:
```py
python eval.py
```

## Training

You can retrain the model using this code:
```py
python train.py
```