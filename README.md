# Deep Learning Pipeline Optimizer (IMDB Sentiment Analysis)

This project demonstrates **AutoML + Hyperparameter Optimization (HPO)** using [Optuna](https://optuna.org/) for different deep learning architectures (CNN & RNN) on the [IMDB movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

The project is structured as a proper Python package, making it easy to extend, reuse, and integrate into MLOps pipelines.

---

## Project Structure

deeplearn_pipeline_optimizer/

    data/
        imdb_loader.py  Loads & preprocesses IMDB dataset 

    models/
        cnn_text.py  CNN architecture for text classification
        rnn_text.py  RNN (GRU) architecture for text classification

    optimizers/
        optuna_runner.py  Optuna search space & optimization loop

    train/
        train_model.py  Training & evaluation helper

    main.py  CLI entry point

    requirements.txt

    README.md

---

##  How It Works

1. **Load & preprocess IMDB data** (padding, tokenization).
2. **Choose architecture** (CNN or RNN).
3. **Define hyperparameter search space** in `optuna_runner.py`.
4. **Run Optuna optimization** to find the best parameters.
5. **Train & evaluate** using the best configuration.

---

## Installation

```bash
git clone https://github.com/b-tanyileke/deeplearn_pipeline_optimizer.git
cd deep-learning-pipeline-optimizer
pip install -r requirements.txt
```

## Usage

Optimize CNN

-  python main.py --model cnn --trials 15

Optimize RNN

-  python main.py --model rnn --trials 15


##  Example Output

Running Optuna optimization for model: cnn
Number of trials: 15

Best trial (cnn):
    Accuracy: 0.8894
    embedding_dim: 128
    dropout_rate: 0.3
    learning_rate: 0.0005
    batch_size: 64
    num_filters: 128
    kernel_size: 5


## Key Features

Architecture-agnostic optimization — easily plug in more models (e.g., Transformer).

Reusable training pipeline.

Search spaces per architecture (CNN vs RNN have different tunable params).

Early stopping to avoid overfitting during trials.


**feel free to use and modify.**
