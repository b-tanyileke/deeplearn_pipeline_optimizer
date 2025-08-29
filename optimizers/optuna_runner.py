"""
Hyperparameter optimization runner using Optuna.

This script defines an Optuna objective function that builds and tunes either a
CNN or RNN model on the IMDB dataset. It supports tuning various hyperparameters 
like embedding dimensions, dropout rates, learning rates, and architecture-specific
parameters such as kernel size or GRU units.

"""

# import necessary modules
import optuna
from data.imdb_loader import load_imdb_data
from models.cnn_text import build_cnn_model
from models.rnn_text import build_rnn_model
from train.train_model import train_eval
from tensorflow.keras.optimizers import Adam

# define objective function
def objective(trial, model_type="cnn"):
    """
    Objective function used as foundation for optimization

    Args:
        trial (int): 
        model_type (str): Type of model, either cnn or rnn

    Return:
        Best validation accuracy
    """

    # define constant parameters
    vocab_size = 10000
    maxlen = 200
    epochs = 5

    # define shared hyperparameters
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Load dataset
    (X_train, y_train), (X_test, y_test) = load_imdb_data(vocab_size=vocab_size, maxlen=maxlen)

    # Build model depending on type
    if model_type == "cnn":
        num_filters = trial.suggest_categorical("num_filters", [64, 128, 256])
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        model = build_cnn_model(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            maxlen=maxlen,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate
        )

    elif model_type == "rnn":
        gru_units = trial.suggest_categorical("gru_units", [64, 128, 256])
        model = build_rnn_model(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            maxlen=maxlen,
            gru_units=gru_units,
            dropout_rate=dropout_rate
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    

    # Compile model with suggested learning rate
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train and evaluate
    _, val_acc, _ = train_eval(
        model, X_train, y_train,
        X_val=X_test, y_val=y_test,
        batch_size=batch_size,
        epochs=epochs,
        use_early_stop=True
    )

    return val_acc


# define function to run search
def run_optuna_search(model_type="cnn", n_trials=20):
    """
    Run an Optuna hyperparameter optimization study.

    Args:
        model_type (str): 'cnn' or 'rnn'
        n_trials (int): Number of Optuna trials

    Returns:
        optuna.Study object with results
    """
    study = optuna.create_study(direction="maximize", study_name="text_classifier_optimization")
    study.optimize(lambda trial: objective(trial, model_type=model_type), n_trials=n_trials)

    print(f"\nBest trial ({model_type}):")
    print(f"Accuracy: {study.best_value:.4f}")
    print("Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    return study
