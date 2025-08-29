""" Module contains necessary function to train models."""

# import necessary modules
from tensorflow.keras.callbacks import EarlyStopping

# define train function
def train_eval(model, X_train, y_train,X_val, y_val,
               epochs=10, batch_size=64, use_early_stop=True):
    """
    Train the model and return evaluation metrics.

    Args:
        model: model architecture (CNN or RNN).
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels (optional).
        batch_size (int): Number of samples per training batch.
        epochs (int): Max number of training epochs.
        use_early_stopping (bool): Whether to apply early stopping.

    Returns:
        history: Training history object.
        final_val_acc: Final validation accuracy (if val data provided).
        final_val_loss: Final validation loss (if val data provided).
    """

    # compile model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # create callback if necessary
    callbacks = []
    if use_early_stop:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True))

    # fit model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # evaluate model
    final_val_loss, final_val_acc = model.evaluate(X_val, y_val, verbose=0)

    return history, final_val_acc, final_val_loss
