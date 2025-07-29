#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# General Info:
"""
Design: OJ

train.py

Summary:
    Trains a U-Net model for downscaling SST data using interpolated inputs.
    Uses patch-based training with optional mixed precision, model checkpoints,
    and early stopping. Saves the model weights and scalers to output directory.

Inputs:
    - Training data from processed pickle files (defined in config.DATA_PATH)
    - Model and training hyperparameters from config.py

Outputs:
    - Trained U-Net model weights (.h5)
    - Scalers used to normalize input/output
    - Training logs and metrics saved to config.OUTPUT_DIR

Usage:
    python train.py
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from config import Config
Config.ensure_directories()

# train.py
import os
import tensorflow as tf
import pickle

from config    import Config
from data_utils import prepare_train_data, generate_patches
from model     import build_unet

# Mixed precision
if Config.MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy('float32')

years = [2015, 2016, 2017, 2018, 2019, 2020]
dayS = 0
dayE = 1
# Prepare data
(X_tr, m_tr, y_tr), (X_val, m_val, y_val), (scaler_X, scaler_y) = prepare_train_data(years, dayS, dayE)

# Create tf.data pipelines
def make_ds(X, m, y, shuffle=False):
    ds = tf.data.Dataset.from_generator(
        lambda: generate_patches(X, m, y),
        output_signature=(
            (tf.TensorSpec((Config.PATCH_SIZE, Config.PATCH_SIZE, X.shape[-1]), tf.float32),
             tf.TensorSpec((Config.PATCH_SIZE, Config.PATCH_SIZE, 1),           tf.float32)),
            tf.TensorSpec((Config.PATCH_SIZE, Config.PATCH_SIZE, 1),           tf.float32)
        )
    )
    if shuffle:
        ds = ds.shuffle(1024)
    return ds.batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(X_tr, m_tr, y_tr, shuffle=True)
val_ds   = make_ds(X_val, m_val, y_val)

# Build & compile model
model = build_unet((Config.PATCH_SIZE, Config.PATCH_SIZE, X_tr.shape[-1]))
model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse'), 'mae']
)

# Callbacks
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
cbs = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(Config.OUTPUT_DIR, 'best_unet.h5'),
        save_best_only=True, monitor='val_loss'
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True, monitor='val_loss'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        patience=3, factor=0.5, monitor='val_loss'
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(Config.OUTPUT_DIR, 'logs')
    )
]

# Train
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=Config.EPOCHS,
    callbacks=cbs
)

# Save scalers
with open(os.path.join(Config.MODEL_DIR, 'scalers.pkl'), 'wb') as f:
    pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)

# Save model structure + weights
model_json = model.to_json()
with open(os.path.join(Config.MODEL_DIR, 'unet_model.json'), 'w') as f:
    f.write(model_json)
model.save_weights(os.path.join(Config.MODEL_DIR, 'unet_weights.h5'))

print("✅ Training complete. Artifacts saved in", Config.MODEL_DIR)
