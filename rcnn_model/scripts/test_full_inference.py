#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
Design: OJ

test_full_inference.py

Summary:
    Performs full-image inference using a trained U-Net model.
    Applies trained scalers to test data, runs predictions, evaluates metrics
    (MSE, RMSE, MAE), and saves results and metrics to disk.

Inputs:
    - Test data loaded from pickle
    - Trained model weights from config.OUTPUT_DIR
    - Scalers from training

Outputs:
    - Predicted SST maps
    - Evaluation metrics (MSE, RMSE, MAE)
    - All results saved to `test_full_results.pkl` in config.OUTPUT_DIR

Usage:
    python test_full_inference.py
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from config import Config
Config.ensure_directories()

# test_full_inference.py

import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config     import Config
from data_utils import prepare_test_data
from model      import build_unet

def main():
    # 1. Load test data (2011)
    X_test, y_test, filenames = prepare_test_data()
    n, H, W, C = X_test.shape

    # 2. Load scalers from training
    with open(os.path.join(Config.OUTPUT_DIR, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    scaler_X, scaler_y = scalers['scaler_X'], scalers['scaler_y']

    # 3. Apply scaling (no .fit, only .transform)
    X_flat       = X_test.reshape(-1, C)
    X_scaled     = scaler_X.transform(X_flat).reshape(n, H, W, C)
    y_flat       = y_test.reshape(-1, 1)
    y_scaled     = scaler_y.transform(y_flat).reshape(n, H, W, 1)

    # 4. Build full-image UNet and load weights
    #    (must match the down/upsampling architecture
    #     so that 640 and 480 are divisible by 2^3=8)
    model = build_unet((H, W, C))
    model.load_weights(os.path.join(Config.OUTPUT_DIR, 'unet_weights.h5'))
    model.compile(
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse'), 'mae']
    )

    # 5. Create the water-mask and predict
    mask = (X_scaled.sum(axis=-1, keepdims=True) != 0).astype(np.float32)
    y_pred_scaled = model.predict([X_scaled, mask], batch_size=Config.BATCH_SIZE)

    # 6. Invert y-scaling back to original units
    y_pred_flat = y_pred_scaled.reshape(-1, 1)
    y_pred_orig = scaler_y.inverse_transform(y_pred_flat).reshape(n, H, W, 1)

    # 7. Compute metrics over water pixels only
    mask_flat   = mask.reshape(-1).astype(bool)
    y_true_flat = y_test.reshape(-1)
    y_pred_flat = y_pred_orig.reshape(-1)

    mse  = mean_squared_error(y_true_flat[mask_flat], y_pred_flat[mask_flat])
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true_flat[mask_flat], y_pred_flat[mask_flat])

    print(f"Full-image Test → MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # 8. Save predictions, true fields, metrics and filenames
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    out = {
        'filenames': filenames,
        'y_true':    y_test,
        'y_pred':    y_pred_orig,
        'mse':       mse,
        'rmse':      rmse,
        'mae':       mae
    }
    with open(os.path.join(Config.OUTPUT_DIR, 'test_full_results.pkl'), 'wb') as f:
        pickle.dump(out, f)

    print("✅ Full-image inference complete. Results in", Config.OUTPUT_DIR)

if __name__ == '__main__':
    main()
