"""
config.py

Summary:
    Contains global constants and configuration settings for the SST U-Net project.
    Also ensures required directories (data, output) exist at runtime.

Key Settings:
    - DATA_PATH, OUTPUT_DIR: where input/output data is stored
    - PATCH_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE: model hyperparameters
    - MIXED_PRECISION: whether to use float16 training
    - RANDOM_SEED: ensures reproducibility

Functions:
    - ensure_directories(): creates paths if they don’t exist

Usage:
    from config import Config
    Config.ensure_directories()
"""

# config.py
import os

class Config:
    ROOT_DIR = os.getenv("SST_MODEL_ROOT", os.getcwd())  # fallback to current directory
    DATA_PATH = os.path.join(ROOT_DIR, 'data')
    MODEL_DIR = os.path.join(ROOT_DIR, 'models')
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

    PATCH_SIZE = 128
    BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    VALIDATION_SPLIT = 0.1
    MIXED_PRECISION = True
    RANDOM_SEED = 42

    @staticmethod
    def ensure_directories():
        os.makedirs(Config.DATA_PATH, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
