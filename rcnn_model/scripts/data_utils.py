#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
Design: OJ

data_utils.py

Summary:
    Utilities for loading, scaling, patching, and preparing training/testing datasets.
    Contains training and testing data preparation functions and generators.

Inputs:
    - Pickle files containing interpolated ERA5 and ACCESS-S2 data
    - Configuration parameters from config.py

Outputs:
    - Prepared train/test sets
    - Mask arrays for ocean-only learning
    - tf.data generators for training

Functions:
    - load_raw_data(year)
    - prepare_train_data()
    - prepare_test_data()
    - generate_patches()

Used In:
    - train.py, test_full_inference.py
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# data_utils.py
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from trainingtestingdatagenerator_cnn_era5 import trainingdata, testingdata
from config import Config

def load_raw_data(year):
    data_file = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed', f'Data{year}_gcm.p')))
    pds_file  = os.path.join(Config.DATA_PATH, f'pds_local_sstnsalt_{year}.p')
    (
        SST, Salt, hfss, rsds, rss, hfls, mld1
    ) = pickle.load(open(data_file, 'rb'))
    pds_local, pds_local_salt, filenames = pickle.load(open(pds_file, 'rb'))
    days = len(pds_local)
    return SST, Salt, hfss, rsds, rss, hfls, mld1, pds_local, pds_local_salt, filenames, days

def prepare_train_data(years, dayS, dayE):
    """
    Loads and prepares training data across multiple years.

    Parameters:
        years (list): List of years (e.g., [2019, 2020, 2021])

    Returns:
        (X_train, m_train, y_train), (X_val, m_val, y_val), (scaler_X, scaler_y)
    """
    all_X, all_y, all_mask = [], [], []

    for year in years:
        print(f"Loading training data for year {year}...")
        SST, Salt, hfss, rsds, rss, hfls, mld1, pds_local, pds_local_salt, _, days = load_raw_data(year)

        X, y = trainingdata(
            SST[-days:], Salt[-days:], hfss[-days:], rsds[-days:], rss[-days:], hfls[-days:], mld1[-days:],
            pds_local, pds_local_salt, days, dayS, dayE
        )

        # compute mask here
        mask = np.where(X.sum(axis=-1, keepdims=True) != 0, 1.0, 0.0)

        all_X.append(X)
        all_y.append(y)
        all_mask.append(mask)

    # Stack all years together
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    mask_all = np.concatenate(all_mask, axis=0)

    # Scaling
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_all.reshape(-1, X_all.shape[-1])).reshape(X_all.shape)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_all.reshape(-1, y_all.shape[-1])).reshape(y_all.shape)

    # Train-validation split
    X_tr, X_val, m_tr, m_val, y_tr, y_val = train_test_split(
        X_scaled, mask_all, y_scaled,
        test_size=Config.VALIDATION_SPLIT,
        random_state=Config.RANDOM_SEED
    )

    return (X_tr, m_tr, y_tr), (X_val, m_val, y_val), (scaler_X, scaler_y)


def generate_patches(X, mask, y):
    ps = Config.PATCH_SIZE
    n = X.shape[0]
    while True:
        i = np.random.randint(n)
        img, msk, lbl = X[i], mask[i], y[i]
        r = np.random.randint(0, img.shape[0]-ps+1)
        c = np.random.randint(0, img.shape[1]-ps+1)
        yield (img[r:r+ps, c:c+ps, :], msk[r:r+ps, c:c+ps, :]), lbl[r:r+ps, c:c+ps, :]

def prepare_test_data(year, dayS, dayE):
    SST, Salt, hfss, rsds, rss, hfls, mld1, pds_local, pds_local_salt, filenames, days = load_raw_data(year)
    X_test, y_test = testingdata(
        SST[-days:], Salt[-days:], hfss[-days:], rsds[-days:], rss[-days:], hfls[-days:], mld1[-days:],
        pds_local, pds_local_salt, days, dayS, dayE
    )
    return X_test, y_test, filenames
