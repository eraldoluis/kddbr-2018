import os

import pandas as pd

path = '../input/'

def addFieldDataFtrs(df, shiftFtrs=['temperature', 'dewpoint', 'windspeed', 'Precipitation', 'Soilwater_L1'],
                    shiftPeriod=2, rolling_wins=[3]):
    # Read field data.
    df_fields = []
    for field in range(28):
        _df_field = pd.read_csv(os.path.join(path, 'field-{}.csv'.format(field)))
        _df_field.insert(0, 'field', field)

        # Generate shift and rolling features.
        for ftr in shiftFtrs:
            for shift in range(1, shiftPeriod):
                _df_field['{}_shift{}'.format(ftr, shift)] = _df_field[ftr].shift(shift)

            for win in rolling_wins:
                _df_field['{}_rolling_mean{}'.format(ftr, win)] = _df_field[ftr].rolling(win, min_periods=1).mean()
                _df_field['{}_rolling_min{}'.format(ftr, win)] = _df_field[ftr].rolling(win, min_periods=1).min()
                _df_field['{}_rolling_max{}'.format(ftr, win)] = _df_field[ftr].rolling(win, min_periods=1).max()

        df_fields.append(_df_field)

    df_field = pd.concat(df_fields)

    # Merge with given data.
    df = pd.merge(df, df_field, left_on=['field', 'harvest_year', 'harvest_month'],
                  right_on=['field', 'year', 'month'], how='inner').reset_index()

    # Return merged data.
    return df

def addSoilFtrs(df):
    df_soil = pd.read_csv(os.path.join(path, 'soil_data.csv'))
    df = pd.merge(df, df_soil, on='field', how='inner')
    return df

def cvPerYear(X, y, fromYear, toYear, numTrainYears=0, evalOnlyLast=False):
    # Reset index is necessary to produce correct indexes for numpy arrays.
    X = X.reset_index()
    
    # Validate using sliding window per year.
    for val_year in range(fromYear, toYear):

        # Train split.
        if numTrainYears == 0:
            # Train split using all 'previous' data.
            idxs_train = X[X.harvest_year < val_year].index.values.astype(int)
        else:
            # Train split using 'previous' data up to a limited window.
            idxs_train = X[(X.harvest_year < val_year) & (X.harvest_year >= (val_year - numTrainYears))].index.values.astype(int)
        
        # Test split.
        if evalOnlyLast:
            # Test split composed of 'present' year only.
            idxs_test = X[X.harvest_year == val_year].index.values.astype(int)
        else:
            # Test split composed of 'present and future' data.
            idxs_test = X[X.harvest_year >= val_year].index.values.astype(int)
        
        yield (idxs_train, idxs_test)

def _test_startswith(name, lst_prefix):
    for p in lst_prefix:
        if name.startswith(p):
            return True
    return False

import torch

def save_model_ignoring(model, name, ignore=[]):
    d = model.model.state_dict()
    for k in list(d.keys()):
        if _test_startswith(k, ignore):
            del d[k]
    torch.save(d, model.get_model_path(name))

def load_model_ignoring(model, name, ignore=[]):
    sd = torch.load(model.get_model_path(name), map_location=lambda storage, loc: storage)
    names = set(model.model.state_dict().keys())
    for n in list(sd.keys()): # list "detatches" the iterator
        if _test_startswith(n, ignore):
            del sd[n]
        elif n not in names and n+'_raw' in names:
            if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
            del sd[n]
    model.model.load_state_dict(sd, strict=False)