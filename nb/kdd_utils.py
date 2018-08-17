import pandas as pd
import numpy as np
import os 

path = '../input/'

def addFieldDataFtrs(df, shiftFtrs=['temperature', 'dewpoint', 'windspeed', 'Precipitation', 'Soilwater_L1'],
                    shiftPeriod=2):
    # Read field data.
    df_field = pd.read_csv(os.path.join(path, 'field-0.csv'))
    df_field['field'] = 0
    for i in range(1, 28):
        _df_field = pd.read_csv(path+'field-{}.csv'.format(i))
        _df_field['field'] = i
        df_field = pd.concat([df_field, _df_field])

    # Remove duplicates.
    df_field = df_field.drop_duplicates()

    # Group by month, year and field.
    df_field = df_field.groupby(['month', 'year', 'field']).mean().reset_index()

    # Merge with given data.
    df = pd.merge(df, df_field, left_on=['harvest_year', 'harvest_month','field'], 
                  right_on=['year', 'month', 'field'], how='inner').reset_index()

    # Remove redundant features.
    # df_all = df_all.drop(columns=['Soilwater_L2', 'Soilwater_L3', 'Soilwater_L4'])

    if shiftFtrs is not None:
        df_group = df.groupby(['field', 'harvest_year', 'harvest_month']).mean().reset_index()
        df_group = df_group[['field', 'harvest_year', 'harvest_month', 'production'] + shiftFtrs]
        df_group = df_group.sort_values(['field', 'harvest_year', 'harvest_month'])

        # Collect shift values of variables in all feature time.
        new_features = {}
        for f in shiftFtrs:
            new_features[f] = []
            for i in range(1, shiftPeriod):
                new_feature = '{}_{}'.format(f, i)
                new_features[f].append(new_feature)
                df_group[new_feature] = df_group[f].shift(i).fillna(df_group[f].mean())

        df_group = df_group.drop(shiftFtrs + ['production'], axis=1)

        df = df.drop(['index', 'month', 'year'], axis=1)
        df = pd.merge(df, df_group, left_on=['field', 'harvest_year', 'harvest_month'], 
                      right_on=['field', 'harvest_year', 'harvest_month'], how='inner')
        df = df.reset_index()

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