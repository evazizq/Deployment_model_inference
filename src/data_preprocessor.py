import pandas as pd
import numpy as np

def preprocessor_data(data:dict, columns_to_impute:list)->pd.DataFrame:


    try:
        df = pd.DataFrame([data])
        df[columns_to_impute] = df[columns_to_impute].replace(0, np.nan)
        return df

    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise