import numpy as np
import pandas as pd

from dpipe.io import load

def get_target_domain_metrics(dataset_path,path_base,s):
    meta = pd.read_csv(f"{dataset_path}/meta.csv",delimiter=",", index_col='id')
    res_row = {}
    
    # one2all results:
    sdices = load(path_base / f'mode_{s}/sdice_score.json')
    #sdices = dict(sorted(sdices.items()))
    for t in sorted(set(meta['fold'].unique()) - {s}):
        df_row = meta[meta['fold'] == t].iloc[0]
        target_name = df_row['tomograph_model'] + str(df_row['tesla_value'])
        
        ids_t = meta[meta['fold'] == t].index
        res_row[target_name] = np.mean([sdsc for _id, sdsc in sdices.items() if _id in ids_t])
    return res_row