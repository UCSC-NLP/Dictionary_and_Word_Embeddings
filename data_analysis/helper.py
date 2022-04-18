import numpy as np
import pandas as pd
# from tqdm import tqdm
# import faiss
# import json
import pickle
# import config

import random
random.seed(1234)

lang = 'en'

def build_sample_data(k=10_000):
    '''
    function to sample k data points from complete dataset and write in a pickle file
    '''
    with open(config.full_data_path, 'rb') as outfile:
        db = pickle.load(outfile)
    samples_indices = random.sample(range(len(db)),k)
    sampled_db = [db[i] for i in samples_indices]
    # return sampled_db
    
    # data_path = f'/data/users/kartik/thesis/processed_data/{lang}.10k_sampled.wordgloss_group.pkl'
    print (f'Writing {len(sampled_db)} samples to {config.sampled_data_path}')
    with open(config.sampled_data_path, 'wb') as outfile:
        pickle.dump(sampled_db, outfile)
    
    return


def read_pickle_file(filepath):
    '''
    function to read any pickle file
    '''
    with open(filepath, 'rb') as outfile:
        return pickle.load(outfile)

def process_rawdf_to_pkl():
    '''
    function to convert raw unstructured dataframe to pickle file 
    '''
    print (f'Loading dataframe for {lang}')
    df = pd.read_csv(config.raw_df_path)
    df = df.reset_index()
    # print (df.head(2))

    emb_types = ['sgns','char','electra']
    for emb_type in emb_types:
        print (f"Converting {emb_type} embeddings from strings to numpy in dataframe...")
        df[emb_type] = df[emb_type].apply(lambda x: np.array(json.loads(x)))


    db = []
    cols = ['index']+emb_types
    for i,group in df.groupby(['word','gloss']):
        item = {'word':i[0],'gloss':i[1]}
        for key in cols:
            item[key]=group[key].tolist()
        db.append(item)
    print (f"Number of unique [word+gloss] : {len(db)}")
    
    with open(config.full_data_path, 'wb') as outfile:
        pickle.dump(db, outfile)
    
if __name__=='__main__':
    pass
    # build_sample_data(k=10_000)
