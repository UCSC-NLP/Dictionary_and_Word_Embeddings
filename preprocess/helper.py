import sys
# sys.path.append('.')
sys.path.append('..')

import numpy as np
import pandas as pd
# from tqdm import tqdm
# import faiss
import json
import pickle
import preprocess.config as config

import random
random.seed(1234)

lang = 'en'

def build_sample_data(k=10_000):
    '''
    function to sample k data points from complete dataset and write in a pickle file
    '''
    with open(config.proc_paths['full_data'], 'rb') as outfile:
        db = pickle.load(outfile)
    samples_indices = random.sample(range(len(db)),k)
    sampled_db = [db[i] for i in samples_indices]
    # return sampled_db
    
    # data_path = f'/data/users/kartik/Dictionary_and_Word_Embeddings/processed_data/{lang}.10k_sampled.wordgloss_group.pkl'
    print (f'Writing {len(sampled_db)} samples to {config.sampled_data_path}')
    write_pickle_file(sampled_db,config.sampled_data_path)
    
    return


def read_pickle_file(filepath):
    '''
    function to read any pickle file
    '''
    with open(filepath, 'rb') as outfile:
        return pickle.load(outfile)

def write_pickle_file(obj,filepath):
    '''
    function to write to any pickle file
    '''
    with open(filepath, 'wb') as outfile:
        return pickle.dump(obj,outfile)

def process_rawdf_to_pkl():
    '''
    function to convert raw unstructured dataframe to pickle file 
    '''
    print (f'Loading dataframe for {lang}')
    df = pd.read_csv(config.raw_paths['full_data'])
    df = df.reset_index()
    # print (df.head(2))

    emb_types = ['sgns','char','electra']
    for emb_type in emb_types:
        print (f"Converting {emb_type} embeddings from strings to numpy in dataframe...")
        df[emb_type] = df[emb_type].apply(lambda x: np.array(json.loads(x)))


    db = []
    cols = ['index','example']+emb_types
    for i,group in df.groupby(['word','gloss']):
        item = {'word':i[0],'gloss':i[1]}
        for key in cols:
            item[key]=group[key].tolist()
        db.append(item)
    print (f"Number of unique [word+gloss] : {len(db)}")
    print (db[0])


    with open(config.proc_paths['full_data'], 'wb') as outfile:
        pickle.dump(db, outfile)
    
def prepare_splits(mode='dev'):
    '''
    function to prepare the train validation and test split from full dataset
    '''

    full_data = read_pickle_file(config.proc_paths['full_data'])
    N=len(full_data)

    df = pd.read_json(config.raw_paths[mode])
    df = df.reset_index()
    print ('Complete data length: ',len(df))
    glosses = set(df['gloss'].values.tolist())
    print ('Unique data length (by gloss): ',len(glosses))
    
    db = []
    for idx,item in enumerate(full_data):
        if item['gloss'] in glosses:
            db.append(item)
    print (f'Number of samples in {mode}: ',len(db))
    print (db[0])

    with open(config.proc_paths[mode], 'wb') as outfile:
        pickle.dump(db, outfile)
    


if __name__=='__main__':
    # process_rawdf_to_pkl()
    # build_sample_data(k=10_000)

    # prepare_splits('train')
    # prepare_splits('dev')
    # prepare_splits('test')
    pass
