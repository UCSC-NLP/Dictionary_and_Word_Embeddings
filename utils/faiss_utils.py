import os
import sys
import pandas as pd
import numpy as np
# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=False)
import joblib
sys.path.append('..')

import random
import collections
from evaluation.similarity import Similarity
import faiss
import torch
from tqdm import tqdm
import _pickle as cPickle

def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    


def load_data():
    """
    desc: 
        Load dataset for creating the index
        
    returns:
        data
    """
    
    seed_everything(42)
    
    # train = joblib.load('../data/clean_data/train.joblib')
    val = joblib.load('../data/clean_data/val.joblib')
    test = joblib.load('../data/clean_data/test.joblib')
    
    # data = pd.concat([train, val, test], axis =0).drop_duplicates(['gloss','word']).reset_index(drop = True)
    data = pd.concat([val, test], axis =0).drop_duplicates(['gloss','word']).reset_index(drop = True)
    return data


def create_and_store_index(emb_type, knn_measure):
    """
    desc:
        Create the FAISS index using one of emb_type -> [l2, cosine, dot]
        and one of knn_measure -> [l2, cosine]
        
        We also create a dictionary that captures the index and word itself
        to map the word back to indexes during predictions as FAISS returns index
        of words.
        
    args:
        emb_type: str: embedding type to create the FAISS index
        knn_measure: str: measure to find nearest neighbours
        unique_mapping: bool: same words multiple times considered as single index. 
                            True for static, False for contextual embeddings
        
    """
    print(f'Loading Data for Indexing...')
    data = load_data()
    
    print(f'Creating FAISS Index...')

    #data['glove'] = data['glove'].apply(lambda x: x.reshape(1,-1))
    data[emb_type] = data[emb_type].apply(lambda x: x.reshape(1,-1))

    data['string_emb'] = data[emb_type].apply(np.sum,axis=1).apply(str)
    
    print (f'Data len before unique_mapping: {len(data)}')
    data = data.drop_duplicates(subset = ['string_emb','word']).reset_index(drop=True)
    print (f'Data len after unique_mapping: {len(data)}')
    
    #store words in dict with index as id and corresponding embeddings in lists
    faiss_idx_index_to_word_lookup = collections.defaultdict()
    embs_list = []
    
    for i in range(len(data)):
        
        faiss_idx_index_to_word_lookup[i] = data.loc[i]['word']
        embs_list.append(data.loc[i][emb_type][0].astype(np.float32))  #faiss needs data to be in float32
        
    embs = np.stack(embs_list) 
    
    print ('Index shape',embs.shape)
    
    sim = Similarity()
    faiss_index = sim.create_index(knn_measure, embs)
    
    faiss.write_index(faiss_index, f'../data/clean_data/faiss_index_{emb_type}_{knn_measure}')
    joblib.dump(faiss_idx_index_to_word_lookup, '../data/clean_data/faiss_idx_index_to_word_lookup.joblib')
    
    print(f'FAISS Index Creation Done...')
    


    
    
    
def get_top_n_accuracy(rank_list, n, ):
    
    
    accuracy = [1 if i <= n else 0 for i in rank_list]
    return np.round(np.mean(accuracy), 3)
    
    
    
