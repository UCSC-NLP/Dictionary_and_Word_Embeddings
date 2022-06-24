import pandas as pd
import helper
import config
import numpy as np
import os
from zipfile import ZipFile
import urllib.request
import io



# glove.6B.zip
def download_fasttext_model(download_url, filename="crawl-300d-2M.vec.zip"):
    """ Download and extract data """
    
    downloaded_filename = os.path.join('.', filename)
    if not os.path.exists(downloaded_filename):
        print ('Downloading fasttext model...')
        urllib.request.urlretrieve(download_url,downloaded_filename)
    else:
        print('FastText model already downloaded...')
    if not os.path.exists('crawl-300d-2M.vec'):
        print ('Extracting fasttext data')
        zipfile=ZipFile(downloaded_filename)
        zipfile.extractall('./')
        zipfile.close()
    else:
        print('FastText data already extracted...')

'''
Code to load FastText vectors in a dictionary obtained from FastText Site with some modification
'''
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        for i in range(1,len(tokens)):
            tokens[i] = float(tokens[i])
        data[tokens[0]] = tokens[1:]
    return data

def write_set(output_filename, wordset):
    with open(output_filename, "w") as f:
        word_list = list(wordset)
        for word in word_list:
            f.write(word+"\n")

def extract_corresponding_fasttext_embeddings(pkl_data, data_dict, output_filepath,infile,outfile):
    print("------------------- Extracting FastText Embeddings -------------------")
    zero_vec_300D = np.zeros(300,dtype = 'float32')
    # word_fasttext_dict = dict()
    words_in_fasttext = 0
    words_not_in_fasttext = 0
    in_fasttext_set = set()
    not_in_fasttext_set = set()
    total = len(pkl_data)


    for dct in pkl_data:
        word = dct['word']
        # making all of our words in lower case to ensure they are found in FastText
        # word = word.lower()
        if word in data_dict:
            # word_fasttext_dict[word] = data_dict[word]
            dct['fasttext'] = np.array(data_dict[word])
            words_in_fasttext += 1
            in_fasttext_set.add(word)
        else:
            # word_fasttext_dict[word] = zero_vec_300D
            dct['fasttext'] = zero_vec_300D
            words_not_in_fasttext += 1
            not_in_fasttext_set.add(word)
    
    print(words_in_fasttext, "corresponding words found in FastText out of", total, "words in our set:",     words_in_fasttext*100/total, "percent and found", len(in_fasttext_set), "unique words")
    print(words_not_in_fasttext, "corresponding words found in FastText out of", total, "words in our set:", words_not_in_fasttext*100/total, "percent and found", len(not_in_fasttext_set), "unique words")
    print("\n\n Keys found in PKL data dictionary:",pkl_data[0].keys())
    
    write_set(outfile,not_in_fasttext_set)
    write_set(infile,in_fasttext_set)

    helper.write_pickle_file(pkl_data,output_filepath)



if __name__ == "__main__":
    train_pkl = helper.read_pickle_file("./../data/train_glove.pkl")
    dev_pkl = helper.read_pickle_file("./../data/dev_glove.pkl")
    test_pkl = helper.read_pickle_file("./../data/test_glove.pkl")

    # train_pkl = helper.read_pickle_file(config.proc_paths['train'])
    # dev_pkl = helper.read_pickle_file(config.proc_paths['dev'])
    # test_pkl = helper.read_pickle_file(config.proc_paths['test'])
    
    filename = "crawl-300d-2M.vec.zip"
    fasttext_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip'

    vector_file = "crawl-300d-2M.vec"
    
    download_fasttext_model(download_url=fasttext_url, filename=filename)
    data_dict = load_vectors(vector_file)

    train_outfile = "./../data/static_train.pkl"
    dev_outfile = "./../data/static_dev.pkl"
    test_outfile = "./../data/static_test.pkl"

    extract_corresponding_fasttext_embeddings(pkl_data = train_pkl,data_dict=data_dict,output_filepath = train_outfile,infile="in_fasttext_train.txt", outfile="not_in_fasttext_train.txt")
    extract_corresponding_fasttext_embeddings(pkl_data = dev_pkl,data_dict=data_dict,output_filepath = dev_outfile, infile="in_fasttext_dev.txt", outfile="not_in_fasttext_dev.txt")
    extract_corresponding_fasttext_embeddings(pkl_data = test_pkl,data_dict=data_dict,output_filepath = test_outfile, infile="in_fasttext_test.txt",outfile="not_in_fasttext_test.txt")