import pandas as pd
import helper
import config
import numpy as np
import os
from zipfile import ZipFile
import urllib.request


# glove.6B.zip
def download_glove_model(download_url, filename='glove.6B.zip'):
    """ Download and extract data """
    
    downloaded_filename = os.path.join('.', filename)
    if not os.path.exists(downloaded_filename):
        print ('Downloading glove model...')
        urllib.request.urlretrieve(download_url,downloaded_filename)
    else:
        print('Glove model already downloaded...')
    if not os.path.exists('glove.42B.300d.txt'):
        print ('Extracting glove data')
        zipfile=ZipFile(downloaded_filename)
        zipfile.extractall('./')
        zipfile.close()

def write_set(output_filename, wordset):
    with open(output_filename, "w") as f:
        word_list = list(wordset)
        for word in word_list:
            f.write(word+"\n")

def make_glove_dictionary():
    embed_dict = {}
    print("------------------- Loading glove Dictionary -------------------")
    with open('./glove.42B.300d.txt','r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:],'float32')
            embed_dict[word]=vector
    return embed_dict

def extract_corresponding_glove_embeddings(pkl_data, embed_dict, output_filepath,infile,outfile):
    print("------------------- Extracting Glove Embeddings -------------------")
    zero_vec_300D = np.zeros(300,dtype = 'float32')
    # word_glove_dict = dict()
    total = len(pkl_data)
    words_in_glove = 0
    words_not_in_glove = 0
    in_glove_set = set()
    not_in_glove_set = set()

    for dct in pkl_data:
        word = dct['word']
        # making all of our words in lower case to ensure they are found in glove
        word = word.lower()
        if word in embed_dict:
            # word_glove_dict[word] = embed_dict[word]
            dct['glove'] = np.array(embed_dict[word])
            words_in_glove += 1
            in_glove_set.add(word)
        else:
            # word_glove_dict[word] = zero_vec_300D
            dct['glove'] = zero_vec_300D
            words_not_in_glove += 1
            not_in_glove_set.add(word)

    print(words_in_glove, "corresponding words found in GLoVe out of", total, "words in our set:", words_in_glove*100/total, "percent and found", len(in_glove_set), "unique words")
    print(words_not_in_glove, "corresponding words found in GLoVe out of", total, "words in our set:", words_not_in_glove*100/total, "percent and found", len(not_in_glove_set), "unique words")

    write_set(outfile,not_in_glove_set)
    write_set(infile,in_glove_set)
    
    helper.write_pickle_file(pkl_data,output_filepath)

if __name__ == "__main__":
    train_pkl = helper.read_pickle_file(config.proc_paths['train'])
    dev_pkl = helper.read_pickle_file(config.proc_paths['dev'])
    test_pkl = helper.read_pickle_file(config.proc_paths['test'])
    filename = "glove.6B.zip"
    glove_url = 'https://nlp.stanford.edu/data/glove.6B.zip'
    glove_42B_url = 'https://nlp.stanford.edu/data/glove.42B.300d.zip'
    
    download_glove_model(download_url=glove_42B_url, filename=filename)

    embed_dict = make_glove_dictionary()

    train_outfile = "./../data/train_glove.pkl"
    dev_outfile = "./../data/dev_glove.pkl"
    test_outfile = "./../data/test_glove.pkl"

    extract_corresponding_glove_embeddings(pkl_data = train_pkl,embed_dict=embed_dict,output_filepath = train_outfile,infile="in_glove_train.txt",outfile="not_in_glove_train.txt")
    extract_corresponding_glove_embeddings(pkl_data = dev_pkl,embed_dict=embed_dict,output_filepath = dev_outfile,infile="in_glove_dev.txt",outfile="not_in_glove_dev.txt")
    extract_corresponding_glove_embeddings(pkl_data = test_pkl,embed_dict=embed_dict,output_filepath = test_outfile,infile="in_glove_test.txt",outfile="not_in_glove_test.txt")

