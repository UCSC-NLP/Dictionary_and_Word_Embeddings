import enum
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# model_checkpoint = ['xlm-roberta-base', 'distilbert-base-uncased','bert-base-uncased','albert-base-v2'][2]


class FeatureExtractor:

    def __init__(self, model_checkpoint):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, output_hidden_states = True)
        self.model = AutoModel.from_pretrained(model_checkpoint,output_attentions=False,output_hidden_states=False)


    def extract_feature(self, sentence, word):
        '''
            tokenize sentence and extract contextual feature of a word from a sentence
        '''

        sentence = sentence.lower().split()

        word_idx_in_sentence = -1
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
        for idx,w in enumerate(sentence):
            
            if w.startswith(word):
                word_idx_in_sentence = idx
                break

        if word_idx_in_sentence == -1:
            return ''

        # print (sentence,word)
        sent_tokenized = self.tokenizer(sentence, return_offsets_mapping=True, is_split_into_words=True,return_tensors='pt')

        with torch.no_grad():
            sent_feature = self.model(sent_tokenized['input_ids'])['last_hidden_state']

        tok_level_embs = []

        for idx, id_ in enumerate(sent_tokenized.word_ids()):
            if id_ == word_idx_in_sentence:
                tok_level_embs.append(sent_feature[0,idx,:].numpy())

        if len(tok_level_embs)==0:
            return ''

        return np.stack(tok_level_embs).mean(axis=0)



    


