lang = 'en'
root = '/data/users/kartik/Dictionary_and_Word_Embeddings'


raw_paths = {}
raw_paths['full_data'] =f'{root}/data/full_dataset/{lang}.complete.csv'
raw_paths['train'] =f'{root}/data/train_data/{lang}.dev.json'
raw_paths['dev'] =f'{root}/data/train_data/{lang}.dev.json'
raw_paths['test'] =f'{root}/data/test_data/{lang}.test.revdict.json'


proc_paths = {}
proc_paths['full_data'] = f'{root}/processed_data/{lang}.complete.wordgloss_group.pkl'

proc_paths['train'] = f'{root}/processed_data/{lang}.train.wordgloss_group.pkl'
proc_paths['dev'] = f'{root}/processed_data/{lang}.dev.wordgloss_group.pkl'
proc_paths['test'] = f'{root}/processed_data/{lang}.test.wordgloss_group.pkl'

sampled_data_path = f'{root}/processed_data/{lang}.10k_sampled.wordgloss_group.pkl'

sampled_rougescore_path = f'{root}/processed_data/{lang}.10k_sampled.pairwise_rouge.pkl'
