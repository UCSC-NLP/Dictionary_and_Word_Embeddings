import faiss

class Similarity:
    def __init__(self, index_name, embeddings):
        
        self.len_of_src_embs = len(embeddings)
        self.index = self.create_index(index_name, embeddings)
        
    def create_index(self, index_name, embeddings):
        '''
        desc: 
            This function creates an exaustive search index
        params: 
            index_name:values: ['l2', 'dot', 'cosine']
            embeddings: embeddings to be indexed. shape -> (num_examples, dim)
                
        return:
            index: faiss index       
        '''
        assert index_name in ['l2', 'dot', 'cosine'], "Choose one of allowed index types: ['l2', 'dot', 'cosine']"
        
        #assumes batch first for embeddings
        dimension = len(embeddings[0])
        
        if index_name == 'l2':
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
        
        if index_name == 'dot':
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            
        if index_name == 'cosine':  
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
        return index
        
        
    def get_knn(self, k, query_embs):
        '''
        desc: 
            Get k nearest neighbours given the query embeddings
        params: 
            k: number of nearest neighbours to find
            query_embs: an array of query embeddings. shape -> (num_examples, dim)
                
        return:
            D: nearest distances
            I: neares indexes (of orignal embeddings on which index was built)
        '''
        D, I = self.index.search(query_embs, k)
        return D, I
    
    
    def get_index_metrics(self, embeddings):
        '''
        desc:
            get mean, min, max of all distances computed with original embeddings with given embeddings
        params:
            embeddings: query embeddings. shape-> (num_examples, dim)
        return:
            None
        '''
        D, I = self.index.search(embeddings, self.len_of_src_embs)
        print('Mean of Distance: ', D.mean())
        print('Min of Distance: ', D.min())
        print('Mxa of Distance: ', D.max())
        del(D,I)
