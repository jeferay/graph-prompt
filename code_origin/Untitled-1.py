
from math import pi
from os import path
from numpy.lib.function_base import average
from  sklearn.feature_extraction.text import TfidfVectorizer
from dataset import simple_load_data
import torch

import numpy as np
def cos_similar(p: torch.Tensor, q: torch.Tensor):
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    return sim_matrix
        
def obo_sim():
    name_array, query_id_array, mention2id, edge_index, triples = simple_load_data('../data/datasets_update/hp.obo',use_text_preprocesser=False,return_triples=True)
    query_array = [query for (query,id) in query_id_array]
    sparse_encoder = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))# only works on cpu
    sparse_encoder.fit(name_array)

    names_sps = torch.FloatTensor(sparse_encoder.transform(name_array).toarray())

    query_sps = torch.FloatTensor(sparse_encoder.transform(query_array).toarray())

    sim_mat = cos_similar(query_sps,names_sps)

    sim_avg = average([sim_mat[i][mention2id[query]] for (i,query) in enumerate(query_array)])
    print(sim_avg)

def other_sim():
    