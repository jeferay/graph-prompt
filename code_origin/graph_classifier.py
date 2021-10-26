import numpy as np
from numpy.core.numeric import indices
from scipy import sparse
import torch
from torch import nn
from torch import optim
from torch.autograd.grad_mode import enable_grad
from torch.nn import parameter
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models import bert

from dataset import Biosyn_Dataset,Mention_Dataset, load_data,data_split,simple_load_data,Graph_Dataset
from models import Biosyn_Model,Bert_Candidate_Generator,Bert_Cross_Encoder,Graphsage_Model
from criterion import marginal_loss
from transformers import *
from sklearn.feature_extraction.text import TfidfVectorizer
import os

from utils import TimeIt
   
class Graph_Classifier():
    def __init__(self,args):
        self.args = args
        self.filename = self.args['filename']
        self.use_text_preprocesser = self.args['use_text_preprocesser']
        self.name_array, query_id_array, self.mention2id, _, self.triples = simple_load_data(self.filename, self.use_text_preprocesser, return_triples=True)#load data
        self.queries_train, self.queries_valid, self.queries_test = data_split(query_id_array=query_id_array, is_unseen=self.args['is_unseen'], test_size=0.33)# data split
        self.tokenizer = BertTokenizer(vocab_file=self.args['vocab_file'])
        #包含bert部分和graph部分
        self.graph_model = Graphsage_Model(feature_size=768,hidden_size=256,output_size=768,model_path=self.agrs['pretrain_model_path'],with_graph=self.args['with_graph'])
    
    #根据现有的bert model得到names embedding
    @torch.no_grad()
    def get_names_bert_embedding(self):
        names_dataset = Mention_Dataset(self.name_array,self.tokenizer)
        names_bert_embedding = []
        data_loader = DataLoader(dataset=names_dataset,batch_size=1024)
        for i,(input_ids, attention_mask) in enumerate(data_loader):
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            cls_embedding = self.graph_model.bert_encoder(input_ids,attention_mask).last_hidden_state[:,0,:]# batch * hidden_size
            names_bert_embedding.append(cls_embedding)
            
        names_bert_embedding = torch.cat(names_bert_embedding, dim=0)# len(mentions) * hidden_size
        return names_bert_embedding# still on the device
    
    def get_edge_index(self,undirected = True):
        edge_index =[[],[]]
        edge_index[0] = [h for h,r,t in self.triples]
        edge_index[1] = [t for h,r,t in self.triples]
        if undirected:
            edge_index[0] +=[t for h,r,t in self.triples]
            edge_index[1] +=[h for h,r,t in self.triples]
        
        return torch.LongTensor(edge_index)
        
    def train(self):
        self.args['logger'].info('stage_1_training')
        train_dataset = Graph_Dataset(query_array=self.queries_train,mention2id=self.mention2id,tokenizer=self.tokenizer)
        train_loader = DataLoader(dataset=train_dataset,batch_size=self.args['batch_size'],shuffle=True, drop_last=True) ###########
        criterion = nn.CrossEntropyLoss(reduction='sum')# take it as an multi class task
        edge_index = self.get_edge_index(undirected=True).cuda()
        
        optimizer = torch.optim.Adam(params=self.graph_model.parameters(),
        lr=self.args['lr'], weight_decay=self.args['weight_decay']
        )
        t_total = self.args['epoch_num'] * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

        accu_temp = 0
        for epoch in range(1,self.args['epoch_num'] + 1):
            # #every epoch we recalculate the embeddings which have been updated
            # names_bert_embedding = self.get_names_bert_embedding()# tensor of shape(N,768)
            # names_sparse_embedding = self.get_names_sparse_embedding()# tensor of shape(N, sparse_feature_size)
            self.graph_model.train()
            loss_sum = 0
            
            pbar = tqdm(enumerate(train_loader),total=len(train_loader))
            for iteration,(batch_query_ids, batch_query_attention_mask,batch_labels,batch_query) in pbar:
                if iteration % 100 == 0:#每100step更新一次
                    with TimeIt('get emb'):
                        names_bert_embedding = self.get_names_bert_embedding()# tensor of shape(N,768)
            
                optimizer.zero_grad()
                with TimeIt('make batch'):
                    batch_query_ids = batch_query_ids.cuda()
                    batch_query_attention_mask = batch_query_attention_mask.cuda()
                    batch_labels =batch_labels.cuda().squeeze()
                    batch_query = np.array(batch_query)
                
                with TimeIt('get scores'):
                    outputs = self.graph_model.forward(query_ids=batch_query_ids,query_attention_mask=batch_query_attention_mask,names_embedding=names_bert_embedding,edge_index=edge_index)
                
                    loss = criterion(outputs,batch_labels)

                with TimeIt('optimizer'):
                    loss_sum+=loss.item()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    loss_sum/=len(self.queries_train)

                m, M, mean, std = outputs.min(), outputs.max(), outputs.mean(), outputs.std()
                pbar.set_postfix({"loss": float(loss), "[min, max, mean, std]": ['%.2e'%i for i in [m, M, mean, std]], "lr":['%.2e'%group["lr"] for group in optimizer.param_groups]})


            accu_1,accu_k = self.eval_stage_1(query_array=self.queries_valid,epoch=epoch)
            print('loss_sum:', float(loss_sum))
            if accu_1>accu_temp:
                accu_temp = accu_1
                checkpoint_dir = os.path.join(self.args['exp_path'],'checkpoint')
                self.save_model_stage_1(checkpoint_dir)

        
        

    @torch.no_grad()
    def eval_stage_1(self,query_array,epoch,load_model = False):
        
        self.graph_model.eval()
        names_bert_embedding = self.get_names_bert_embedding()# tensor of shape(N,768)
        
        eval_dataset = Graph_Dataset(query_array=query_array,mention2id=self.mention2id,tokenizer=self.tokenizer)
        eval_loader = DataLoader(dataset=eval_dataset,batch_size = 1024,shuffle=False)
        
        accu_1 = torch.FloatTensor([0]).cuda()
        accu_k = torch.FloatTensor([0]).cuda()
        with torch.no_grad():
            pbar = tqdm(enumerate(eval_loader),total=len(eval_loader))
            for iteration,(batch_query_ids, batch_query_attention_mask,batch_labels,batch_query) in pbar:
                with TimeIt('make batch'):
                    batch_query_ids = batch_query_ids.cuda()
                    batch_query_attention_mask = batch_query_attention_mask.cuda()
                    batch_labels =batch_labels.cuda()
                    batch_query = np.array(batch_query)
                
                with TimeIt('get scores'):
                    outputs = self.graph_model.forward(query_ids=batch_query_ids,query_attention_mask=batch_query_attention_mask,names_embedding=names_bert_embedding,edge_index=edge_index)
                
                values,indices = torch.topk(input=outputs,k=self.args['eval_k'],dim=-1)
                
                accu_1 += (indices[:,0]==batch_labels).sum()/len(query_array)
                accu_k += (indices== torch.unsqueeze(batch_labels,dim=1)).sum()/len(query_array)
        self.args['logger'].info("epoch %d done, accu_1 = %f, accu_%d = %f"%(epoch,float(accu_1),self.args['eval_k'], float(accu_k)))
        return accu_1,accu_k
    
    
    def save_model_stage_1(self,checkpoint_dir):
        self.graph_model.save_bert_model(checkpoint_dir)
        self.graph_model.save_sage_model(checkpoint_dir)
        self.args['logger'].info('graph model saved at %s'%checkpoint_dir)

    def load_model_stage_1(self,checkpoint_dir):
        self.graph_model.load_pretrained_model(checkpoint_dir)
        self.graph_model.load_sage_model(checkpoint_dir)
        self.args['logger'].info('graph model loaded at %s'%checkpoint_dir)
