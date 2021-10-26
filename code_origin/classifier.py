from collections import defaultdict
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
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.utils.graph_shortest_path import graph_shortest_path

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
        self.queries_train, self.queries_valid, self.queries_test = data_split(query_id_array=query_id_array, is_unseen=self.args['is_unseen'], test_size=0.33,seed=self.args['seed'])# data split
        self.tokenizer = BertTokenizer(vocab_file=self.args['vocab_file'])
        #包含bert部分和graph部分
        self.graph_model = Graphsage_Model(feature_size=768,hidden_size=256,output_size=768,model_path=self.args['pretrain_model_path'],with_graph=self.args['with_graph'],graph_rate=self.args['graph_rate'])

        children2parent = defaultdict(set)
        for h,r,t in self.triples:
            if r=='is_a':
                children2parent[h].add(t)
        self.siblings = set()
        self.grandpas = set()
        self.grandsons = set()
        for a in range(len(self.name_array)):
            for c in range(len(self.name_array)):
                if len(children2parent[a].intersection(children2parent[c]))>0:
                    self.siblings.add((a,c))
                    self.siblings.add((c,a))
        for a in range(len(self.name_array)):
            for parent in children2parent[a]:
                for grandparent in children2parent[parent]:
                    self.grandpas.add((grandparent,a))
                    self.grandsons.add((a,grandparent))

        #np.save(self.name_array,'name_query.npy')

        #print(self.siblings)
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
        
        return torch.LongTensor(edge_index).cuda()
    
    #计算similarity并画箱型图
    @torch.no_grad()
    def get_similarity(self):
        def cos_similar(p: torch.Tensor, q: torch.Tensor):
            sim_matrix = p.matmul(q.transpose(-2, -1))
            a = torch.norm(p, p=2, dim=-1)
            b = torch.norm(q, p=2, dim=-1)
            sim_matrix /= a.unsqueeze(-1)
            sim_matrix /= b.unsqueeze(-2)
            return sim_matrix
        names_bert_embedding = self.get_names_bert_embedding()
        edge_index = self.get_edge_index().cuda()
        names_graph_embedding = self.graph_model.get_graph_embedding(names_embedding=names_bert_embedding,edge_index=edge_index).cuda()
        names_embedding = (names_bert_embedding+names_graph_embedding * self.args['graph_rate'])
        cos_sim_matrix = cos_similar(names_embedding,names_embedding)
        
        torch.save(cos_sim_matrix,'cos_sim_matrix_hp_seen.pth')
        print(cos_sim_matrix.shape)
        print('---cos_sim_matrix---')
        print(cos_sim_matrix)
        dist_matrix=np.array(torch.zeros_like(cos_sim_matrix,dtype=torch.int).cpu())
        
        for h,t in zip(edge_index[0],edge_index[1]):
            dist_matrix[h][t]=1
            
        dist_matrix = graph_shortest_path(dist_matrix,directed=False)
        for i in range(cos_sim_matrix.shape[0]):
            dist_matrix[i][i]=0
        #np.save('dist_matrix_hp_seen.npy',dist_matrix)
        
        #cos_sim_matrix = torch.Tensor(cos_sim_matrix)
        dist_matrix = torch.LongTensor(dist_matrix)
        sim_res = [cos_sim_matrix[dist_matrix == i] for i in range(1, 6)]
        
        print(len(sim_res[0]),len(sim_res[1]),len(sim_res[2]),len(sim_res[3]),len(sim_res[4]))

        #plt.boxplot(sim_res,labels=["1","2","3","4","5"])
        
        s1 = pd.Series(np.array(sim_res[0].cpu()))
        s2 = pd.Series(np.array(sim_res[1].cpu()))
        s3 = pd.Series(np.array(sim_res[2].cpu()))
        s4 = pd.Series(np.array(sim_res[3].cpu()))
        s5 = pd.Series(np.array(sim_res[4].cpu()))
        # 把四个list导入到pandas的数据结构中，dataframe
        data = pd.DataFrame({"1": s1, "2": s2, "3": s3, "4": s4,"5":s5})
        data.boxplot()  # 这里，pandas自己有处理的过程，很方便哦。
        
        plt.ylabel("ylabel")
        plt.xlabel("xlabel")  # 我们设置横纵坐标的标题。
        plt.show()

        plt.savefig('sim_res_hp_seen.png')
        print('done')
    
    #得到跳数分布
    @torch.no_grad()
    def get_hop_count(self,query_array):
        edge_index = self.get_edge_index()
        names_bert_embedding = self.get_names_bert_embedding()# tensor of shape(N,768)
        dist_matrix=np.array(torch.zeros(size=(len(self.name_array),len(self.name_array)),dtype=torch.int).cpu())
        
        for h,t in zip(edge_index[0],edge_index[1]):
            dist_matrix[h][t]=1
        
        dist_matrix = graph_shortest_path(dist_matrix,directed=False)
        dist_matrix[dist_matrix==0]=-1
        for i in range(len(self.name_array)):
            dist_matrix[i][i]=0
        """
        neis = torch.load('neis.bin')
        for dist in range(6):
            for center in neis[dist]:
                for x in neis[dist][center]:
                    assert dist_matrix[center][x] == dist_matrix[x][center] == dist
        """

        self.graph_model.eval()
        eval_dataset = Graph_Dataset(query_array=query_array,mention2id=self.mention2id,tokenizer=self.tokenizer)
        eval_loader = DataLoader(dataset=eval_dataset,batch_size = 1024,shuffle=False)
        accu_1 = torch.FloatTensor([0]).cuda()
        accu_k = torch.FloatTensor([0]).cuda()
        results=[0 for _ in range(6)]
        res_2_hop={'sib':0,'grandpa':0,'grandson':0,'other':0}
        with torch.no_grad():
            pbar = tqdm(enumerate(eval_loader),total=len(eval_loader))
            for iteration,(batch_query_ids, batch_query_attention_mask,batch_labels,batch_query) in pbar:
                with TimeIt('make batch'):
                    batch_query_ids = batch_query_ids.cuda()
                    batch_query_attention_mask = batch_query_attention_mask.cuda()
                    batch_labels =batch_labels.cuda().squeeze()
                    batch_query = np.array(batch_query)
                
                with TimeIt('get scores'):
                    outputs = self.graph_model.forward(query_ids=batch_query_ids,query_attention_mask=batch_query_attention_mask,names_embedding=names_bert_embedding,edge_index=edge_index)
                
                values,indices = torch.topk(input=outputs,k=self.args['eval_k'],dim=-1)

                for i in range(len(batch_labels)):
                    label = batch_labels[i]
                    for j in range(1):
                        pred = indices[i][j]
                        hop = int(dist_matrix[label][pred])
                        if hop>=0 and hop<=5:
                            results[hop]+=1
                            if hop==2:
                                if (int(label),int(pred)) in self.siblings:
                                    res_2_hop['sib']+=1
                                elif (int(label),int(pred)) in self.grandpas:
                                    res_2_hop['grandpa'] +=1
                                elif (int(label),int(pred)) in self.grandsons:
                                    res_2_hop['grandson']+=1
                                else: res_2_hop['other'] +=1
                
        results = [float(r/len(query_array)/1) for r in results]
        _sum = sum(res_2_hop.values())
        res_2_hop={key:value/_sum for (key,value) in res_2_hop.items()}

        for i,value in enumerate(results):
            print("hop-%d neighbor in top_10, value = %f"%(i,value))
        print(res_2_hop)
                

    def train(self):
        self.args['logger'].info('stage_1_training')
        train_dataset = Graph_Dataset(query_array=self.queries_train,mention2id=self.mention2id,tokenizer=self.tokenizer)
        train_loader = DataLoader(dataset=train_dataset,batch_size=self.args['batch_size'],shuffle=True, drop_last=True) ###########
        criterion = nn.CrossEntropyLoss(reduction='sum')# take it as an multi class task
        edge_index = self.get_edge_index(undirected=True).cuda()
        
        optimizer = torch.optim.Adam([
                {'params': self.graph_model.bert_encoder.parameters()},
                {'params': self.graph_model.sage1.parameters(), 'lr':1e-3},
                {'params': self.graph_model.sage2.parameters(), 'lr':1e-3},
                {'params': self.graph_model._cls_bn.parameters(), 'lr': 1e-3},
                ], 
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
                if iteration % 10 == 0:#每1step更新一次
                    with TimeIt('get emb'):
                        names_bert_embedding = self.get_names_bert_embedding()# tensor of shape(N,768)
            
                optimizer.zero_grad()
                with TimeIt('make batch'):
                    batch_query_ids = batch_query_ids.cuda()
                    batch_query_attention_mask = batch_query_attention_mask.cuda()
                    batch_labels =batch_labels.cuda().squeeze()
                    batch_query = np.array(batch_query)
                
                
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


            accu_1,accu_k = self.eval(query_array=self.queries_valid,epoch=epoch)
            print('loss_sum:', float(loss_sum))
            if accu_1>accu_temp:
                accu_temp = accu_1
                checkpoint_dir = os.path.join(self.args['exp_path'],'checkpoint')
                self.save_model(checkpoint_dir)

    
    @torch.no_grad()
    def eval(self,query_array,epoch,load_model = False):
        
        all_label = []
        all_pred = []
        self.graph_model.eval()
        names_bert_embedding = self.get_names_bert_embedding()# tensor of shape(N,768)
        edge_index = self.get_edge_index(True).cuda()
        if self.args['replace_graph']:
            names_graph_embedding = self.graph_model.get_graph_embedding(names_embedding=names_bert_embedding,edge_index=edge_index)
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
                    batch_labels =batch_labels.cuda().squeeze()
                    batch_query = np.array(batch_query)
                
                with TimeIt('get scores'):
                    outputs = self.graph_model.forward(query_ids=batch_query_ids,query_attention_mask=batch_query_attention_mask,names_embedding=names_bert_embedding,edge_index=edge_index)
                
                values,indices = torch.topk(input=outputs,k=self.args['eval_k'],dim=-1)
                
                if self.args['replace_graph']:
                    query_bert_embedding =  self.graph_model.bert_encoder(batch_query_ids,batch_query_attention_mask).last_hidden_state[:,0,:]# batch * hidden_size
                    
                    batch_size = batch_labels.shape[0]
                    replace_score = torch.Tensor([[-np.inf]])#默认为负无穷
                    replace_score=replace_score.expand_as(outputs).cuda() #记录replace graph之后重新计算的图的分数，batch*E,非candidate的默认为负无穷
                    #print(replace_score.shape)

                    for i in range(batch_size):
                        for entity_index in indices[i]:
                            replace_names_embedding = torch.zeros_like(names_bert_embedding).cuda()
                            replace_names_embedding.copy_(names_bert_embedding)
                            replace_names_embedding[entity_index] = query_bert_embedding[i]#feature完成替换
                            replace_names_embedding = self.graph_model.get_graph_embedding(names_embedding=replace_names_embedding,edge_index=edge_index)
                            query_embedding = replace_names_embedding[entity_index] + query_bert_embedding[i]
                            entity_embedding = self.args['graph_rate'] * names_graph_embedding[entity_index] + names_bert_embedding[entity_index]
                            replace_score[i][entity_index] = (query_embedding * entity_embedding).sum()


                all_label.append(batch_labels)
                all_pred.append(indices[:,0])
                if self.args['replace_graph']:
                    preds = torch.argmax(replace_score,dim=-1)
                    accu_1 +=(preds==batch_labels).sum()/len(query_array)
                else:
                    accu_1 += (indices[:,0]==batch_labels).sum()/len(query_array)
                    
                accu_k += (indices== torch.unsqueeze(batch_labels,dim=1)).sum()/len(query_array)
        self.args['logger'].info("epoch %d done, accu_1 = %f, accu_%d = %f"%(epoch,float(accu_1),self.args['eval_k'], float(accu_k)))
        all_label = torch.cat(all_label,dim=0).cpu()
        all_pred = torch.cat(all_pred,dim=0).cpu()

        results=[str(int(all_label[i]))+'\t'+str(int(all_pred[i]))+'\t'+query_array[i]+'\t'+str(int(self.mention2id[query_array[i]]))+'\n' for i in range(len(query_array))]

        #print(results[:3])
        if self.args['output_file'] !='none':
            with open(self.args['output_file'],mode='w',encoding='utf-8') as f:
                f.writelines(results)
        
        return accu_1,accu_k
    
    
    def save_model(self,checkpoint_dir):
        self.graph_model.save_bert_model(checkpoint_dir)
        self.graph_model.save_sage_model(checkpoint_dir)
        self.args['logger'].info('graph model saved at %s'%checkpoint_dir)

    def load_model(self,checkpoint_dir):
        self.graph_model.load_pretrained_model(checkpoint_dir)
        self.graph_model.load_sage_model(checkpoint_dir)
        self.args['logger'].info('graph model loaded at %s'%checkpoint_dir)

class Biosyn_Classifier():
    def __init__(self,args):
        self.args = args
        self.filename = self.args['filename']
        self.use_text_preprocesser = self.args['use_text_preprocesser']
        self.name_array,query_id_array,self.mention2id,self.egde_index = load_data(self.filename,self.use_text_preprocesser)
        self.queries_train,self.queries_valid,self.queries_test = data_split(query_id_array=query_id_array,is_unseen=self.args['is_unseen'],test_size=0.33,seed=self.args['seed'])
        self.tokenizer = BertTokenizer(vocab_file=self.args['vocab_file'])

        self.biosyn_model =Biosyn_Model(model_path = self.args['pretrain_model_path'],initial_sparse_weight = self.args['initial_sparse_weight'])
        
        self.sparse_encoder = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))# only works on cpu
        self.sparse_encoder.fit(self.name_array)
        
    # get the embeddings of mention_array(name_array or query_array)
    def get_mention_array_bert_embedding(self,mention_array):
        
        # use dataset to help embed the mention_array
        self.biosyn_model.eval()#进入eval模式
        
        mention_dataset = Mention_Dataset(mention_array,self.tokenizer)
        mentions_embedding = []
        data_loader = DataLoader(dataset=mention_dataset,batch_size=1024)
        with torch.no_grad():# here we just use this function to retrieve the candidates first, so we set torch no grad
            for i,(input_ids, attention_mask) in enumerate(data_loader):
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                cls_embedding = self.biosyn_model.bert_encoder(input_ids,attention_mask).last_hidden_state[:,0,:]# batch * hidden_size
                mentions_embedding.append(cls_embedding)
            
            mentions_embedding = torch.cat(mentions_embedding, dim=0)# len(mentions) * hidden_size
            #print(mentions_embedding.shape)
        
        return mentions_embedding# still on the device

    # this function will use too much memory, so we calculate the score for single batch
    def get_score_matrix(self,query_array):
        query_sparse_embedding = torch.FloatTensor(self.sparse_encoder.transform(query_array).toarray()).cuda()
        name_sparse_embedding = torch.FloatTensor(self.sparse_encoder.transform(self.name_array).toarray()).cuda()
        sparse_score_matrix = torch.matmul(query_sparse_embedding,name_sparse_embedding.transpose(0,1))

        query_bert_embedding = self.get_mention_array_bert_embedding(query_array).cuda()
        name_bert_embedding = self.get_mention_array_bert_embedding(self.name_array).cuda()
        bert_score_matrix = torch.matmul(query_bert_embedding,name_bert_embedding.transpose(0,1))

        return sparse_score_matrix,bert_score_matrix

    def train(self):
        print('in train')
        criterion = marginal_loss
        optimizer = torch.optim.Adam([
            {'params': self.biosyn_model.bert_encoder.parameters()},
            {'params': self.biosyn_model.sparse_weight, 'lr': 0.001, 'weight_decay': 0}
            ], 
            lr=self.args['lr'], weight_decay=self.args['weight_decay']
        )
        t_total = self.args['epoch_num'] * len(self.queries_train) / self.args['batch_size']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='max',factor=0.8,patience=2)
        accu_temp = 0
        for epoch in range(1, self.args['epoch_num'] + 1):
            loss_sum = 0
            self.biosyn_model.train()

            names_sparse_embedding = torch.FloatTensor(self.sparse_encoder.transform(self.name_array).toarray()).cuda()
            names_bert_embedding = self.get_mention_array_bert_embedding(self.name_array).cuda()


            biosyn_dataset = Biosyn_Dataset(self.name_array,self.queries_train,self.mention2id,self.args['top_k'],
            sparse_encoder=self.sparse_encoder,bert_encoder=self.biosyn_model.bert_encoder,
            names_sparse_embedding=names_sparse_embedding,names_bert_embedding=names_bert_embedding, 
            bert_ratio=self.args['bert_ratio'],tokenizer=self.tokenizer)

            data_loader = DataLoader(dataset=biosyn_dataset,batch_size=self.args['batch_size'])
            pbar = tqdm(enumerate(data_loader),total=len(data_loader))
            for iteration,batch_data in pbar:

                optimizer.zero_grad()

                query_ids,query_attention_mask,candidates_names_ids,candidates_names_attention_mask,candidates_sparse_score,labels = batch_data
                query_ids = query_ids.cuda()
                query_attention_mask = query_attention_mask.cuda()
                candidates_names_ids = candidates_names_ids.cuda()
                candidates_names_attention_mask = candidates_names_attention_mask.cuda()
                candidates_sparse_score = candidates_sparse_score.cuda()
                labels = labels.cuda()
                score = self.biosyn_model.forward(query_ids,query_attention_mask,candidates_names_ids,candidates_names_attention_mask,candidates_sparse_score)
                
                loss = criterion(score,labels)
                loss_sum+=loss.item()
                loss.backward()
                optimizer.step()
                
                m, M, mean, std = score.min(), score.max(), score.mean(), score.std()
                pbar.set_postfix({"loss": float(loss), "[min, max, mean, std]": ['%.2e'%i for i in [m, M, mean, std]], "lr":['%.2e'%group["lr"] for group in optimizer.param_groups]})
            
            loss_sum/=len(self.queries_train)


            
            accu_1,accu_k = self.eval(self.queries_valid,epoch = epoch)
            scheduler.step(accu_1)
            if accu_temp<accu_1:
                accu_temp = accu_1
                checkpoint_dir = os.path.join(self.args['exp_path'], "checkpoint")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                self.save_model(checkpoint_dir)
            

    #@torch.no_grad()
    def eval(self,query_array,load_model=False,epoch = 0):
        self.biosyn_model.eval()# for nn.module
        accu_1 = torch.FloatTensor([0]).cuda()
        accu_k = torch.FloatTensor([0]).cuda()

        with torch.no_grad():
            eval_dataloader = DataLoader(dataset=query_array,batch_size=1024,shuffle=False)
            for array in eval_dataloader:
                sparse_score_matrix,bert_score_matrix = self.get_score_matrix(array)
                if self.args['score_mode'] == 'hybrid':
                    score_matrix = self.biosyn_model.sparse_weight * sparse_score_matrix + bert_score_matrix
                elif self.args['score_mode'] == 'sparse':
                    score_matrix = sparse_score_matrix
                else:
                    score_matrix = bert_score_matrix
                sorted,indices = torch.sort(score_matrix,descending=True)# 降序，重要
                query_indices = torch.LongTensor([self.mention2id[query] for query in array]).cuda()
                accu_1 += (indices[:,0]==query_indices).sum()/len(query_array)
                accu_k += (indices[:,:self.args['eval_k']]== torch.unsqueeze(query_indices,dim=1)).sum()/len(query_array)
        self.args['logger'].info("epoch %d done, accu_1 = %f, accu_%d = %f"%(epoch,float(accu_1),self.args['eval_k'], float(accu_k)))
        return accu_1,accu_k
        
    def save_model(self,checkpoint_dir):
        self.biosyn_model.bert_encoder.save_pretrained(checkpoint_dir)
        torch.save(self.biosyn_model.sparse_weight,os.path.join(checkpoint_dir,'sparse_weight.pth'))

    def load_model(self,model_path):
        self.args['logger'].info('model loaded at %s'%model_path)
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.biosyn_model.bert_encoder.load_state_dict(state_dict,False)
        self.biosyn_model.sparse_weight = torch.load(os.path.join(model_path,'sparse_weight.pth'))
        

class CrossEncoder_Classifier():
    def __init__(self,args):
        self.args = args
        self.filename = self.args['filename']
        self.use_text_preprocesser = self.args['use_text_preprocesser']
        self.name_array, query_id_array, self.mention2id, self.edge_index, self.triples = simple_load_data(self.filename, self.use_text_preprocesser, return_triples=True)#load data
        self.queries_train, self.queries_valid, self.queries_test = data_split(query_id_array=query_id_array, is_unseen=self.args['is_unseen'], test_size=0.33,seed=self.args['seed'])# data split
        self.tokenizer = BertTokenizer(vocab_file=self.args['vocab_file'])

        # the entire stage_1 model
        self.bert_candidate_generator =Bert_Candidate_Generator(model_path = self.args['pretrain_model_path'],initial_sparse_weight = self.args['initial_sparse_weight'])
        self.sparse_encoder = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))# only works on cpu
        self.sparse_encoder.fit(self.name_array)
        sparse_feature_size = torch.FloatTensor(self.sparse_encoder.transform(self.queries_test).toarray()).shape[1]

        # the entire stage_2 model
        self.bert_cross_encoder = Bert_Cross_Encoder(model_path = self.args['pretrain_model_path'],feature_size=768+sparse_feature_size,sparse_encoder=self.sparse_encoder)


    #we can not put all the names bert in the calculation graph, otherwise we will get an out of memory error
    @torch.no_grad()
    def get_names_bert_embedding(self):
        names_dataset = Mention_Dataset(self.name_array,self.tokenizer)
        names_bert_embedding = []
        data_loader = DataLoader(dataset=names_dataset,batch_size=1024)
        for i,(input_ids, attention_mask) in enumerate(data_loader):
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            cls_embedding = self.bert_candidate_generator.bert_encoder(input_ids,attention_mask).last_hidden_state[:,0,:]# batch * hidden_size
            names_bert_embedding.append(cls_embedding)
            
        names_bert_embedding = torch.cat(names_bert_embedding, dim=0)# len(mentions) * hidden_size
        return names_bert_embedding# still on the device
    
    def get_names_sparse_embedding(self):
        names_sparse_embedding = torch.FloatTensor(self.sparse_encoder.transform(self.name_array).toarray()).cuda()# tensor of shape(N,hidden)
        return names_sparse_embedding# still on device
    

    @torch.no_grad()
    def candidates_retrieve_separate(self,batch_query_ids, batch_query_attention_mask,batch_query_index,batch_query,names_sparse_embedding,names_bert_embedding,top_k,is_training):
        batch_query_sparse_embedding = torch.FloatTensor(self.sparse_encoder.transform(batch_query).toarray()).cuda()# tensor of shape(batch,hidden)
        batch_query_bert_embedding = self.bert_candidate_generator.bert_encoder(batch_query_ids,batch_query_attention_mask).last_hidden_state[:,0,:]# shape of (batch,hidden_size)
        sparse_score = torch.matmul(batch_query_sparse_embedding,torch.transpose(names_sparse_embedding,dim0=0,dim1=1))# tensor of shape(batch,N)
        bert_score = torch.matmul(batch_query_bert_embedding,torch.transpose(names_bert_embedding,dim0=0,dim1=1))
        
        # we get sparse indices and bert indices separately
        sorted_sparse_score,sparse_indices = torch.sort(sparse_score,descending=True) 
        sorted_bert_score,bert_indices = torch.sort(bert_score,descending=True)
        n_bert = int(top_k * self.args['bert_ratio'])
        n_sparse = top_k - n_bert
        batch_size = batch_query_attention_mask.shape[0]
        candidates_indices = torch.LongTensor(size=(batch_size,top_k)).cuda()
        candidates_indices[:,:n_sparse] =  sparse_indices[:,:n_sparse]
        for i in range(batch_size):
            j=0
            for k in range(n_sparse,top_k):
                while bert_indices[i][j] in candidates_indices[i][:n_sparse]:
                    j+=1
                bert_index = bert_indices[i][j]
                candidates_indices[i][k] = bert_index
                j+=1
            assert(len(candidates_indices[i]) == len(candidates_indices[i].unique()))
        # put the ground truth index in the training data
        if is_training:
            for i in range(batch_size):
                query_index = batch_query_index[i]
                if query_index not in candidates_indices[i]:
                    candidates_indices[i][-1] = query_index
        
        # calculate candidates score according to the candidates_indices
        candidates_sparse_score = []
        for i in range(batch_size):
            candidates_sparse_score.append(torch.unsqueeze(sparse_score[i][candidates_indices[i]],dim=0))
        candidates_sparse_score = torch.cat(candidates_sparse_score,dim=0).cuda()# shape(batch,top_k)
        return candidates_indices,candidates_sparse_score#tensors of shape(batch,top_k)

    @torch.no_grad()# with pretrained and fine tuned bert model, we could get candidates with about 80 accu_k
    def candidates_retrieve_mix(self,batch_query_ids, batch_query_attention_mask,batch_query_index,batch_query,names_sparse_embedding,names_bert_embedding,top_k,is_training):
        batch_query_sparse_embedding = torch.FloatTensor(self.sparse_encoder.transform(batch_query).toarray()).cuda()# tensor of shape(batch,hidden)
        batch_query_bert_embedding = self.bert_candidate_generator.bert_encoder(batch_query_ids,batch_query_attention_mask).last_hidden_state[:,0,:]# shape of (batch,hidden_size)
        sparse_score = torch.matmul(batch_query_sparse_embedding,torch.transpose(names_sparse_embedding,dim0=0,dim1=1))# tensor of shape(batch,N)
        bert_score = torch.matmul(batch_query_bert_embedding,torch.transpose(names_bert_embedding,dim0=0,dim1=1))
        bert_score = self.bert_candidate_generator.cls_bn(bert_score)
        score = self.bert_candidate_generator.sparse_weight * sparse_score + bert_score# shape(batch,N)
        sorted_score,candidates_indices =torch.sort(score,descending=True)# descending
        candidates_indices = candidates_indices[:,:top_k]

        batch_size = candidates_indices.shape[0]
        if is_training:
            for i in range(batch_size):
                query_index = batch_query_index[i]
                if query_index not in candidates_indices[i]:
                    candidates_indices[i][-1] = query_index

        candidates_sparse_score = []
        for i in range(batch_size):
            candidates_sparse_score.append(torch.unsqueeze(sparse_score[i][candidates_indices[i]],dim=0))
        candidates_sparse_score = torch.cat(candidates_sparse_score,dim=0).cuda()# shape(batch,top_k)


        return candidates_indices,candidates_sparse_score#tensors of shape(batch,top_k)

    def train_stage_1(self):
        self.args['logger'].info('stage_1_training')
        train_dataset = Graph_Dataset(query_array=self.queries_train,mention2id=self.mention2id,tokenizer=self.tokenizer)
        train_loader = DataLoader(dataset=train_dataset,batch_size=self.args['batch_size'],shuffle=True, drop_last=True) ###########
        criterion = nn.CrossEntropyLoss(reduction='sum')# take it as an multi class task
        
        if self.args['initial_sparse_weight'] != 0:
            optimizer = torch.optim.Adam([
                {'params': self.bert_candidate_generator.bert_encoder.parameters()},
                {'params': self.bert_candidate_generator.sparse_weight, 'lr': 0.001, 'weight_decay': 0},
                {'params': self.bert_candidate_generator._cls_bn.parameters(), 'lr': 0.001},
                ], 
                lr=self.args['lr'], weight_decay=self.args['weight_decay']
            )
        else:
            optimizer = torch.optim.Adam([
                {'params': self.bert_candidate_generator.bert_encoder.parameters()},
                {'params': self.bert_candidate_generator._cls_bn.parameters(), 'lr': 0.001},
                ], 
                lr=self.args['lr'], weight_decay=self.args['weight_decay']
            )
        t_total = self.args['epoch_num'] * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

        accu_temp = 0
        for epoch in range(1,self.args['epoch_num'] + 1):
            # #every epoch we recalculate the embeddings which have been updated
            # names_bert_embedding = self.get_names_bert_embedding()# tensor of shape(N,768)
            # names_sparse_embedding = self.get_names_sparse_embedding()# tensor of shape(N, sparse_feature_size)
            self.bert_candidate_generator.train()
            loss_sum = 0
            
            pbar = tqdm(enumerate(train_loader),total=len(train_loader))
            for iteration,(batch_query_ids, batch_query_attention_mask,batch_query_index,batch_query) in pbar:
                if iteration % 100 == 0:
                    with TimeIt('get emb'):
                        names_bert_embedding = self.get_names_bert_embedding()# tensor of shape(N,768)
                        names_sparse_embedding = self.get_names_sparse_embedding()# tensor of shape(N, sparse_feature_size)
            
                optimizer.zero_grad()

                with TimeIt('make batch'):
                    batch_query_ids = batch_query_ids.cuda()
                    batch_query_attention_mask = batch_query_attention_mask.cuda()
                    batch_query_index =batch_query_index.cuda().squeeze(dim=1)
                    batch_query = np.array(batch_query)

                with TimeIt('candidates_retrieve_separate'):
                    candidates_indices,candidates_sparse_score = self.candidates_retrieve_separate(
                        batch_query_ids, batch_query_attention_mask,batch_query_index,batch_query,
                        names_sparse_embedding,names_bert_embedding,top_k=self.args['top_k'],is_training=True
                    )# tensors of shape (batch,top_k)

                with TimeIt('get_batch_inputs_for_stage_1'):
                    batch_size = batch_query_ids.shape[0]
                    labels = self.get_labels(batch_size=batch_size,candidates_indices=candidates_indices,batch_query_index=batch_query_index)
                    candidates_ids,candidates_attention_mask = self.get_batch_inputs_for_stage_1(
                        batch_size=batch_size,candidates_indices=candidates_indices)
                
                with TimeIt('bert_candidate_generator'):
                    outputs = self.bert_candidate_generator.forward(
                        query_ids=batch_query_ids,query_attention_mask=batch_query_attention_mask,
                        candidates_ids=candidates_ids,candidates_attention_mask=candidates_attention_mask,
                        candidates_sparse_score=candidates_sparse_score
                        )
                    #print(labels)
                    assert((labels==-1).sum()==0)
                    loss = criterion(outputs,labels)

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

        all_label=[]
        all_pred = []

        self.bert_candidate_generator.eval()
        names_bert_embedding = self.get_names_bert_embedding()# tensor of shape(N,768)
        names_sparse_embedding = self.get_names_sparse_embedding()
        
        eval_dataset = Graph_Dataset(query_array=query_array,mention2id=self.mention2id,tokenizer=self.tokenizer)
        eval_loader = DataLoader(dataset=eval_dataset,batch_size = 1024,shuffle=False)
        
        accu_1 = torch.FloatTensor([0]).cuda()
        accu_k = torch.FloatTensor([0]).cuda()
        with torch.no_grad():
            for iteration,(batch_query_ids, batch_query_attention_mask,batch_query_index,batch_query) in tqdm(enumerate(eval_loader),total=len(eval_loader)):
            
                batch_query_ids = batch_query_ids.cuda()
                batch_query_attention_mask = batch_query_attention_mask.cuda()
                batch_query_index =batch_query_index.cuda().squeeze()
                batch_query = np.array(batch_query)

                candidates_indices,candidates_sparse_score = self.candidates_retrieve_mix(
                    batch_query_ids, batch_query_attention_mask,batch_query_index,batch_query,
                    names_sparse_embedding,names_bert_embedding,top_k=self.args['top_k'],is_training=False
                )# tensors of shape (batch,top_k)

                batch_size = batch_query_index.shape[0]

                labels = self.get_labels(batch_size=batch_size,candidates_indices=candidates_indices,batch_query_index=batch_query_index)

                
                all_label.append(batch_query_index)
                all_pred.append(candidates_indices[:,0])
                accu_1 += (candidates_indices[:,0]==batch_query_index).sum()/len(query_array)
                accu_k += (candidates_indices[:,:self.args['eval_k']]== torch.unsqueeze(batch_query_index,dim=1)).sum()/len(query_array)
        self.args['logger'].info("epoch %d done, accu_1 = %f, accu_%d = %f"%(epoch,float(accu_1),self.args['eval_k'], float(accu_k)))
        
        all_label = torch.cat(all_label,dim=0).cpu()
        all_pred = torch.cat(all_pred,dim=0).cpu()
        results=[str(int(all_label[i]))+'\t'+str(int(all_pred[i]))+'\t'+query_array[i]+'\t'+str(int(self.mention2id[query_array[i]]))+'\n' for i in range(len(query_array))]

        #print(results[:3])
        if self.args['output_file'] !='none':
            with open(self.args['output_file'],mode='w',encoding='utf-8') as f:
                f.writelines(results)

        return accu_1,accu_k
    
    
    def save_model_stage_1(self,checkpoint_dir):
        self.bert_candidate_generator.save_model(checkpoint_dir)
        self.args['logger'].info('stage 1 model saved at %s'%checkpoint_dir)

    def load_model_stage_1(self,checkpoint_dir):
        self.bert_candidate_generator.load_model(checkpoint_dir)
        self.args['logger'].info('stage 1 model loaded at %s'%checkpoint_dir)

    def train_stage_2(self):
        # we need to load stage 1 model before stage 2 training
        self.bert_candidate_generator.load_model(model_path=self.args['pretrain_model_path'])

        self.args['logger'].info('stage_2_training')

        train_dataset = Graph_Dataset(query_array=self.queries_train,mention2id=self.mention2id,tokenizer=self.tokenizer)
        train_loader = DataLoader(dataset=train_dataset,batch_size=self.args['batch_size'],shuffle=False)
        criterion = nn.CrossEntropyLoss(reduction='sum')# take it as an multi class task
        

        ################# todo! 
        optimizer = torch.optim.AdamW([
            {'params': self.bert_cross_encoder.bert_encoder.parameters(),'lr':1e-5,'weight_decay':0},
            {'params': self.bert_cross_encoder.score_network.parameters(), 'lr': 0.001, 'weight_decay': 0}
            ]
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='max',patience = 3,factor=0.35)


        # the stage_1 model is fixed during stage 2, we do not need to recalculate them
        names_sparse_embedding = self.get_names_sparse_embedding()# tensor of shape(N,hidden)
        names_bert_embedding = self.get_names_bert_embedding()# tensor of shape(N,768)
        for epoch in range(1,self.args['epoch_num'] + 1):
            
            #self.adjust_learning_rate(optimizer,epoch)
            print(optimizer.param_groups[0]['lr'])
            self.bert_cross_encoder.train()
            
            loss_sum = 0
            for iteration,(batch_query_ids, batch_query_attention_mask,batch_query_index,batch_query) in tqdm(enumerate(train_loader),total=len(train_loader)):
            
                optimizer.zero_grad()
                batch_query_ids = batch_query_ids.cuda()# tensor of shape(batch,top_k,max_len)
                batch_query_attention_mask = batch_query_attention_mask.cuda()#tensor of shape(batch,top_k,max_len)
                batch_query_index =batch_query_index.cuda().squeeze(dim=1)# tensor of shape(batch,)
                batch_query = np.array(batch_query)# str array of shape(batch,)

                candidates_indices,candidates_sparse_score = self.candidates_retrieve_mix(
                    batch_query_ids, batch_query_attention_mask,batch_query_index,batch_query,
                    names_sparse_embedding,names_bert_embedding,top_k=self.args['top_k'],is_training=True
                )# tensors of shape (batch,top_k)
                
                batch_size = batch_query_index.shape[0]
                labels = self.get_labels(batch_size=batch_size,candidates_indices=candidates_indices,batch_query_index=batch_query_index)
                
                #print(labels)
                assert((labels==-1).sum()==0)

                """
                batch_pair_ids,batch_pair_attn_mask = self.get_batch_inputs_for_stage_2(
                    batch_query_index=batch_query_index,batch_query=batch_query,candidates_indices=candidates_indices
                    )
                """

                candidates_ids,candidates_attention_mask = self.get_batch_inputs_for_stage_1(
                    batch_size=batch_size,candidates_indices=candidates_indices)
                
                outputs = self.bert_cross_encoder.forward(
                    query_ids=batch_query_ids,query_attention_mask=batch_query_attention_mask,
                    candidates_ids=candidates_ids,candidates_attention_mask=candidates_attention_mask,
                    query=batch_query,names_sparse_embedding = names_sparse_embedding,candidates_indices = candidates_indices)
                
                # when training, ground truth is included in the candidates

                loss = criterion(outputs,labels)

                loss_sum+=loss.item()
                loss.backward()
                optimizer.step()
                loss_sum/=len(self.queries_train)
                loss=0
                
            #print(self.bert_cross_encoder.linear.weight)
            print('loss_sum')
            print(loss_sum)
            accu_1,accu_k = self.eval_stage_2(self.queries_valid,epoch=epoch)
            scheduler.step(accu_1)

        checkpoint_dir = os.path.join(self.args['exp_path'],'epoch%d'%self.args['epoch_num'])
        self.save_model_stage_2(checkpoint_dir)

    @torch.no_grad()
    def eval_stage_2(self,query_array,epoch,load_model = False):
        self.bert_candidate_generator.eval()
        self.bert_cross_encoder.eval()
        names_sparse_embedding = self.get_names_sparse_embedding()# tensor of shape(N,hidden)
        names_bert_embedding = self.get_names_bert_embedding()# tensor of shape(N,768)
        eval_dataset = Graph_Dataset(query_array=query_array,mention2id=self.mention2id,tokenizer=self.tokenizer)
        eval_loader = DataLoader(dataset=eval_dataset,batch_size = 1024,shuffle=False)
        
        accu_1 = torch.FloatTensor([0]).cuda()
        accu_k = torch.FloatTensor([0]).cuda()
        with torch.no_grad():
            for iteration,(batch_query_ids, batch_query_attention_mask,batch_query_index,batch_query) in tqdm(enumerate(eval_loader),total=len(eval_loader)):
            
                batch_query_ids = batch_query_ids.cuda()
                batch_query_attention_mask = batch_query_attention_mask.cuda()
                batch_query_index =batch_query_index.cuda().squeeze()
                batch_query = np.array(batch_query)

                candidates_indices,candidates_sparse_score = self.candidates_retrieve_mix(
                    batch_query_ids, batch_query_attention_mask,batch_query_index,batch_query,
                    names_sparse_embedding,names_bert_embedding,top_k=self.args['top_k'],is_training=False
                )# tensors of shape (batch,top_k),remember that we set is_training to False

                """
                batch_pair_ids,batch_pair_attn_mask = self.get_batch_inputs_for_stage_2(
                    batch_query_index=batch_query_index,batch_query=batch_query,candidates_indices=candidates_indices
                    )
                """
                batch_size = batch_query_index.shape[0]

                labels = self.get_labels(batch_size=batch_size,candidates_indices=candidates_indices,batch_query_index=batch_query_index)
                #outputs = self.bert_cross_encoder.forward(batch_pair_ids,batch_pair_attn_mask)# tensors of shape(batch,top_k)

                candidates_ids,candidates_attention_mask = self.get_batch_inputs_for_stage_1(
                    batch_size=batch_size,candidates_indices=candidates_indices)
                
                outputs = self.bert_cross_encoder.forward(
                    query_ids=batch_query_ids,query_attention_mask=batch_query_attention_mask,
                    candidates_ids=candidates_ids,candidates_attention_mask=candidates_attention_mask,
                    query=batch_query,names_sparse_embedding = names_sparse_embedding,candidates_indices = candidates_indices)

                sorted_score,preds = torch.sort(outputs,descending=True)
                if iteration == int(len(eval_dataset)/batch_size):
                    print('---preds---')
                    print(preds.shape)
                    print(preds)
                    print('---labels---')
                    print(labels.shape)
                    print(labels)
                    print('---candidate_indices---')
                    print(candidates_indices.shape)
                    print(candidates_indices)
                    print('---batch_query_index---')
                    print(batch_query_index)

                accu_1 += (preds[:,0]==labels).sum()/len(query_array)
                repeated_labels = torch.unsqueeze(labels,dim=1).repeat(1,self.args['eval_k'])
                accu_k += (preds[:,:self.args['eval_k']]==repeated_labels).sum()/len(query_array)
                # for situations where ground truth is not in candidateds indices, still work(initial -1)
        
        self.args['logger'].info("epoch %d done, accu_1 = %f, accu_%d = %f"%(epoch,float(accu_1),self.args['eval_k'], float(accu_k)))
        return accu_1,accu_k

    def save_model_stage_2(self,checkpoint_dir):
        self.bert_cross_encoder.bert_encoder.save_pretrained(checkpoint_dir)
        
        torch.save(self.bert_cross_encoder.score_network.state_dict(),os.path.join(checkpoint_dir,'score_network.pth'))
        self.args['logger'].info('pretrain_model saved at %s'%checkpoint_dir)

    def get_batch_inputs_for_stage_1(self,batch_size,candidates_indices):
        candidates_ids,candidates_attention_mask = [],[]
        for i in range(batch_size):
            ids_k,mask_k=[],[]
            for k in range(self.args['top_k']):
                entity_index = candidates_indices[i][k]
                entity = self.name_array[entity_index]
                tokens = self.tokenizer(entity, add_special_tokens=True, max_length = 24, padding='max_length',truncation=True,return_attention_mask = True, return_tensors='pt')
                input_ids = torch.squeeze(tokens['input_ids']).reshape(1,-1)
                attention_mask = torch.squeeze(tokens['attention_mask']).reshape(1,-1)
                ids_k.append(input_ids)
                mask_k.append(attention_mask)
            ids_k = torch.cat(ids_k,dim=0).reshape(1,self.args['top_k'],-1)# tensor of shape(1,top_k,max_len)
            mask_k = torch.cat(mask_k,dim=0).reshape(1,self.args['top_k'],-1)
            candidates_ids.append(ids_k)
            candidates_attention_mask.append(mask_k)
                
        candidates_ids = torch.cat(candidates_ids,dim=0).cuda()
        candidates_attention_mask = torch.cat(candidates_attention_mask,dim=0).cuda()
        return candidates_ids,candidates_attention_mask
                        
    def get_batch_inputs_for_stage_2(self,batch_query_index,batch_query,candidates_indices):
        batch_size = batch_query_index.shape[0]
        batch_pair_ids = []
        batch_pair_attn_mask = []# record the tokens of (query,name) pairs
        batch_pair_type_ids = []
        for i in range(batch_size):
            pair_ids,pair_attn_mask,pair_type_ids=[],[],[]
            query = batch_query[i]
            query_index = batch_query_index[i]# label of name in name_array
            for k in range(self.args['top_k']):
                entity_index = candidates_indices[i][k]
                entity = self.name_array[entity_index]
                        
                # tokenizer (query,entity) pair together; for asingle pair
                tokens = self.tokenizer(query,entity,add_special_tokens=True, max_length = 24, padding='max_length',truncation=True,return_attention_mask = True, return_tensors='pt')
                k_ids,k_attn_mask,k_type_ids = torch.squeeze(tokens['input_ids']).cuda(),torch.squeeze(tokens['attention_mask']),torch.squeeze(tokens['token_type_ids'])# tensor of shape (max_len,)
                
                pair_ids.append(torch.unsqueeze(k_ids,dim=0))# list of tensor of shape(1,max_len)
                pair_attn_mask.append(torch.unsqueeze(k_attn_mask,dim=0))# list of tensor of shape(1,max_len)
                pair_type_ids.append(torch.unsqueeze(k_type_ids,dim=0))

            pair_ids = torch.cat(pair_ids,dim=0)
            pair_attn_mask = torch.cat(pair_attn_mask,dim=0)# tensor of shape(top_k,max_len)
            pair_type_ids = torch.cat(pair_type_ids,dim=0)#tensor of shape(top_k, max_len)

            batch_pair_ids.append(torch.unsqueeze(pair_ids,dim=0))# list of tensors of shape(top_k,max_len)
            batch_pair_attn_mask.append(torch.unsqueeze(pair_attn_mask,dim=0))
            batch_pair_type_ids.append(torch.unsqueeze(pair_type_ids,dim=0))

        batch_pair_ids = torch.cat(batch_pair_ids,dim=0).cuda()
        batch_pair_attn_mask = torch.cat(batch_pair_attn_mask,dim=0).cuda()
        batch_pair_type_ids = torch.cat(batch_pair_type_ids,dim=0).cuda()

        return batch_pair_ids,batch_pair_attn_mask,batch_pair_type_ids



    def get_labels(self,batch_size,candidates_indices,batch_query_index):
        labels = torch.LongTensor([-1] * batch_size).cuda()
        for i in range(batch_size):
            ids_k,mask_k=[],[]
            for k in range(self.args['top_k']):
                entity_index = candidates_indices[i][k]
                if entity_index==batch_query_index[i]:
                    labels[i] = k
        return labels







            










                    




            






