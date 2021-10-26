
import numpy as np
import random
import os
import torch
import logging
import argparse

from classifier import  Biosyn_Classifier,CrossEncoder_Classifier,Graph_Classifier




#set up seed         
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# set up logger
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    if not os.path.exists(name):
        os.makedirs(name)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',type=str,default='../data/datesets/cl.obo')
    parser.add_argument('--use_text_preprocesser',action='store_true',default=False)
    parser.add_argument('--zero_shot_mode',type=str,default='unseen')
    parser.add_argument('--exp_path',type=str,default='../exp/eco/cross_encoder/unseen')
    parser.add_argument('--run_stage_1',action='store_true',default=False)
    parser.add_argument('--run_stage_2',action='store_true',default=False)
    parser.add_argument('--train',action='store_true',default=False)
    parser.add_argument('--test',action='store_true',default=False)
    

    parser.add_argument('--model',type=str,default='biosyn')
    parser.add_argument('--pretrain_model_path',type=str,default='../biobert')
    parser.add_argument('--vocab_file',type=str,default='../biobert/vocab.txt')

    parser.add_argument('--load_model',action='store_true',default=False)


    parser.add_argument('--initial_sparse_weight',type=float,default=0.)
    parser.add_argument('--bert_ratio',type=float,default=0.5)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--weight_decay',type=float,default=0)

    parser.add_argument('--epoch_num',type=int,default=50)
    parser.add_argument('--top_k',type=int,default=10)
    parser.add_argument('--eval_k',type=int,default=10)
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--score_mode',type=str,default='hybrid')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--save_checkpoint_all',action='store_true',default=False)
    parser.add_argument('--iteration_num',type=int,default=10)
    parser.add_argument('--with_graph',action='store_true',default=False)
    parser.add_argument('--graph_rate',type=float,default=0.5)
    parser.add_argument('--replace_graph',action='store_true',default=False)
    parser.add_argument('--output_file',type=str,default='none')
    

    
   
    args = parser.parse_args()
    args = args.__dict__
    args['is_unseen'] = True if args['zero_shot_mode'] == 'unseen' else False
    if args['output_file']!='none':args['output_file'] = os.path.join(args['exp_path'],args['output_file'])

    logger = setup_logger(name=args['exp_path'][:],log_file=os.path.join(args['exp_path'],'log.log'))
    args['logger'] = logger
    print(args)

    setup_seed(args['seed'])
    
    if args['model'] == 'cross_encoder':
        b=CrossEncoder_Classifier(args)
        if args['run_stage_1']:
            if args['train']:
                b.train_stage_1()
            #只有test的时候考虑load model，load在valid上表现最好的checkpoint
            if args['test']:
                if args['load_model']:
                    checkpoint_dir = os.path.join(args['exp_path'],'checkpoint')
                    b.load_model_stage_1(checkpoint_dir)
                #b.eval_stage_1(b.queries_valid,epoch=-1)
                b.eval_stage_1(b.queries_test,epoch=-1)
        if args['run_stage_2']:
            b.train_stage_2()
    elif args['model'] == 'biosyn':
        b=Biosyn_Classifier(args)
        if args['train']:
            b.train()
        if args['test']:
            if args['load_model']:
                checkpoint_dir = os.path.join(args['exp_path'],'checkpoint')
                b.load_model(checkpoint_dir)
            #b.eval(b.queries_valid,epoch = -1)
            b.eval(b.queries_test,epoch=-1)
    
    elif args['model']=='gcn':
        b=Graph_Classifier(args)
        
        if args['train']:
            b.train()
        if args['test']:
            if args['load_model']:
                checkpoint_dir = os.path.join(args['exp_path'],'checkpoint')
                b.load_model(checkpoint_dir)
            #b.eval(b.queries_valid,epoch = -1)
            b.eval(b.queries_test,epoch=-1)
            #b.get_hop_count(b.queries_valid)
            

            
        








