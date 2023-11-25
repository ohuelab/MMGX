######################
### Import Library ###
######################

# general library
import os as os
import numpy as np
import pandas as pd
import random as random
import time as time
import argparse
# pytorch
import torch

###########################
### Experiment Function ###
###########################

# parser
class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        ### Experiment 
        self.parser.add_argument('--seed', type=int, default=42, help='seed')
        self.parser.add_argument('--experiment_number', default='001', type=str)
        self.parser.add_argument('-m', '--model', choices=['GAT', 'GIN', 'GAT_edge', 'Benchmark_GCN', 'Benchmark_GIN', 'Benchmark_AttentiveFP', 'ECFP4', 'Descriptor'], default='GAT', type=str)

        ### Dataset
        self.parser.add_argument('-f', '--file', default='freesolv', type=str, help='dataset type')
        self.parser.add_argument("--schema", default='AR', choices=['A', 'R', 'R_0', 'R_N', 'AR', 'AR_0', 'AR_N'], type=str)
        self.parser.add_argument("--reduced", nargs='*', default='', choices=['junctiontree', 'cluster', 'functional', 'pharmacophore', 'substructure'], type=str)
        self.parser.add_argument('--fold', default=1, type=int, help='fold')
        self.parser.add_argument('--vocab_len', default=100, type=int, help='vocab_len')

        ### Model
        self.parser.add_argument('--num_layers', default=3, type=int, help='graph attention layers')
        self.parser.add_argument('--num_layers_self', default=3, type=int, help='self-attention layers')
        self.parser.add_argument('--num_layers_reduced', default=3, type=int, help='graph attention layers')
        self.parser.add_argument('--num_layers_self_reduced', default=3, type=int, help='self-attention layers')
        self.parser.add_argument('--heads', default=1, type=int, help='attention heads')
        self.parser.add_argument('--in_channels', type=int, default=256, help='graph in size')
        self.parser.add_argument('--hidden_channels', type=int, default=256, help='hidden size')
        self.parser.add_argument('--out_channels', type=int, default=256, help='graph out size')
        self.parser.add_argument('--edge_dim', type=int, default=32, help='edge dim size')
        self.parser.add_argument('--mol_embedding', type=int, default=256, help='mol embedding')

        ### Training
        self.parser.add_argument('--batch_size', default=256, type=int, help='train batch size')
        self.parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=0.00005, help='weight decay')
        self.parser.add_argument("--grad_norm", type=float, default=1.0)
        self.parser.add_argument("--dropout", type=float, default=0.4)
        self.parser.add_argument('--num_epochs', default=300, type=int, help='train epochs number')
        self.parser.add_argument('--patience', type=int, default=30, help='patience for earlystopping')
        self.parser.add_argument("--lr_schedule", default=True, action='store_true')
        self.parser.add_argument("--batch_normalize", default=False, action='store_true')

    def getParser(self):
        return self.parser

    def getArgument(self, arguments=None):
        if arguments is not None:
            args = self.parser.parse_args(arguments)
        else:
            args = self.parser.parse_args()

        # set seed
        set_seed(args.seed)
        
        # dataset setup
        dataset_df = pd.read_csv('./dataset/_dataset.csv')
        dataset_dict = dict()
        for i, row in dataset_df.iterrows():
            dataset_dict[row['file']] = {'smiles': row['smiles'],
                                         'task': row['task'],
                                         'splitting': row['splitting'],
                                         'graphtask': row['graphtask'],
                                         'class_number': row['class_number']}
        file = args.file
        args.smiles = dataset_dict[file]['smiles']
        args.task = dataset_dict[file]['task']
        args.splitting = dataset_dict[file]['splitting']
        args.splitting_seed = args.seed
        args.graphtask = dataset_dict[file]['graphtask']
        args.class_number = dataset_dict[file]['class_number']
        
        # Setting device
        if torch.cuda.is_available():
            args.device = torch.device("cuda:0")  
            args.gpu = 0
        else:
            args.device =  torch.device("cpu")
            args.gpu = -1

        return args


# logging class
class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str):
        if self.lock:
            self.lock.acquire()
        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')
        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()

# set all seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
    torch.set_deterministic_debug_mode("warn")
    torch.set_printoptions(precision=16)
    torch.set_default_dtype(torch.float64)

    return seed

# create checkpoints, result, and log folder  
def set_experiment_name(args, hyper=False):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.localtime())

    reduced_list = '_'.join(args.reduced)
    if hyper:
        log_folder_name = os.path.join(*[args.file, args.model+'_'+args.schema+'_'+reduced_list, 'hyper', f"{ts}"])
    else:
        log_folder_name = os.path.join(*[args.file, args.model+'_'+args.schema+'_'+reduced_list, f"{ts}"])

    if not(os.path.isdir('./dataset/{}/checkpoints'.format(log_folder_name))):
        os.makedirs(os.path.join('./dataset/{}/checkpoints'.format(log_folder_name)))

    if not(os.path.isdir('./dataset/{}/results'.format(log_folder_name))):
        os.makedirs(os.path.join('./dataset/{}/results'.format(log_folder_name)))

    if not(os.path.isdir('./dataset/{}/logs'.format(log_folder_name))):
        os.makedirs(os.path.join('./dataset/{}/logs'.format(log_folder_name)))

    print("Make Directory {} in Logs, Checkpoints and Results Folders".format(log_folder_name))

    exp_name = str()
    exp_name += args.experiment_number

    # Save training arguments for reproduction
    torch.save(args, os.path.join('./dataset/{}/checkpoints'.format(log_folder_name), 'training_args.bin'))

    return log_folder_name, exp_name

# set logger
def set_logger(log_folder_name, exp_name, seed):
    logger = Logger(str(os.path.join(f'./dataset/{log_folder_name}/logs/', 
                    f'exp-{exp_name}_seed-{seed}.log')), mode='a')
    return logger

# set logger fold
def set_logger_fold(log_folder_name, exp_name, seed, fold_number):
    logger = Logger(str(os.path.join(f'./dataset/{log_folder_name}/logs/', 
                    f'exp-{exp_name}_seed-{seed}_fold-{fold_number}.log')), mode='a')
    return logger

# set logger fold and trial
def set_logger_fold_trial(log_folder_name, exp_name, seed, fold_number, trial):
    logger = Logger(str(os.path.join(f'./dataset/{log_folder_name}/logs/', 
                    f'exp-{exp_name}_seed-{seed}_fold-{fold_number}_trial-{trial}.log')), mode='a')
    return logger

# set device
def set_device(args):
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
        
    # print(device)
    return device