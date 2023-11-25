######################
### Import Library ###
######################

# My library
from molgraph.dataset import *
from molgraph.simplemodel import *
from molgraph.hyperparameter_simple import *
from molgraph.testing_simple import *
from molgraph.visualize import *
from molgraph.experiment import *
# General library
import os
import argparse
import numpy as np
# pytorch
import torch
import pytorch_lightning as pl
# optuna
import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_param_importances
import joblib
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.set_default_dtype(torch.float64)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

#####################
### Argument List ###
#####################

####################
### Main Program ###
####################

if __name__ == '__main__':

    parser = ArgumentParser()
    args = parser.getArgument()
    print(args)

    file = args.file
    smiles = args.smiles 
    task = args.task
    splitting = args.splitting 
    splitting_fold = args.fold
    splitting_seed = args.splitting_seed

    # get validated dataset
    datasets = getDataset(file, smiles, task, splitting)
    # compute positive weight for classification
    if args.graphtask == 'classification':
        args.pos_weight = getPosWeight(datasets)
        print('pos_weight:', args.pos_weight)
    # generate dataset splitting
    datasets_splitted = generateDatasetSplitting(file, splitting, splitting_fold, splitting_seed)
    # generate all graph dataset
    datasets_graph = generateGraphDataset(file)
    # generate all reduced graph dataset
    dict_reducedgraph = dict()
    for g in args.reduced:
        if g == 'substructure':
            for i in range(splitting_fold):
                vocab_file = file+'_'+str(i)
                if not os.path.exists('vocab/'+vocab_file+'.txt'):
                    generateVocabTrain(file, splitting_seed, splitting_fold, vocab_len=args.vocab_len)
                dict_reducedgraph[g] = generateReducedGraphDict(file, g, vocab_file=vocab_file)
        else:
            dict_reducedgraph[g] = generateReducedGraphDict(file, g)
        
    hyper = Hyper(args)

    if args.graphtask == 'regression':
        study = optuna.create_study(direction="minimize")
    elif args.graphtask == 'classification':
        study = optuna.create_study(direction="maximize")

    t_start = time.time()
    study.optimize(hyper.objective, n_trials=20, timeout=75600)
    len(study.get_trials())
    print("Time: {:.3f}s".format(time.time() - t_start))

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        if key == 'channels':
            print("--in_channels {}".format(value))
            print("--hidden_channels {}".format(value))
            print("--out_channels {}".format(value))
        else:
            print("--{} {}".format(key, value))
    
    with open('dataset/{}/hyperparams_full.txt'.format(hyper.log_folder_name), 'w') as f:
        f.write("-f {} \\".format(args.file))
        f.write('\n')
        f.write("-m {} \\".format(args.model))
        f.write('\n')
        f.write("--schema {} \\".format(args.schema))
        f.write('\n')
        f.write("--reduced {} \\".format(" ".join(args.reduced)))
        f.write('\n')
        f.write("--vocab_len {} \\".format(str(args.vocab_len)))
        f.write('\n')
        f.write("--mol_embedding {} \\".format(str(args.mol_embedding)))
        f.write('\n')
        f.write("--batch_normalize \\")
        f.write('\n')
        f.write("--fold {} \\".format(str(args.fold)))
        f.write('\n')
        f.write("--seed {} \\".format(str(args.seed)))
        f.write('\n')
        for key, value in trial.params.items():
            if key == 'channels':
                f.write("--in_channels {} \\".format(value))
                f.write('\n')
                f.write("--hidden_channels {} \\".format(value))
                f.write('\n')
                f.write("--out_channels {} \\".format(value))
            else:
                f.write("--{} {} \\".format(key, value))
            f.write('\n')

    print(optuna.importance.get_param_importances(study))
    print('COMPLETED!')