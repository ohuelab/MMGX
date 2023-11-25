######################
### Import Library ###
######################

# my library
from molgraph.graphmodel import *
# from molgraph.graphmodel_star import *
# from molgraph.graphmodel_signed import *
from molgraph.dataset import *
from molgraph.experiment import *
# general
import math as math
from tqdm.notebook import tqdm
from tqdm import tqdm, trange
# sklearn
import sklearn.metrics as metrics
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
# pytorch
from torch.autograd import Variable
from transformers.optimization import get_cosine_schedule_with_warmup
# optuna
import optuna
from optuna.trial import TrialState

#####################
### Trainer Class ###
#####################

# Hyper
class Hyper(object):

    def __init__(self, args):
        super(Hyper, self).__init__()

        # experiment
        self.args = args
        self.seed = set_seed(self.args.seed)
        self.args.experiment_number = self.args.experiment_number
        self.log_folder_name, self.exp_name = set_experiment_name(self.args, hyper=True)
        self.device = set_device(self.args) 
        torch.manual_seed(self.seed)

        # dataset
        self.file = self.args.file
        

    def objective(self, trial):

        if self.args.graphtask == 'regression':
            print('Trial:', trial.number)
            # hyperparameter
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
            self.args.batch_size = batch_size

            num_layers = trial.suggest_int("num_layers", 2, 4)
            self.args.num_layers = num_layers

            if len(self.args.reduced) != 0:
                num_layers_reduced = trial.suggest_int("num_layers_reduced", 2, 4)
                self.args.num_layers_reduced = num_layers_reduced
            
            channels = trial.suggest_categorical("channels", [32, 64, 128, 256])
            self.args.in_channels = channels
            self.args.hidden_channels = channels
            self.args.out_channels = channels
            
            # edge_dim = trial.suggest_categorical("edge_dim", [16, 32, 64])
            # self.args.edge_dim = edge_dim

            num_layers_self = trial.suggest_int("num_layers_self", 2, 4)
            self.args.num_layers_self = num_layers_self

            if len(self.args.reduced) != 0:
                num_layers_self_reduced = trial.suggest_int("num_layers_self_reduced", 2, 4)
                self.args.num_layers_self_reduced = num_layers_self_reduced
            
            # dropout = trial.suggest_categorical("dropout", [0.25, 0.35, 0.45])
            # self.args.dropout = dropout
            
            # lr = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3])
            # self.args.lr = lr
            
            # weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3])
            # self.args.weight_decay = weight_decay

            print(trial.params.items())

            ### Train
            overall_results = {
                'val_rmse': [],
                'test_rmse': [],
                'test_r2': []
            }

            fold_iter = tqdm(range(0, self.args.fold), desc='Training')

            for fold_number in fold_iter:

                loss_fn = F.mse_loss
                        
                ### Set logger, loss and accuracy for each fold
                logger = set_logger_fold_trial(self.log_folder_name, self.exp_name, self.seed, fold_number, trial.number)

                patience = 0
                best_loss_epoch = 0
                best_rmse_epoch = 0
                best_loss = 1e9
                best_loss_rmse = 1e9 
                best_rmse = 1e9 
                best_rmse_loss = 1e9

                train_loader, val_loader, test_loader, datasets_train, datasets_val, datasets_test =  generateDataLoader(self.file, self.args.batch_size, self.seed, fold_number)

                # Load model and optimizer
                self.model = load_model(self.args).to(self.device)
                self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
                
                if self.args.lr_schedule:
                    self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 
                                                                     self.args.patience * len(train_loader), 
                                                                     self.args.num_epochs * len(train_loader))

                t_start = time.perf_counter()
                ### K-Fold Training
                for epoch in trange(0, (self.args.num_epochs), desc = '[Epoch]', position = 1):

                    self.model.train()
                    total_loss = 0

                    for _, data in enumerate(train_loader):

                        data = data.to(self.device)
                        out = self.model(data)
                        loss = loss_fn(out, data.y)
                        # Before the backward pass, use the optimizer object to zero all of the
                        # gradients for the variables it will update (which are the learnable
                        # weights of the model)
                        self.optimizer.zero_grad()
                        # Backward pass: compute gradient of the loss with respect to model
                        loss.backward()
                        # keep the gradients within a specific range.
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
                        # Calling the step function on an Optimizer makes an update to its
                        self.optimizer.step()

                        total_loss += loss.item() * getNumberofSmiles(data)

                        if self.args.lr_schedule:
                            self.scheduler.step()

                    total_loss = total_loss / len(train_loader.dataset)

                    ### Validation
                    val_rsme, val_loss, val_r2 = self.eval_regression(val_loader, loss_fn)
                    
                    if val_loss < best_loss:
                        best_loss_rmse = val_rsme
                        best_loss = val_loss
                        best_loss_epoch = epoch

                    if val_rsme < best_rmse:
                        best_rmse = val_rsme
                        best_rmse_loss = val_loss
                        best_rmse_epoch = epoch
                        torch.save(self.model.state_dict(), 
                            f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                            f'fold-{fold_number}_seed-{self.seed}_best-model.pth')
                        patience = 0
                    else:
                        patience += 1

                    ### Validation log
                    logger.log(f"[Val: Fold {fold_number}-Epoch {epoch}] "
                               f"TrainLoss: {total_loss:.4f}, ValLoss: {val_loss:.4f}, ValRMSE: {val_rsme:.4f}, ValR2: {val_r2:.4f} //"
                               f"[Val: Fold {fold_number}-Epoch {epoch}] "
                               f"Best Loss> Loss: {best_loss:.4f}, RMSE: {best_loss_rmse:.4f}, at Epoch: {best_loss_epoch} / "
                               f"Best RMSE> Loss: {best_rmse_loss:.4f}, RMSE: {best_rmse:.4f}, at Epoch: {best_rmse_epoch}")

                    fold_iter.set_description(f'[Fold {fold_number}]-Epoch: {epoch} TrainLoss: {total_loss:.4f} '
                                              f'ValLoss: {val_loss:.4f} ValRMSE: {val_rsme:.4f} patience: {patience}')
                    
                    fold_iter.refresh()
                    if patience > self.args.patience: break

                t_end = time.perf_counter()

                ### Test log
                checkpoint = torch.load(f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                                        f'fold-{fold_number}_seed-{self.seed}_best-model.pth')
                self.model.load_state_dict(checkpoint)
                
                os.remove(f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                          f'fold-{fold_number}_seed-{self.seed}_best-model.pth')
                
                test_rmse, test_loss, test_r2 = self.eval_regression(test_loader, loss_fn)
                
                logger.log(f"[Test: Fold {fold_number}] "
                           f"Best Loss> Loss: {best_loss:4f}, RMSE: {best_loss_rmse:4f}, at Epoch: {best_loss_epoch} /"
                           f"Best RMSE> Loss: {best_rmse_loss:4f}, RMSE: {best_rmse:4f}, at Epoch: {best_rmse_epoch} //"
                           f"[Test: Fold {fold_number}] Test> RMSE: {test_rmse:4f}, R2: {test_r2:4f}, with Time: {t_end-t_start:.2f}")

                test_result_file = "./dataset/{}/results/{}-results.txt".format(self.log_folder_name, self.exp_name)
                with open(test_result_file, 'a+') as f:
                    f.write(f"[FOLD {fold_number}] {self.seed}: BEST Loss: {best_loss:.4f}, BEST RMSE: {best_rmse:.4f} //"
                            f"Test> Loss: {test_loss:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:4f}\n")

                ### Report results
                overall_results['val_rmse'].append(best_rmse)
                overall_results['test_rmse'].append(test_rmse)
                overall_results['test_r2'].append(test_r2)

                final_result_file = f"./dataset/{self.log_folder_name}/results/{self.exp_name}-final.txt"
                with open(final_result_file, 'a+') as f:
                    f.write(f"{self.seed}: ValRMSE_Mean: {np.array(overall_results['val_rmse']).mean():.4f}, "
                            f"ValRMSE_Std: {np.array(overall_results['val_rmse']).std():.4f}, " 
                            f"TestRMSE_Mean: {np.array(overall_results['test_rmse']).mean():.4f}, "
                            f"TestRMSE_Std: {np.array(overall_results['test_rmse']).std():.4f}, " 
                            f"TestR2_Mean: {np.array(overall_results['test_r2']).mean():.4f}, "
                            f"TestR2_Std: {np.array(overall_results['test_r2']).std():.4f}\n")

                print('- ValRMSE ', str(np.array(overall_results['val_rmse']).mean()), '+/-', str(np.array(overall_results['val_rmse']).std()))
                print('- TestRMSE', str(np.array(overall_results['test_rmse']).mean()), '+/-', str(np.array(overall_results['test_rmse']).std()))

                # hyperparameter
                trial.report(np.array(overall_results['val_rmse']).mean(), fold_number)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            return np.array(overall_results['val_rmse']).mean()

        elif self.args.graphtask == 'classification':
            print('Trial:', trial.number)
            # hyperparameter
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
            self.args.batch_size = batch_size

            num_layers = trial.suggest_int("num_layers", 2, 4)
            self.args.num_layers = num_layers
            
            if len(self.args.reduced) != 0:
                num_layers_reduced = trial.suggest_int("num_layers_reduced", 2, 4)
                self.args.num_layers_reduced = num_layers_reduced
            
            channels = trial.suggest_categorical("channels", [32, 64, 128, 256])
            self.args.in_channels = channels
            self.args.hidden_channels = channels
            self.args.out_channels = channels
            
            # edge_dim = trial.suggest_categorical("edge_dim", [16, 32, 64])
            # self.args.edge_dim = edge_dim

            num_layers_self = trial.suggest_int("num_layers_self", 2, 4)
            self.args.num_layers_self = num_layers_self

            if len(self.args.reduced) != 0:
                num_layers_self_reduced = trial.suggest_int("num_layers_self_reduced", 2, 4)
                self.args.num_layers_self_reduced = num_layers_self_reduced
            
            # dropout = trial.suggest_categorical("dropout", [0.25, 0.35, 0.45])
            # self.args.dropout = dropout
            
            # lr = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3])
            # self.args.lr = lr
            
            # weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3])
            # self.args.weight_decay = weight_decay

            print(trial.params.items())

            ### Train
            overall_results = {
                'val_auc': [],
                'val_acc': [],
                'test_auc': [],
                'test_acc': []
            }

            fold_iter = tqdm(range(0, self.args.fold), desc='Training')

            for fold_number in fold_iter:

                loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([float(self.args.pos_weight)]).type(torch.DoubleTensor)).to(self.device)
                
                ### Set logger, loss and accuracy for each fold
                logger = set_logger_fold_trial(self.log_folder_name, self.exp_name, self.seed, fold_number, trial.number)

                patience = 0
                best_loss_epoch = 0
                best_auc_epoch = 0
                best_loss = 1e9
                best_loss_auc = -1e9
                best_auc = -1e9
                best_auc_loss = 1e9

                train_loader, val_loader, test_loader, datasets_train, datasets_val, datasets_test =  generateDataLoader(self.file, self.args.batch_size, self.seed, fold_number)

                # Load model and optimizer
                self.model = load_model(self.args).to(self.device)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
                
                if self.args.lr_schedule:
                    self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 
                                                                     self.args.patience * len(train_loader), 
                                                                     self.args.num_epochs * len(train_loader))

                t_start = time.perf_counter()
                ### K-Fold Training
                for epoch in trange(0, (self.args.num_epochs), desc = '[Epoch]', position = 1):

                    self.model.train()
                    total_loss = 0

                    for _, data in enumerate(train_loader):

                        data = data.to(self.device)
                        out = self.model(data)
                        loss = loss_fn(out, data.y)
                        # Before the backward pass, use the optimizer object to zero all of the
                        # gradients for the variables it will update (which are the learnable
                        # weights of the model)
                        self.optimizer.zero_grad()
                        # Backward pass: compute gradient of the loss with respect to model
                        loss.backward()
                        # keep the gradients within a specific range.
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
                        # Calling the step function on an Optimizer makes an update to its
                        self.optimizer.step()

                        total_loss += loss.item() * getNumberofSmiles(data)

                        if self.args.lr_schedule:
                            self.scheduler.step()

                    total_loss = total_loss / len(train_loader.dataset)

                    ### Validation
                    val_acc, val_loss, val_auc = self.eval_classification(val_loader, loss_fn)
                    
                    if val_loss < best_loss:
                        best_loss_auc = val_auc
                        best_loss = val_loss
                        best_loss_epoch = epoch

                    if val_auc > best_auc:
                        best_auc = val_auc
                        best_auc_loss = val_loss
                        best_auc_epoch = epoch
                        torch.save(self.model.state_dict(), 
                            f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                            f'fold-{fold_number}_seed-{self.seed}_best-model.pth')
                        patience = 0
                    else:
                        patience += 1

                    ### Validation log
                    logger.log(f"[Val: Fold {fold_number}-Epoch {epoch}] "
                               f"TrainLoss: {total_loss:.4f}, ValLoss: {val_loss:.4f}, ValAUC: {val_auc:.4f}, ValACC: {val_acc:.4f} //"
                               f"[Val: Fold {fold_number}-Epoch {epoch}] "
                               f"Best Loss> Loss: {best_loss:.4f}, AUC: {best_loss_auc:.4f}, at Epoch: {best_loss_epoch} / "
                               f"Best AUC> Loss: {best_auc_loss:.4f}, AUC: {best_auc:.4f}, at Epoch: {best_auc_epoch}")

                    fold_iter.set_description(f'[Fold {fold_number}]-Epoch: {epoch} TrainLoss: {total_loss:.4f} '
                                              f'ValLoss: {val_loss:.4f} ValAUC: {val_auc:.4f} patience: {patience}')

                    fold_iter.refresh()
                    if patience > self.args.patience: break

                t_end = time.perf_counter()

                ### Test log
                checkpoint = torch.load(f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                                        f'fold-{fold_number}_seed-{self.seed}_best-model.pth')
                self.model.load_state_dict(checkpoint)

                os.remove(f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                          f'fold-{fold_number}_seed-{self.seed}_best-model.pth')
                
                test_acc, test_loss, test_auc = self.eval_classification(test_loader, loss_fn)
                
                logger.log(f"[Test: Fold {fold_number}] "
                           f"Best Loss> Loss: {best_loss:4f}, AUC: {best_loss_auc:4f}, at Epoch: {best_loss_epoch} /"
                           f"Best AUC> Loss: {best_auc_loss:4f}, AUC: {best_auc:4f}, at Epoch: {best_auc_epoch} //"
                           f"[Test: Fold {fold_number}] Test> AUC: {test_auc:4f}, ACC: {test_acc:4f}, with Time: {t_end-t_start:.2f}")

                test_result_file = "./dataset/{}/results/{}-results.txt".format(self.log_folder_name, self.exp_name)
                with open(test_result_file, 'a+') as f:
                    f.write(f"[FOLD {fold_number}] {self.seed}: BEST Loss: {best_loss:.4f}, BEST AUC: {best_auc:.4f} //"
                            f"Test> Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, ACC: {test_acc:4f}\n")

                ### Report results
                overall_results['val_auc'].append(best_auc)
                overall_results['test_auc'].append(test_auc)
                overall_results['test_acc'].append(test_acc)

                final_result_file = f"./dataset/{self.log_folder_name}/results/{self.exp_name}-final.txt"
                with open(final_result_file, 'a+') as f:
                    f.write(f"{self.seed}: ValAUC_Mean: {np.array(overall_results['val_auc']).mean():.4f}, "
                            f"ValAUC_Std: {np.array(overall_results['val_auc']).std():.4f}, " 
                            f"TestAUC_Mean: {np.array(overall_results['test_auc']).mean():.4f}, "
                            f"TestAUC_Std: {np.array(overall_results['test_auc']).std():.4f}, " 
                            f"TestACC_Mean: {np.array(overall_results['test_acc']).mean():.4f}, "
                            f"TestACC_Std: {np.array(overall_results['test_acc']).std():.4f}\n")

                print('- ValAUC ', str(np.array(overall_results['val_auc']).mean()), '+/-', str(np.array(overall_results['val_auc']).std()))
                print('- TestAUC', str(np.array(overall_results['test_auc']).mean()), '+/-', str(np.array(overall_results['test_auc']).std()))

                # hyperparameter
                trial.report(np.array(overall_results['val_auc']).mean(), fold_number)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return np.array(overall_results['val_auc']).mean()
        
        elif self.args.graphtask == 'multiclass':

            ### Train
            overall_results = {
                'val_kap': [],
                'val_acc': [],
                'test_kap': [],
                'test_acc': []
            }

            fold_iter = tqdm(range(0, self.args.fold), desc='Training')

            for fold_number in fold_iter:

                class_weights=torch.tensor(np.array([11.54842398,  8.31405056,  0.35802132]),dtype=torch.float).to(self.device)
                loss_fn = nn.CrossEntropyLoss(class_weights, reduction='mean')
                # loss_fn = FocalLoss(alpha=class_weights)
                # loss_fn = nn.CrossEntropyLoss()
                # m_fn = nn.Softmax(dim=1)
                
                ### Set logger, loss and accuracy for each fold
                logger = set_logger_fold_trial(self.log_folder_name, self.exp_name, self.seed, fold_number, trial.number)

                patience = 0
                best_loss_epoch = 0
                best_kap_epoch = 0
                best_loss = 1e9
                best_loss_kap = -1e9
                best_kap = -1e9
                best_kap_loss = 1e9

                train_loader, val_loader, test_loader, datasets_train, datasets_val, datasets_test =  generateDataLoader(self.file, self.args.batch_size, self.seed, fold_number)

                # Load model and optimizer
                self.model = load_model(self.args).to(self.device)
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
                
                if self.args.lr_schedule:
                    self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 
                                                                     self.args.patience * len(train_loader), 
                                                                     self.args.num_epochs * len(train_loader))
                
                t_start = time.perf_counter()
                ### K-Fold Training
                for epoch in trange(0, (self.args.num_epochs), desc = '[Epoch]', position = 1):

                    self.model.train()
                    total_loss = 0

                    for _, data in enumerate(train_loader):

                        data = data.to(self.device)
                        out = self.model(data)
                        # data_y = data.y.squeeze().type(torch.LongTensor).to(self.device)
                        data_y = F.one_hot(data.y.squeeze().to(torch.int64), num_classes=self.args.class_number).type(torch.DoubleTensor).to(self.device)
                        # loss = loss_fn(m_fn(out), data_y)
                        loss = loss_fn(out, data_y)
                        # Before the backward pass, use the optimizer object to zero all of the
                        # gradients for the variables it will update (which are the learnable
                        # weights of the model)
                        self.optimizer.zero_grad()
                        # Backward pass: compute gradient of the loss with respect to model
                        loss.backward()
                        # keep the gradients within a specific range.
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
                        # Calling the step function on an Optimizer makes an update to its
                        self.optimizer.step()

                        total_loss += loss.item() * getNumberofSmiles(data)
                        # total_loss += loss.item()

                        if self.args.lr_schedule:
                            self.scheduler.step()

                    total_loss = total_loss / len(train_loader.dataset)
                    # total_loss = total_loss / len(train_loader)

                    ### Validation
                    val_kap, val_loss = self.eval_multiclass(val_loader, loss_fn)
                    
                    if val_loss < best_loss:
                        best_loss_kap = val_kap
                        best_loss = val_loss
                        best_loss_epoch = epoch

                    if val_kap > best_kap:
                        best_kap = val_kap
                        best_kap_loss = val_loss
                        best_kap_epoch = epoch
                        torch.save(self.model.state_dict(), 
                            f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                            f'fold-{fold_number}_seed-{self.seed}_best-model.pth')
                        patience = 0
                    else:
                        patience += 1

                    ### Validation log
                    logger.log(f"[Val: Fold {fold_number}-Epoch {epoch}] "
                               f"TrainLoss: {total_loss:.4f}, ValLoss: {val_loss:.4f}, ValKAP: {val_kap:.4f} //"
                               f"[Val: Fold {fold_number}-Epoch {epoch}] "
                               f"Best Loss> Loss: {best_loss:.4f}, KAPP: {best_loss_kap:.4f}, at Epoch: {best_loss_epoch} / "
                               f"Best KAPPA> Loss: {best_kap_loss:.4f}, KAPPA: {best_kap:.4f}, at Epoch: {best_kap_epoch}")

                    fold_iter.set_description(f'[Fold {fold_number}]-Epoch: {epoch} TrainLoss: {total_loss:.4f} '
                                              f'ValLoss: {val_loss:.4f} ValKAP: {val_kap:.4f} patience: {patience}')
                    fold_iter.refresh()
                    if patience > self.args.patience: break

                t_end = time.perf_counter()

                ### Test log
                checkpoint = torch.load(f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                                        f'fold-{fold_number}_seed-{self.seed}_best-model.pth')
                self.model.load_state_dict(checkpoint)
                
                test_kap, test_loss = self.eval_multiclass(test_loader, loss_fn)
                
                logger.log(f"[Test: Fold {fold_number}] "
                           f"Best Loss> Loss: {best_loss:4f}, ValKAP: {val_kap:.4f}, at Epoch: {best_loss_epoch} /"
                           f"Best KAPPA> Loss: {best_kap_loss:4f}, KAPPA: {best_kap:4f}, at Epoch: {best_kap_epoch} //"
                           f"[Test: Fold {fold_number}] Test> KAP: {test_kap:4f}, with Time: {t_end-t_start:.2f}")

                test_result_file = "./dataset/{}/results/{}-results.txt".format(self.log_folder_name, self.exp_name)
                with open(test_result_file, 'a+') as f:
                    f.write(f"[FOLD {fold_number}] {self.seed}: BEST Loss: {best_loss:.4f}, BEST KAPPA: {best_kap:.4f} //"
                            f"Test> Loss: {test_loss:.4f}, KAP: {test_kap:4f}\n")

                ### Report results
                overall_results['val_kap'].append(best_kap)
                overall_results['test_kap'].append(test_kap)

                final_result_file = f"./dataset/{self.log_folder_name}/results/{self.exp_name}-final.txt"
                with open(final_result_file, 'a+') as f:
                    f.write(f"{self.seed}: ValKAP_Mean: {np.array(overall_results['val_kap']).mean():.4f}, "
                            f"ValKAP_Std: {np.array(overall_results['val_kap']).std():.4f}, " 
                            f"TestKAP_Mean: {np.array(overall_results['test_kap']).mean():.4f}, "
                            f"TestKAP_Std: {np.array(overall_results['test_kap']).std():.4f}\n")

                print('TestKAP', str(np.array(overall_results['test_kap']).mean()), '+/-', str(np.array(overall_results['test_kap']).std()))


    ### Evaluate
    def eval_regression(self, loader, loss_fn):

        self.model.eval()
        with torch.no_grad():
            rmse = 0.
            loss = 0.

            y_test = list()
            y_pred = list()

            for data in loader:
                data = data.to(self.device)
                out = self.model(data)
                loss += loss_fn(out, data.y).item() * getNumberofSmiles(data)
                # y_test.extend((data.y.cpu()).detach().numpy())
                # y_pred.extend((out.cpu()).detach().numpy())
                y_test.append((data.y).detach())
                y_pred.append((out).detach())

            y_test = torch.squeeze(torch.cat(y_test, dim=0))
            y_pred = torch.squeeze(torch.cat(y_pred, dim=0))
            
            rsme_final = math.sqrt(loss / len(loader.dataset))
            loss_final = loss / len(loader.dataset)
            # r2score = R2Score()
            # r2_final = r2score(y_pred, y_test)
            # print('TORCH:', r2_final)
            r2_final = r2_score(y_test.cpu(), y_pred.cpu())
            # print('SCIKIT:', r2_final)
            # print(rsme_final, '?==?', math.sqrt(mean_squared_error(y_test, y_pred)))
            # print(loss_final, '?==?', F.mse_loss(torch.Tensor(y_pred),torch.Tensor(y_test)).item())
            # print(r2_final)
        return rsme_final, loss_final, r2_final

    ### Evaluate
    def eval_classification(self, loader, loss_fn):
        
        self.model.eval()
        with torch.no_grad():
            m_fn = nn.Sigmoid()

            correct = 0.
            loss = 0.
            
            y_test = list()
            y_pred = list()

            for data in loader:
                data = data.to(self.device)
                out = self.model(data)
                # pred = out.max(dim=1)[1]
                pred = m_fn(out)
                pred_round = pred > 0.5
                correct += pred_round.eq(data.y).sum().item()
                loss += loss_fn(out, data.y).item() * getNumberofSmiles(data)
                # y_test.extend((data.y.cpu()).detach().numpy())
                # y_pred.extend((pred.cpu()).detach().numpy())
                y_test.append((data.y).detach())
                y_pred.append((pred).detach())

            y_test = torch.squeeze(torch.cat(y_test, dim=0))
            y_pred = torch.squeeze(torch.cat(y_pred, dim=0))
            
            acc_final = correct / len(loader.dataset)
            loss_final = loss / len(loader.dataset)
            # y_pred = torch.Tensor(np.array(y_pred))
            # y_test = torch.Tensor(y_test).type(torch.IntTensor)
            # auroc = AUROC(num_classes=args.class_number)
            # auc_final = auroc(y_pred, y_test)
            # roc = ROC(pos_label=1)
            # fpr, tpr, threshold = roc(y_pred, y_test)
            # auc_final = auc(fpr, tpr).item()
            # print('TORCH:', auc_final)
            fpr, tpr, threshold = metrics.roc_curve(y_test.cpu(), y_pred.cpu())
            auc_final = metrics.auc(fpr, tpr)
            # print('SCIKIT:', auc_final)
        return acc_final, loss_final, auc_final

    ### Evaluate
    def eval_multiclass(self, loader, loss_fn):
        
        self.model.eval()
        with torch.no_grad():

            # correct = 0.
            loss = 0.
            
            y_test = list()
            y_pred = list()

            for data in loader:

                data = data.to(self.device)
                out = self.model(data)
                # pred = out.max(dim=1)[1]
                # pred = m_fn(out).cpu()
                pred = out.to(self.device)
                # pred_round = pred > 0.5
                # pred_hat = torch.max(pred, 1).indices
                pred_hat = pred.argmax(dim=1)
                # correct += pred_round.eq(data.y.cpu()).sum().item()
                # data_y = data.y.squeeze().type(torch.LongTensor).to(self.device)
                data_y = F.one_hot(data.y.squeeze().to(torch.int64), num_classes=self.args.class_number).type(torch.DoubleTensor).to(self.device)
                # loss += loss_fn(m_fn(out), data_y.cpu()).item()
                loss += loss_fn(out.to(self.device), data_y.to(self.device)).item() * getNumberofSmiles(data)
                # y_test.extend((torch.squeeze(data.y).cpu()).detach().numpy())
                # y_pred.extend((pred_hat.cpu()).detach().numpy())
                y_test.append((torch.squeeze(data.y)).detach())
                y_pred.append((pred_hat).detach())

            y_test = torch.squeeze(torch.cat(y_test, dim=0))
            y_pred = torch.squeeze(torch.cat(y_pred, dim=0))
            
            # acc_final = correct / len(loader.dataset)
            # loss_final = loss / len(loader.dataset)
            loss_final = loss / len(loader)
            # y_pred = torch.Tensor(np.array(y_pred))
            # y_test = torch.Tensor(y_test).type(torch.IntTensor)
            # auroc = AUROC(num_classes=args.class_number)
            # auc_final = auroc(y_pred, y_test)
            # fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
            # auc_final = metrics.auc(fpr, tpr)
            kappa_final = cohen_kappa_score(y_test.cpu(), y_pred.cpu(), weights='quadratic')
            # return kappa_final, loss_final, auc_final
        return kappa_final, loss_final
