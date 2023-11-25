######################
### Import Library ###
######################

# my library
from molgraph.graphmodel import *
# from molgraph.graphmodel_star import *
# from molgraph.graphmodel_signed import *
from molgraph.dataset import *
from molgraph.experiment import *
from molgraph.training import *
from molgraph.visualize import *
# general
import math as math
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from tqdm import tqdm, trange
# sklearn
import sklearn.metrics as metrics
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_recall_curve
# pytorch
from torch.autograd import Variable
from torchmetrics import AUROC


#####################
### Testing Class ###
#####################

# Tester
class Tester(object):

    def __init__(self, args, args_test, print_model=True):
        super(Tester, self).__init__()

        # experiment
        self.args = args
        self.device = set_device(self.args) 
        torch.manual_seed(self.args.seed)
        
        self.log_folder_name = args_test['log_folder_name']
        self.exp_name = args_test['exp_name']
        self.fold_number = args_test['fold_number']
        self.seed = args_test['seed']

        # dataset
        self.checkpoint = torch.load(f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                          f'fold-{self.fold_number}_seed-{self.seed}_best-model.pth', map_location=torch.device(self.device))
        self.model_test = load_model(self.args).to(self.device)
        self.model_test.load_state_dict(self.checkpoint)
        self.model_test.eval()
        if print_model: print(self.model_test)

        self.x_embed = None
        self.y_test = None
        self.att_mol = None

        self.performance_report = None

    def test(self, test_loader, return_attention_weights=False):
        self.model_test.eval()

        with torch.no_grad():
            if self.args.graphtask == 'regression':
                y_test = list()
                y_pred = list()
                x_embed = list()

                loss_fn = F.mse_loss
                
                # rmse = 0
                loss = 0

                for data in test_loader:
                    data = data.to(self.device)
                    if return_attention_weights:
                        out, att_mol = self.model_test(data, return_attention_weights=return_attention_weights)
                        out = out.to(self.device)
                    else:
                        out = self.model_test(data).to(self.device)
                    # rmse += mean_squared_error(out, data.y.cpu()).item()
                    loss += loss_fn(out, data.y, reduction='sum').item()
                    y_test.extend((data.y.cpu()).detach().numpy())
                    y_pred.extend((out.cpu()).detach().numpy())
                    x_embed.extend(self.model_test.last_mol_embedding.detach().cpu().numpy())

                print(len(y_test), len(y_pred))
                print(loss / len(test_loader.dataset), math.sqrt(loss / len(test_loader.dataset)))
                # The root mean squared error: lower is better
                print('Root mean squared error RMSE: %f' % math.sqrt(mean_squared_error(y_test, y_pred)))
                # The mean squared error: lower is better
                print("Mean squared error MSE: %f" % mean_squared_error(y_test, y_pred))
                # The coefficient of determination: 1 is perfect prediction
                print("Coefficient of determination R2: %f" % r2_score(y_test, y_pred))

                # Plot outputs
                fig = plt.figure(figsize=(5, 5), dpi=150)

                max_value = np.ceil(np.max(y_test+y_pred))
                min_value = np.floor(np.min(y_test+y_pred))
                plt.plot([min_value, max_value], [min_value, max_value], 'r')
                plt.scatter(y_test, y_pred, color="black", alpha=0.5, s=10)
                plt.xlabel('true')
                plt.ylabel('predict')
                plt.show()

                y_test_list = [yt[0] for yt in y_test]
                y_pred_list = [yt[0] for yt in y_pred]

                # Start with a square Figure.
                fig = plt.figure(figsize=(6, 6))
                # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
                # the size of the marginal axes and the main axes in both directions.
                # Also adjust the subplot parameters for a square plot.
                gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                                    left=0.1, right=0.9, bottom=0.1, top=0.9,
                                    wspace=0.05, hspace=0.05)
                # Create the Axes.
                ax = fig.add_subplot(gs[1, 0])
                ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
                ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
                # Draw the scatter plot and marginals.
                scatter_hist(y_test_list, y_pred_list, ax, ax_histx, ax_histy)

                y_test = torch.Tensor(np.array(y_test, dtype=float)).type(torch.DoubleTensor)
                self.x_embed = x_embed
                self.y_test = y_test

                self.performance_report = {"RMSE": math.sqrt(mean_squared_error(y_test, y_pred)),
                                            "MSE": mean_squared_error(y_test, y_pred),
                                            "R2": r2_score(y_test, y_pred)}

                # interpretation
                if return_attention_weights:
                    self.att_mol = att_mol

            elif self.args.graphtask == 'classification':
                y_test = list()
                y_pred = list()
                x_embed = list()

                loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([float(self.args.pos_weight)]).type(torch.DoubleTensor)).to(self.device)
                m_fn = nn.Sigmoid()

                correct = 0.
                loss = 0.
                
                y_test = list()
                y_pred = list()

                for data in test_loader:
                    data = data.to(self.device)
                    if return_attention_weights:
                        out, att_mol = self.model_test(data, return_attention_weights=return_attention_weights)
                        out = out.to(self.device)
                    else:
                        out = self.model_test(data).to(self.device)
                    pred = m_fn(out).cpu()
                    pred_round = pred > 0.5
                    correct += pred_round.eq(data.y.cpu()).sum().item()
                    loss += loss_fn(out, data.y).item()
                    y_test.extend((data.y.cpu()).detach().numpy())
                    y_pred.extend((pred.cpu()).detach().numpy())
                    x_embed.extend(self.model_test.last_mol_embedding.detach().cpu().numpy())
                
                # calculate the auc acc from Torch
                # acc_final = correct / len(test_loader.dataset)
                # loss_final = loss / len(test_loader.dataset)
                y_pred = torch.Tensor(np.array(y_pred, dtype=float))
                y_test = torch.Tensor(np.array(y_test, dtype='int64')).type(torch.IntTensor)
                # auroc = AUROC(num_classes=self.args.class_number)
                # auc_final = auroc(y_pred, y_test).item()
                # print('AUC from Torch:', auc_final)
                # print('ACC from Torch:', acc_final)
                # print('loss:', loss_final)

                # calculate the fpr and tpr for all thresholds of the classification
                fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
                roc_auc = metrics.auc(fpr, tpr)
                pred_round = y_pred > 0.5
                acc_score = accuracy_score(y_test, pred_round)
                print('AUC from scikit:', roc_auc)
                print('ACC from scikit:', acc_score)

                # PR-curve and F1-score
                precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred)
                f1_scores = 2*recall*precision/(recall+precision)
                print('Best threshold: ', thresholds_pr[np.argmax(f1_scores)])
                print('Best F1-Score: ', np.nanmax(f1_scores))

                # classification report
                print(classification_report(y_test, pred_round))
                print("""Note that in binary classification, 
                        recall of the positive class is also known as “sensitivity”; 
                        recall of the negative class is “specificity”.""")

                # method I: plt
                fig = plt.figure(figsize=(5, 5), dpi=150)
                plt.title('Receiver Operating Characteristic (ROC)')
                plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
                plt.legend(loc = 'lower right')
                plt.plot([0, 1], [0, 1], 'r--')
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.ylabel('True Positive Rate (TPR)')
                plt.xlabel('False Positive Rate (FPR)')
                plt.show()

                self.x_embed = x_embed
                self.y_test = y_test

                self.performance_report = {"AUC": roc_auc,
                                            "ACC": acc_score,
                                            "F1": np.nanmax(f1_scores)}

                # interpretation
                if return_attention_weights:
                    self.att_mol = att_mol

            elif self.args.graphtask == 'multiclass':
                y_test = list()
                y_pred = list()
                x_embed = list()

                class_weights=torch.tensor(np.array([11.54842398,  8.31405056,  0.35802132]),dtype=torch.float).cpu()
                # class_weights=torch.tensor(np.array([12.0,  8.5,  0.3]),dtype=torch.float).cpu()
                loss_fn = nn.CrossEntropyLoss(class_weights, reduction='mean')
                # loss_fn = FocalLoss(alpha=class_weights)
                # loss_fn = nn.CrossEntropyLoss()
                # m_fn = nn.Softmax(dim=1)

                # correct = 0.
                loss = 0.
                
                y_test = list()
                y_pred = list()

                for data in test_loader:
                    data = data.to(self.device)
                    if return_attention_weights:
                        out, att_mol = self.model_test(data, return_attention_weights=return_attention_weights)
                        out = out.to(self.device)
                    else:
                        out = self.model_test(data).to(self.device)
                    # pred = m_fn(out).cpu()
                    pred = out.cpu()
                    # pred_round = pred > 0.5
                    # pred_hat = torch.max(pred, 1).indices
                    pred_hat = pred.argmax(dim=1)
                    # correct += pred_round.eq(data.y.cpu()).sum().item()
                    # data_y = data.y.squeeze().type(torch.LongTensor).to(self.device)
                    data_y = F.one_hot(data.y.squeeze().to(torch.int64), num_classes=self.args.class_number).type(torch.DoubleTensor).to(self.device)
                    # loss += loss_fn(m_fn(out.cpu()), data_y.cpu()).item()
                    loss += loss_fn(out.cpu(), data_y.cpu()).item()
                    y_test.extend((torch.squeeze(data.y).cpu()).detach().numpy())
                    y_pred.extend((pred_hat.cpu()).detach().numpy())
                    x_embed.extend(self.model_test.last_mol_embedding.detach().cpu().numpy())
                
                # calculate the auc acc from Torch
                # acc_final = correct / len(test_loader.dataset)
                # loss_final = loss / len(test_loader.dataset)
                y_pred = torch.Tensor(np.array(y_pred, dtype='int64')).type(torch.IntTensor)
                y_test = torch.Tensor(np.array(y_test, dtype='int64')).type(torch.IntTensor)
                # auroc = AUROC(num_classes=self.args.class_number)
                # auc_final = auroc(y_pred, y_test).item()
                # print('AUC from Torch:', auc_final)
                # print('ACC from Torch:', acc_final)
                # print('loss:', loss_final)

                # calculate the fpr and tpr for all thresholds of the classification
                # fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
                # roc_auc = metrics.auc(fpr, tpr)
                # pred_round = y_pred > 0.5
                # acc_score = accuracy_score(y_test, pred_round)
                kappa_score = cohen_kappa_score(y_test, y_pred, weights='quadratic')
                # print('AUC from scikit:', roc_auc)
                print('KAPPA from function:', kappa_score)

                # classification report
                print(classification_report(y_test, y_pred))
                print("""Note that in binary classification, 
                        recall of the positive class is also known as “sensitivity”; 
                        recall of the negative class is “specificity”.""")

                # method I: plt
                # fig = plt.figure(figsize=(5, 5), dpi=150)
                # plt.title('Receiver Operating Characteristic (ROC)')
                # plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
                # plt.legend(loc = 'lower right')
                # plt.plot([0, 1], [0, 1], 'r--')
                # plt.xlim([0, 1])
                # plt.ylim([0, 1])
                # plt.ylabel('True Positive Rate (TPR)')
                # plt.xlabel('False Positive Rate (FPR)')
                # plt.show()

                self.x_embed = x_embed
                self.y_test = y_test

                # interpretation
                if return_attention_weights:
                    self.att_mol = att_mol

    def test_single(self, test_loader, return_attention_weights=False, print_result=True, raw_prediction=False):
        self.model_test.eval()

        with torch.no_grad():
            if self.args.graphtask == 'regression':
                y_test = list()
                y_pred = list()
                x_embed = list()

                loss_fn = F.mse_loss
                
                # rmse = 0
                loss = 0

                for data in test_loader:
                    data = data.to(self.device)
                    if return_attention_weights:
                        out, att_mol = self.model_test(data, return_attention_weights=return_attention_weights)
                        out = out.to(self.device)
                    else:
                        out = self.model_test(data).to(self.device)
                    # rmse += mean_squared_error(out, data.y.cpu()).item()
                    loss += loss_fn(out, data.y, reduction='sum').item()
                    y_test.extend((data.y.cpu()).detach().numpy())
                    y_pred.extend((out.cpu()).detach().numpy())
                    x_embed.extend(self.model_test.last_mol_embedding.detach().cpu().numpy())

                if print_result:
                    print('y_test:', y_test)
                    print('y_pred:', y_pred)

                y_test = torch.Tensor(np.array(y_test, dtype=float)).type(torch.DoubleTensor)
                self.x_embed = x_embed
                self.y_test = y_test

                # interpretation
                if return_attention_weights:
                    self.att_mol = att_mol

                return y_pred

            elif self.args.graphtask == 'classification':
                y_test = list()
                y_pred = list()
                x_embed = list()

                loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([float(self.args.pos_weight)]).type(torch.DoubleTensor)).to(self.device)
                m_fn = nn.Sigmoid()

                correct = 0.
                loss = 0.
                
                y_test = list()
                y_pred = list()

                for data in test_loader:
                    data = data.to(self.device)
                    if return_attention_weights:
                        out, att_mol = self.model_test(data, return_attention_weights=return_attention_weights)
                        out = out.to(self.device)
                    else:
                        out = self.model_test(data).to(self.device)
                    pred = m_fn(out).cpu()
                    pred_round = pred > 0.5
                    correct += pred_round.eq(data.y.cpu()).sum().item()
                    loss += loss_fn(out, data.y).item()
                    y_test.extend((data.y.cpu()).detach().numpy())
                    y_pred.extend((pred.cpu()).detach().numpy())
                    x_embed.extend(self.model_test.last_mol_embedding.detach().cpu().numpy())
                
                y_pred = torch.Tensor(np.array(y_pred, dtype=float))
                y_test = torch.Tensor(np.array(y_test, dtype='int64')).type(torch.IntTensor)
                
                if print_result:
                    print('y_test:', y_test)
                    print('y_pred:', y_pred)
                    print('y_pred > 0.5:', y_pred > 0.5)

                self.x_embed = x_embed
                self.y_test = y_test

                # interpretation
                if return_attention_weights:
                    self.att_mol = att_mol

                if raw_prediction:
                    return y_pred
                else:
                    return y_pred > 0.5

            elif self.args.graphtask == 'multiclass':
                y_test = list()
                y_pred = list()
                x_embed = list()

                class_weights=torch.tensor(np.array([11.54842398,  8.31405056,  0.35802132]),dtype=torch.float).to(self.device)
                # class_weights=torch.tensor(np.array([12.0,  8.5,  0.3]),dtype=torch.float).to(self.device)
                loss_fn = nn.CrossEntropyLoss(class_weights, reduction='mean')
                # loss_fn = FocalLoss(alpha=class_weights)
                # loss_fn = nn.CrossEntropyLoss()
                # m_fn = nn.Softmax(dim=1)

                # correct = 0.
                loss = 0.
                
                y_test = list()
                y_pred = list()

                for data in test_loader:
                    data = data.to(self.device)
                    if return_attention_weights:
                        out, att_mol = self.model_test(data, return_attention_weights=return_attention_weights)
                        out = out.to(self.device)
                    else:
                        out = self.model_test(data).to(self.device)
                    # pred = m_fn(out).cpu()
                    pred = out.to(self.device)
                    # pred_round = pred > 0.5
                    # pred_hat = torch.max(pred, 1).indices
                    pred_hat = pred.argmax(dim=1)
                    # correct += pred_round.eq(data.y.cpu()).sum().item()
                    # data_y = data.y.squeeze().type(torch.LongTensor).to(self.device)
                    data_y = F.one_hot(data.y.squeeze().to(torch.int64), num_classes=self.args.class_number).type(torch.DoubleTensor).to(self.device)
                    # loss += loss_fn(m_fn(out.cpu()), data_y.cpu()).item()
                    loss += loss_fn(out.to(self.device), data_y.to(self.device)).item()
                    y_test.extend((torch.squeeze(data.y).cpu()).detach().numpy())
                    y_pred.extend((pred_hat.cpu()).detach().numpy())
                    x_embed.extend(self.model_test.last_mol_embedding.detach().cpu().numpy())
                
                y_pred = torch.Tensor(np.array(y_pred, dtype='int64')).type(torch.IntTensor)
                y_test = torch.Tensor(np.array(y_test, dtype='int64')).type(torch.IntTensor)
                
                if print_result:
                    print('y_test:', y_test)
                    print('y_pred:', y_pred)

                self.x_embed = x_embed
                self.y_test = y_test

                # interpretation
                if return_attention_weights:
                    self.att_mol = att_mol

                return y_pred

    def test_unk(self, test_loader, return_attention_weights=False, print_result=True):
        self.model_test.eval()

        with torch.no_grad():
            if self.args.graphtask == 'multiclass':
                y_test = list()
                y_pred = list()
                x_embed = list()

                class_weights=torch.tensor(np.array([11.54842398,  8.31405056,  0.35802132]),dtype=torch.float).cpu()
                # class_weights=torch.tensor(np.array([12.0,  8.5,  0.3]),dtype=torch.float).cpu()
                loss_fn = nn.CrossEntropyLoss(class_weights, reduction='mean')
                # loss_fn = FocalLoss(alpha=class_weights)
                # loss_fn = nn.CrossEntropyLoss()
                # m_fn = nn.Softmax(dim=1)

                # correct = 0.
                loss = 0.
                
                ids = list()
                m = nn.Softmax(dim=1)
                y_prob = list()
                y_test = list()
                y_pred = list()

                for data in test_loader:
                    data = data.to(self.device)
                    if return_attention_weights:
                        out, att_mol = self.model_test(data, return_attention_weights=return_attention_weights)
                        out = out.to(self.device)
                    else:
                        out = self.model_test(data).to(self.device)
                    # pred = m_fn(out).cpu()
                    pred = out.cpu()
                    pred_softmax = m(pred)
                    # pred_round = pred > 0.5
                    # pred_hat = torch.max(pred, 1).indices
                    pred_hat = pred.argmax(dim=1)
                    # correct += pred_round.eq(data.y.cpu()).sum().item()
                    # data_y = data.y.squeeze().type(torch.LongTensor).to(self.device)
                    data_y = F.one_hot(data.y.squeeze().to(torch.int64), num_classes=self.args.class_number).type(torch.DoubleTensor).to(self.device)
                    # loss += loss_fn(m_fn(out.cpu()), data_y.cpu()).item()
                    loss += loss_fn(out.cpu(), data_y.cpu()).item()
                    y_test.extend((torch.squeeze(data.y).cpu()).detach().numpy())
                    y_pred.extend((pred_hat.cpu()).detach().numpy())
                    x_embed.extend(self.model_test.last_mol_embedding.detach().cpu().numpy())
                    ids.extend(data.ids)
                    y_prob.extend((pred_softmax.cpu()).detach().numpy())
                
                y_pred = torch.Tensor(np.array(y_pred, dtype='int64')).type(torch.IntTensor)
                y_test = torch.Tensor(np.array(y_test, dtype='int64')).type(torch.IntTensor)
                
                if print_result:
                    print('y_test:', y_test)
                    print('y_pred:', y_pred)

                self.x_embed = x_embed
                self.y_test = y_test

                # interpretation
                if return_attention_weights:
                    self.att_mol = att_mol

                y_prob = np.array(y_prob)
                # return pd.DataFrame(data={'Id': ids, 'pred': y_pred, 'class0': y_prob.T[0], 'class1': y_prob.T[1], 'class2': y_prob.T[2]})
                return pd.DataFrame(data={'Id': ids, 'pred': y_pred})


    def getXEmbed(self):
        return np.array(self.x_embed)

    def getYTest(self):
        return np.array(self.y_test.squeeze())

    def getAttentionMol(self):
        return self.att_mol
