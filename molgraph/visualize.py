######################
### Import Library ###
######################

# general library
import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import umap
# import umap.plot
# rdkit
from rdkit.Chem import Draw

##############################
### Visualization Function ###
##############################

# Molecule in grid
def drawGridMolecule(mols, labels=[]):
    if len(mols) <= 5:
        fig, ax = plt.subplots(1, len(mols), figsize=(30,5))
        for i, a_i in enumerate(ax):
            try:
                a_i.grid(False)
                a_i.axis('off')
                a_i.imshow(Draw.MolToImage(mols[i]))
                if len(labels) != 0:
                    a_i.text(50, 350, labels[i], size=18)
            except:
                print('molecule grid error')
    else:
        fig, ax = plt.subplots(math.ceil(len(mols)/5), 5, figsize=(30,5*math.ceil(len(mols)/5)))
        for i, a_i in enumerate(ax):
            for j, a_j in enumerate(a_i):
                try:
                    a_j.grid(False)
                    a_j.axis('off')
                    a_j.imshow(Draw.MolToImage(mols[i*5+j]))
                    if len(labels) != 0:
                        a_j.text(50, 350, labels[i*5+j], size=18)
                except:
                    print('not divided by 5')

# Scatter hist
def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.1
    xymax = max(np.max(x), np.max(y))
    xymin = min(np.min(x), np.min(y))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(math.floor(xymin), lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

# Legend
def getLegend(graphtask, y_test):
    if graphtask == 'regression':
        min_value = np.min(y_test)
        max_value = np.max(y_test)
        
        interval_num = 10
        interval = (max_value-min_value)/interval_num
        ranging = [(min_value+(interval*i), min_value+(interval*(i+1))) for i in range(interval_num)]

        y_test_new = list()
        for y in y_test:
            for i, r in enumerate(ranging):
                if r[0] <= y < r[1]:
                    y_test_new.append(i)
                    break
                elif y == max_value:
                    y_test_new.append(interval_num-1)
                    break

        y_test = np.array(y_test_new)

        legend_new = list()
        for i, r in enumerate(ranging):
            if i != len(ranging)-1:
                legend_new.append('['+str("{:.2f}".format(r[0]))+','+str("{:.2f}".format(r[1]))+')')
            else:
                legend_new.append('['+str("{:.2f}".format(r[0]))+','+str("{:.2f}".format(r[1]))+']')

        legend = legend_new
        return y_test, legend
    else:
        return None

# PCA
def visualize_pca(x_embed, y_test, title=None, path=None, legend=None):
    pca = PCA(n_components=2)
    components = pca.fit_transform(x_embed)

    df = pd.DataFrame()
    df["y"] = y_test 
    df["comp-1"] = components[:,0]
    df["comp-2"] = components[:,1]

    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.to_list(),
                    palette=sns.color_palette("hls", len(set(y_test))),
                    data=df)
    ax.set(title=title+" - PCA projection") 
    handles, labels  =  ax.get_legend_handles_labels()
    legend = legend if legend is not None else labels
    lgd = ax.legend(handles, legend, loc='right', bbox_to_anchor=(1.43, 0.72))
    fig.savefig(path+'/'+title+'_pca.png', dpi=150, bbox_extra_artists=(lgd,), bbox_inches='tight') 

# T-SNE
def visaulize_tsne(x_embed, y_test, title=None, path=None, legend=None):
    tsne = TSNE(n_components=2, verbose=1, random_state=42)
    x = x_embed
    y = y_test
    z = tsne.fit_transform(x) 

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", len(set(y_test))),
                    data=df)
    ax.set(title=title+" - T-SNE projection") 
    handles, labels  =  ax.get_legend_handles_labels()
    legend = legend if legend is not None else labels
    lgd = ax.legend(handles, legend, loc='right', bbox_to_anchor=(1.43, 0.72))
    fig.savefig(path+'/'+title+'_tsne.png', dpi=150, bbox_extra_artists=(lgd,), bbox_inches='tight') 

# # UMAP
# def visualize_umap(x_embed, y_test, title=None, path=None):
#     print(title+" - UMAP projection")
#     mapper = umap.UMAP(n_neighbors=15).fit(x_embed)
#     umap.plot.points(mapper, labels=y_test)