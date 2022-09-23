# use t-SNE to show the feature distribution

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from einops import reduce
import scipy.io
import torch

def plt_tsne(data, label, per):
    data = data.cpu().detach().numpy()
    data = reduce(data, 'b n e -> b e', reduction='mean')
    label = label.cpu().detach().numpy()

    tsne = manifold.TSNE(n_components=2, perplexity=per, init='pca', random_state=166)
    X_tsne = tsne.fit_transform(data)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure()
    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.Set1(label[i]))
        plt.xticks([])
        plt.yticks([])
    # plt.show()
    plt.savefig('./test.png', dpi=600)
