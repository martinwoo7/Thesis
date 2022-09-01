import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns
import numpy as np

from evolving.util import load_dataset
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


# First, creating a benchmark using t-SNE on the dataset. 
# Embed to 3 components
data, labels = load_dataset.load_dataset("mobiscenario")

le = preprocessing.LabelEncoder()
le.fit(labels)
changed_labels = le.transform(labels)

df = pd.DataFrame(data)
df['y'] = labels
df['changed_y'] = changed_labels

perplex = 300
# print("Emebdding scenario using TSNE in 2 components")
# print("Perplexity: ", perplex)
# tsne = TSNE(n_components=2, verbose=1, init='random', random_state=42, perplexity=perplex)
# tsne_results = tsne.fit_transform(data)

# df['tsne-2d-one'] = tsne_results[:,0]
# df['tsne-2d-two'] = tsne_results[:,1]
# plt.figure(figsize=(16, 10))
# plt.title("TSNE 2D plot")
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette('hls', 6),
#     data=df,
#     legend="full",
#     alpha=0.3
# )

# Trying with 3 dimensions
print("perplexity", perplex)
tsne = TSNE(n_components=3, verbose=1, init='random', random_state=42, perplexity=perplex)
tsne_results = tsne.fit_transform(data)
df['tsne-3d-one'] = tsne_results[:,0]
df['tsne-3d-two'] = tsne_results[:,1]
df['tsne-3d-three'] = tsne_results[:,2]

ax = plt.figure(figsize=(20,10)).gca(projection='3d')
plt.title("TSNE 3D plot")
# plt.axis('off')
cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

labels = np.unique(df['y'])
palette =  sns.color_palette("husl", len(labels))


for label, color in zip(labels, palette):
    # ax = fig.add_subplot(2, 3, j, projection='3d')
    df1 = df[df['y'] == label]
    sc = ax.scatter(
        xs=df1['tsne-3d-one'],
        ys=df1["tsne-3d-two"],
        zs=df1["tsne-3d-three"],
        s=5,
        color=color,
        alpha=0.3,
        label=label
    )
ax.set_xlabel('tsne-one')
ax.set_ylabel('tsne-two')
ax.set_zlabel('tsne-three')
plt.legend(bbox_to_anchor = (1.05, 1), loc=2)



fig = plt.figure(figsize=(20,10))
plt.title("TSNE 3D plot")
plt.axis('off')
cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

labels = np.unique(df['y'])
palette =  sns.color_palette("husl", len(labels))


j = 1
for label, color in zip(labels, palette):
    ax = fig.add_subplot(2, 3, j, projection='3d')
    df1 = df[df['y'] == label]
    sc = ax.scatter(
        xs=df1['tsne-3d-one'],
        ys=df1["tsne-3d-two"],
        zs=df1["tsne-3d-three"],
        s=5,
        color=color,
        alpha=0.3,
        label=label
    )
    ax.set_xlabel('tsne-one')
    ax.set_ylabel('tsne-two')
    ax.set_zlabel('tsne-three')
    plt.legend(bbox_to_anchor = (1.05, 1), loc=2)

    j += 1




# Second, compare to PCA in 2 and 3 dimensions

# print("Embedding scenario using PCA in 3 components")
# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(data)

# df['pca-one'] = pca_result[:,0]
# df['pca-two'] = pca_result[:,1]
# df['pca-three'] = pca_result[:,2]


# print("Explained variation per princiapal component: {}".format(pca.explained_variance_ratio_))
# fig = plt.figure(figsize=(16, 10))
# plt.title("PCA reduction 2-D plotting")
# sns.scatterplot(
#     x="pca-one", y="pca-two",
#     hue="y",
#     palette=sns.color_palette("hls", 6),
#     data=df,
#     legend="full",
#     alpha=0.3
# )

# fig = plt.figure(figsize=(16, 10))
# ax = Axes3D(fig, auto_add_to_figure=False)
# fig.add_axes(ax)
# # cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

# plt.title("PCA reduction 3-D plotting")
# sc = ax.scatter(
#     df['pca-one'], df['pca-two'], df['pca-three'], s=20, cmap=cmap, alpha=0.3, c=df['changed_y']
# )
# ax.set_xlabel('pca-one')
# ax.set_ylabel('pca-two')
# ax.set_zlabel('pca-three')

# plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

