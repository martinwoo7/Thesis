import umap
import umap.plot
import os
import pandas as pd
import matplotlib.pyplot as plt
from faerun import Faerun
import sklearn.datasets

# pendigits = sklearn.datasets.load_digits()
# mnist = sklearn.datasets.fetch_openml("mnist_784")
# loading dataset
DATA = []

path = os.path.join(os.getcwd(), "data/Annotated Data/scenarios/SLH/SLH_1_1_annotated.csv")
DATA = pd.read_csv(path, sep=",")
# print(DATA.head())
X_columns = DATA.columns[2:-1]
X = DATA[X_columns].values
LABELS = DATA['label'].values

if __name__ == "__main__":
    # main()
    print("Displaying with densitymap")
    for i in [300]:
        mapper = umap.UMAP(densmap=False, random_state=42, n_neighbors=i, min_dist=0.01)
        embedding = mapper.fit(X)
        p = umap.plot.points(embedding, labels=LABELS)
        umap.plot.plt.show()
