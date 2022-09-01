import os
import pandas as pd
from ClusterPlot import ClusterPlot
import numpy as np

RANDOM_STATE = 42

DATA = []

path = os.path.join(os.getcwd(), "data/Annotated Data/scenarios/SLH/SLH_1_1_annotated.csv")
DATA = pd.read_csv(path, sep=",")
# print(DATA.head())
X_columns = DATA.columns[2:-1]
X = DATA[X_columns].values
LABELS = DATA['label'].values

PLOT_LABELS = []
lbl_tmp = []
for label in LABELS:
    if "STD" in label:
        lbl_tmp.append(0)
    elif "WAL" in label:
        lbl_tmp.append(1)
    elif "STN" in label:
        lbl_tmp.append(2)
    elif "CSI" in label:
        lbl_tmp.append(3)
    elif "SIT" in label:
        lbl_tmp.append(4)
    elif "CSO" in label:
        lbl_tmp.append(5)
    else:
        lbl_tmp.append(0)

PLOT_LABELS = np.asarray(lbl_tmp)

tmp_labels = {
    0: "STD",
    1: "WAL",
    2: "STN",
    3: "CSI",
    4: "SIT",
    5: "CSO"
}

cplot = ClusterPlot.ClusterPlot(
    learning_rate=0.5,
    show_fig=True,
    random_state=RANDOM_STATE,
    n_iter=50,
    class_to_label=tmp_labels,
    k=300,
    anchors_method='birch',
    birch_threshold=5.6,
    batch_size=1,
    magnitude_step=True,
    top_greedy=1,
    alpha=0.3,
    douglas_peucker_tolerance=0.3,
    smooth_iter=3
)

if __name__ == "__main__":
    plot = cplot.fit_transform(X, PLOT_LABELS)