'''
Creating a visualization for the mobiact v2.0 dataset using faerun and tmap
'''

import tmap as tm
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# import umap
import os

# from matplotlib.colors import LinearSegmentedColormap
from faerun import Faerun
# from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

# loading dataset
DATA = []

# path = "Thesis work/data/Annotated Data/scenarios/SLH/SLH_1_1_annotated.csv"
path = os.path.join(os.path.dirname(os.getcwd()), "data/Annotated Data/scenarios/SLH/SLH_1_1_annotated.csv")
DATA = pd.read_csv(path, sep=",")
# print(DATA.head())
X_columns = DATA.columns[2:-1]
X = DATA[X_columns].values
# print(X[0:3])
FAERUN_LABELS = DATA['label'].values
LABELS = []

lbl_tmp = []
for label in FAERUN_LABELS:
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

LABELS = lbl_tmp
# assert False

CFG_TMAP = tm.LayoutConfiguration()
CFG_TMAP.k = 80
CFG_TMAP.kc = 50
CFG_TMAP.fme_iterations = 1000
CFG_TMAP.node_size = 1/10

def main(value):
    ''' Main Function '''
    dims = 1024
    enc = tm.Minhash(dims)
    lf = tm.LSHForest(dims, 128, store=True)

    print("Running tmap with value: ", value)
    fps = []

    for row in X:
        tmp = tm.VectorFloat(list(row))
        # fps.append(tmp)
        lf.add(enc.from_weight_array(tmp))

    # lf.add(enc.from_weight_array(tm.VectorFloat()))
    lf.index()

    x_tmap, y_tmap, s, t, _ = tm.layout_from_lsh_forest(lf, config=CFG_TMAP)
    lf.clear()

    legend_labels = [
        (0, "Standing"),
        (1, "Walking"),
        (2, "Stair Down"),
        (3, "Car Step In"),
        (4, "Sit"),
        (5, "Car Step Out"),
    ]

    faerun = Faerun(
        clear_color="#111111", 
        view="front", 
        coords=False, 
        alpha_blending=True,
        legend_title="",
    )
    faerun.add_scatter(
        "mobiact",
        {"x": x_tmap, "y": y_tmap, "c": LABELS, "labels": FAERUN_LABELS},
        colormap="tab10",
        point_scale=2.5,
        max_point_size=10,
        has_legend=True,
        categorical=True,
        legend_title="Activities",
        legend_labels=legend_labels,
        shader="smoothCircle",
    )
    faerun.add_tree(
        "mobiact_tree",
        {"from": s, "to": t},
        point_helper="mobiact",
        color="#222222",
    )
    print("Plotting...")
    faerun.plot("mobiact" + str(value))
    print("Finished!")


if __name__ == "__main__":
    for i in [80]:
        CFG_TMAP.kc = i
        main(i)