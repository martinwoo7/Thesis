class Visual:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        print("Visual loaded")
    

    def tmap(self, k, kc):
        from faerun import Faerun
        import tmap as tm

        print("Visualizing with TMAP")
        CFG_TMAP = tm.LayoutConfiguration()
        CFG_TMAP.k = k # 100
        CFG_TMAP.kc = kc # 10
        CFG_TMAP.fme_iterations = 1000
        CFG_TMAP.node_size = 1/ 55

        dims = 128
        seed = 42
        enc = tm.Minhash(d=dims, seed=seed)
        lf = tm.LSHForest(d=dims, l=128, store=True)


        for row in self.data:
            tmp = tm.VectorFloat(list(row))
            lf.add(enc.from_weight_array(tmp))
        
        # lf.batch_add(enc.batch_from_weight_array(fps))
        lf.index()

        x_tmap, y_tmap, s, t, _ = tm.layout_from_lsh_forest(lf, config=CFG_TMAP)
        lf.clear()
        
        import numpy as np
        unique_labels = len(np.unique(self.labels))
        print(unique_labels)
        dataset = "MobiAct"
        legend_labels = [
            (0, "Standing"),
            (1, "Walking"),
            (2, "Jogging"),
            (3, "Jumping"),
            (4, "STU"),
            (5, "STN"),
            (6, "SCH"),
            (7, "Sitting"),
            (8, "CHU"),
            (9, "Car Sit In"),
            (10, "Car Sit Out"),
            (11, "Lying"),
            (12, "FOL"),
            (13, "FLK"),
            (14, "BSC"),
            (15, "SDL")
        ]

        # if unique_labels == 16:
        #     dataset = "MobiAct"
        #     legend_labels = [
        #         (0, "Standing"),
        #         (1, "Walking"),
        #         (2, "Jogging"),
        #         (3, "Jumping"),
        #         (4, "STU"),
        #         (5, "STN"),
        #         (6, "SCH"),
        #         (7, "Sitting"),
        #         (8, "CHU"),
        #         (9, "Car Sit In"),
        #         (10, "Car Sit Out"),
        #         (11, "Lying"),
        #         (12, "FOL"),
        #         (13, "FLK"),
        #         (14, "BSC"),
        #         (15, "SDL")
        #     ]

        # else:
        #     dataset = "UCI"
        #     legend_labels = [
        #         (0, "Walking"),
        #         (1, "Walking Up"),
        #         (2, "Walking Down"),
        #         (3, "Sitting"),
        #         (4, "Standing"),
        #         (5, "Laying")
        #     ]

        faerun = Faerun(
            clear_color="#111111",
            view="front",
            coords=False,
            alpha_blending=True,
            legend_title="",
        )

        faerun.add_scatter(
            dataset,
            {"x": x_tmap, "y":y_tmap, "c": self.labels, "labels": self.labels},
            colormap='tab20',
            point_scale=10,
            max_point_size=50,
            has_legend=True,
            categorical=True,
            legend_title="Activities",
            legend_labels=legend_labels,
            shader="smoothCircle",
        )

        faerun.add_tree(
            dataset + "_tree",
            {"from": s, "to": t},
            point_helper=dataset,
            color="#222222"
        )

        print("Plotting...")
        faerun.plot(dataset + str(CFG_TMAP.k))
        print("Finished!")

    def umap(self, neighbours, dist):
        from faerun import Faerun
        import umap
        import umap.plot
        print("Visualizing with UMAP")

        # neighbours = 300
        # min_dist = 0.01
        mapper = umap.UMAP(densmap=False, n_neighbors=neighbours, min_dist=dist)
        embedding = mapper.fit(self.data)
        p = umap.plot.points(embedding, labels=self.labels)
        umap.plot.plt.show()
    
    def densmap(self, neighbours, dist):
        from faerun import Faerun
        import umap
        import umap.plot
        print("Visualizing with DensMAP")

        mapper = umap.UMAP(densmap=True, n_neighbors=neighbours, min_dist=dist)
        embedding = mapper.fit(self.data)
        p = umap.plot.points(embedding, labels=self.labels)
        umap.plot.plt.show()

    def clusterplot(self):
        from ClusterPlot import ClusterPlot
        import numpy as np
        print("Visualizing with ClusterPlot")
        
        unique_labels = len(np.unique(self.labels))
        if unique_labels == 16:
            dataset_name="MobiAct"
            legend_labels = {
                0: "STD",
                1: "WAL",
                2: "JOG",
                3: "JUM",
                4: "STU",
                5: "STN",
                6: "SCH",
                7: "SIT",
                8: "CHU",
                9: "CSI",
                10: "CSO",
                11: "LYI",
                12: "FOL",
                13: "FLK",
                14: "BSC",
                15: "SDL"
            }
        elif unique_labels == 6:
            dataset_name="UCI"
            legend_labels = {
                0: "Walking",
                1: "Walking Up",
                2: "Walking Down",
                3: "Sitting",
                4: "Standing",
                5: "Laying"
            }

        cplot = ClusterPlot.ClusterPlot(
            learning_rate=0.5,
            show_fig=True,
            n_iter=14,
            class_to_label=legend_labels,
            k=300,
            anchors_method='birch',
            birch_threshold=0.05,
            batch_size=1,
            magnitude_step=True,
            top_greedy=1,
            alpha=0.3,
            douglas_peucker_tolerance=0.3,
            smooth_iter=3,
            umap_n_neighbors=300,
            umap_min_dist=0.01,
            dataset=dataset_name,
        )
        plot = cplot.fit_transform(self.data, self.labels)


# from te_utils import *
# from sklearn import preprocessing as preproc
# import numpy as np
# DATASET = "SLH"
# training_data, training_labels = load_train(DATASET)
# testing_data, testing_labels = load_test(DATASET)

# data = np.concatenate((training_data, testing_data))
# labels = np.concatenate((training_labels, testing_labels))

# orig_shape = data.shape
# scaler = preproc.Normalizer().fit(data.reshape(-1, orig_shape[2]))
# data = scaler.transform(data.reshape(-1, orig_shape[2])).reshape(orig_shape)
# visual = Visual(data, labels)
# visual.tmap(100, 10)
