import os
from typing import List, Tuple
from openbabel import pybel
import pickle
import numpy as np
from sklearn.model_selection import KFold
from keras.utils import Sequence
import matplotlib.pyplot as plt

# from utils.feature_extractor import FeatureExtractor
# from utils.mol_3d_grid import Mol3DGrid
# from model.PUResNet import PUResNet
from keras.optimizers import Adam, Nadam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.layers import Flatten
import tensorflow as tf
from feature_extractor import FeatureExtractor
from mol_3d_grid import Mol3DGrid
from PUResNet import PUResNet
from ResUNetPP import ResUnetPlusPlus
from metrics import tversky_loss, f1_score, f2_score, iou_metric
from keras.callbacks import ModelCheckpoint, EarlyStopping

lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9
)


def dice_coef(y_true, y_pred):
    y_true_f = Flatten()(y_true)
    y_pred_f = Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1e-7) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-7
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


class ModelTrainer:
    def __init__(self):
        self.proteins_data: np.ndarray = None
        self.mol_grid = Mol3DGrid(max_dist=35.0, scale=0.5)
        d = self.mol_grid.box_size
        f = len(self.mol_grid.fe.FEATURE_NAMES)
        # self.model = PUResNet(d, f)
        self.model = ResUnetPlusPlus().build_model()
        #self.model.load_weights("/content/weights.h5")

        pass

    def train_model(self) -> None:
        print("Compiling model...")
        self.model.compile(
            # optimizer=Adam(learning_rate=lr_schedule),
            optimizer=Nadam(1e-2),
            loss=tversky_loss,
            # metrics=[BinaryIoU(num_classes=2, name='iou_metric')],
            metrics=["acc", iou_metric],
        )

        print("Loading Data...")
        self.load_training_data()

        print("Starting Training...")
        batch_size = 20
        kf = KFold(n_splits=4, shuffle=True, random_state=50)

        # Define callbacks.
        checkpoint_cb = ModelCheckpoint(
            "weights.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )
        early_stopping_cb = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        folds_results = []
        folds_evaluations = []

        for i, (train_index, test_index) in enumerate(kf.split(self.proteins_data)):
            print(f"Fold #{i+1} => train:{len(train_index)} & test:{len(test_index)}.")

            # spliting data
            Xy_train = []
            for i in train_index:
                Xy_train.append(self.proteins_data[i])
            Xy_test = []
            for i in test_index:
                Xy_test.append(self.proteins_data[i])

            train_data_generator = ProteinsGridsGenerator(
                self.proteins_data, batch_size, self.mol_grid
            )
            test_data_generator = ProteinsGridsGenerator(
                Xy_test, batch_size, self.mol_grid
            )
            
            
            


            fold_results = self.model.fit(
                x=train_data_generator,
                epochs=50,
                validation_data=test_data_generator,
                callbacks=[checkpoint_cb, early_stopping_cb],
            )

            evaluations = self.model.evaluate_generator(generator=test_data_generator)

            folds_results.append(fold_results.history)
            folds_evaluations.append(evaluations)

        print("Training is complete.")

        print("Saving results")
        self.model.save_weights("weights.h5", overwrite=True)

        with open(f"folds_results.pickle", "wb") as f:
            pickle.dump(folds_results, f)
        with open(f"folds_evaluations.pickle", "wb") as f:
            pickle.dump(folds_evaluations, f)

        for history, evaluation, fold_i in zip(
            folds_results, folds_evaluations, range(1, 6)
        ):
            plt.plot(history["loss"], label="Training Loss")
            plt.plot(history["val_loss"], label="Validation Loss")

            # plt.plot(history["acc"], label="Accuracy")
            # plt.plot(history["val_acc"], label="Val Accuracy")

            # plt.plot(history["f1_score"], label="F1 Score")
            # plt.plot(history["val_f1_score"], label="F1 Val Score")

            # plt.plot(history["f2_score"], label="F2 Score")
            # plt.plot(history["val_f2_score"], label="F2 Val Score")

            plt.plot(history["iou_metric"], label="IOU")
            plt.plot(history["val_iou_metric"], label="Val IOU")

            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss/Accuracy")
            plt.title(f"Fold #{fold_i} Training History")
            plt.savefig(fname=f"fold-{fold_i}")
            plt.show()

        print("Done.")

    def load_training_data(self):
        with open("train_data.pickle", "rb") as f:
            # [rows[x[pcoords, pfeats], scoords]
            self.proteins_data = pickle.load(f)

    def save_training_data(
        self,
        training_data_path="/Users/Lenovo/Desktop/final_data",
    ) -> Tuple[List[pybel.Molecule], List[pybel.Molecule]]:
        # get list of proteins files storted by name
        protein_names = os.listdir(training_data_path)
        protein_names.sort()

        proteins_data = []
        fe = FeatureExtractor()

        # determine section of train data to save in this session
        start = 0
        end = len(protein_names)
        
        

        for i in range(start, end):
            
            
            protein_name = protein_names[i]
            if protein_name.startswith("."):
                continue
            print("Protein", protein_name, "#", i)

            path = os.path.join(training_data_path, protein_name)
            mol_path = path + "/protein.mol2"
            site_path = path + "/site.mol2"
            ligand_path = path + "/ligand.mol2"

            protein = next(pybel.readfile("mol2", mol_path))
            site = next(pybel.readfile("mol2", site_path))
            ligand = next(pybel.readfile("mol2", ligand_path))

            protein_coords, protein_features = fe.get_feature(protein)
            site_coords = fe.get_all_coords(site)
            ligand_coords = fe.get_all_coords(ligand)

            proteins_data_row = [
                [protein_coords, protein_features],
                site_coords,
                ligand_coords,
            ]
            proteins_data.append(proteins_data_row)

        with open(f"train_data.pickle", "wb") as f:
            pickle.dump(proteins_data, f)

        self.proteins_data = proteins_data

    # def train_model(
    #     self, proteins: List[pybel.Molecule], proteins_sites: List[pybel.Molecule]
    # ):
    #     # iterate over data
    #     # create metrics for training

    #     X = map(proteins, lambda p: self.mol_grid.setMol(p).transform())
    #     y = map(proteins_sites, lambda ps: self.mol_grid.setMol(ps).transform())
    #     print(X.shape, y.shape, X[0], y[0])

    #     pass

    def success_rate(self, y_true, y_pred) -> float:
        y_true

        pass

    def preprocess_train_data(self):
        X = []
        y = []
        for row in self.proteins_data:
            coords, feats = row[0]
            coords, feats = np.array(coords), np.array(feats)
            grid = self.mol_grid.setMolAsCoords(coords, feats).transform()
            X.append(grid)

            site_coords = np.array(row[1])
            grid = self.mol_grid.setMolAsCoords(site_coords).transform()
            y.append(grid)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        print(X.shape, y.shape)

        return X, y


import tensorflow as tf
from keras import backend as K


class ProteinsGridsGenerator(Sequence):
    def __init__(self, proteins_data, batch_size, mol_grid):
        self.proteins_data = proteins_data
        self.mol_grid = mol_grid
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.proteins_data) / float(self.batch_size))).astype(
            np.int
        )

    def __getitem__(self, idx):
        batch_data = self.proteins_data[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        batch_x = []
        batch_y = []
        for row in batch_data:
            coords, feats = row[0]
            grid = self.mol_grid.setMolAsCoords(coords, feats).transform()
            batch_x.append(grid)

            # fig = plt.figure(figsize=(8, 6))
            # ax = fig.add_subplot(111, projection="3d")
            # xs = self.mol_grid.coords[:, 0]
            # ys = self.mol_grid.coords[:, 1]
            # zs = self.mol_grid.coords[:, 2]
            # ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors="w")
            # ax.set_xlabel("X")
            # ax.set_ylabel("Y")
            # ax.set_zlabel("Z")
            # plt.show()

            site_coords = np.asarray(row[1])
            grid = self.mol_grid.setMolAsCoords(site_coords).transform()
            batch_y.append(grid)

            # fig = plt.figure(figsize=(8, 6))
            # ax = fig.add_subplot(111, projection="3d")
            # xs = self.mol_grid.coords[:, 0]
            # ys = self.mol_grid.coords[:, 1]
            # zs = self.mol_grid.coords[:, 2]
            # ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors="w")
            # ax.set_xlabel("X")
            # ax.set_ylabel("Y")
            # ax.set_zlabel("Z")
            # plt.show()

        batch_x = np.asarray(batch_x, dtype=np.float32)
        batch_y = np.asarray(batch_y, dtype=np.float32)

        return batch_x, batch_y


mt = ModelTrainer()
mt.save_training_data()
#!cp /content/train_data.pickle /content/drive/MyDrive/GP/train_data.pickle
mt.train_model()
#!cp /content/weights/2.h5 /content/drive/MyDrive/GP/Results/test_2.h5

print("Number of training samples:", len(self.proteins_data))
