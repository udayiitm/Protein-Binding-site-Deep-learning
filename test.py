from openbabel import pybel
import os
import sys
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns


from feature_extractor import FeatureExtractor
from mol_3d_grid import Mol3DGrid
from PUResNet import PUResNet


mol = next(pybel.readfile("mol2", "/Users/Lenovo/Desktop/Prj/test/1a2n_1/protein.mol2"))

#o_path = make_output_folder("test/output", "test/1a2n_1/protein.mol2")

fe = FeatureExtractor()
print(len(mol.atoms))
print(fe.get_feature(mol)[1][0])
print(fe.FEATURE_NAMES)

mol_grid = Mol3DGrid(max_dist=35.0, scale=0.5)
grid = mol_grid.setMol(mol).transform()

d = mol_grid.box_size
f = len(mol_grid.fe.FEATURE_NAMES)
model = PUResNet(d, f)
model.load_weights("/Users/Lenovo/Downloads/whole_trained_model1.hdf")
#model.summary()
x = model.predict(np.array([grid]))
print(x.sum(), x)

with open("prediction.pickle", "wb") as f:
     pickle.dump(x, f)

x = None
with open("prediction.pickle", "rb") as f:
    x = pickle.load(f)

mol_grid = Mol3DGrid(max_dist=35.0, scale=0.5)
grid = mol_grid.setMol(mol).transform()

pockets = mol_grid.segment_grid_to_pockets(x).get_pockets_mols()
print(pockets, len(pockets[0].atoms))



from openbabel import pybel


for i, pocket in enumerate(pockets):
    # Create an output MOL2 file path
    output_mol2_path = f"output_pocket_{i + 1}.mol2"

    # Save the pocket as a MOL2 file
    with pybel.Outputfile("mol2", "/Users/Lenovo/Desktop/Prj", overwrite=True) as mol2file:
        mol2file.write(pocket)

#    print(f"Pocket {i + 1} saved to {"/Users/Lenovo/Desktop"}")
    print(f"Pocket {i + 1} saved to { '/Users/Lenovo/Desktop' }")

    


