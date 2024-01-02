from skimage.segmentation import clear_border
from skimage.morphology import closing
from skimage.measure import label
import numpy as np
from math import ceil
from openbabel import pybel, openbabel
from numpy import ndarray
from feature_extractor import FeatureExtractor
from typing import List, Union


class Mol3DGrid:
    """
    Transform molecule atoms to 4D array having it's features arrays distributed in 3D space
    so that it can be used as model input. X[Y[Z[AtomsFeatures[]]]]
    """

    def __init__(
        self, max_dist: Union[float, int] = 10.0, scale: Union[float, int] = 1.0
    ) -> None:
        # validate arguments
        if scale is None:
            raise ValueError("scale must be set to make predictions")
        if not isinstance(scale, (float, int)):
            raise TypeError("scale must be number")
        if scale <= 0:
            raise ValueError("scale must be positive")

        if not isinstance(max_dist, (float, int)):
            raise TypeError("max_dist must be number")
        if max_dist <= 0:
            raise ValueError("max_dist must be positive")

        # initialize attributes
        self.max_dist = float(max_dist)
        """
        Maximum distance (in Angstroms) between atom and box center. Resulting box has
        size of 2*`max_dist`+1 Angstroms and atoms that are too far away are not included.
        """
        self.scale = float(scale)
        """Make atoms bigger (> 1) or smaller (< 1) inside the grid."""
        self.resolution = 1.0 / self.scale
        """Resolution of a grid (in Angstroms)."""
        self.box_size = ceil(2 * self.max_dist / self.resolution + 1)
        """Grid box size (in Angstroms)."""
        self.step = np.array([1.0 / self.scale] * 3)
        """Step is the dimension (in Angstroms) of one cell in the new scaled 3D grid."""
        self.fe = FeatureExtractor()
        """Feature Extractor to get atoms coordinates and features."""

        self.mol: pybel.Molecule = None
        self.coords: ndarray = None
        self.features: ndarray = None
        self.centroid: ndarray = None
        self.origin: ndarray = None
        self.grid_labeled_pockets: ndarray = None
        self.pockets_num: int = None

        pass

    def setMol(self, mol: pybel.Molecule):
        if not isinstance(mol, pybel.Molecule):
            raise TypeError(
                "mol should be a pybel.Molecule object, got %s " "instead" % type(mol)
            )

        self.mol = mol

        prot_coords, prot_features = self.fe.get_feature(self.mol)
        self.coords = prot_coords
        self.features = prot_features
        self.centroid = None
        self.origin = None
        self.grid_labeled_pockets = None
        self.pockets_num = None

        return self

    def setMolAsCoords(self, coords: ndarray, feats: ndarray = None):
        self.coords = coords
        self.features = feats
        self.centroid = None
        self.origin = None
        self.grid_labeled_pockets = None
        self.pockets_num = None

        return self

    def transform(self) -> ndarray:
        self._translateToCenter()
        self._scaleAndCrop()
        mol_grid = self._insertInFixed3DGrid()

        return mol_grid

    def _translateToCenter(self) -> None:
        """Move centroid to zero origin in 3D space"""
        self.centroid = self.coords.mean(axis=0)
        self.coords -= self.centroid
        self.origin = self.centroid - self.max_dist

        pass

    def _scaleAndCrop(self) -> None:
        # translate with max included distance and scale it
        grid_coords = (self.coords + self.max_dist) / self.resolution

        # converts data to nearest integers
        grid_coords = grid_coords.round().astype(int)

        # crop and return in box not cropped atoms coords and features only
        in_box = ((grid_coords >= 0) & (grid_coords < self.box_size)).all(axis=1)
        self.coords = grid_coords[in_box]
        self.features = self.features[in_box] if self.features is not None else None

        pass

    def _insertInFixed3DGrid(self) -> ndarray:
        """
        Merge atom coordinates and features both represented as 2D arrays into one
        fixed-sized 3D box.

        Returns
        -------
        grid: np.ndarray, shape = (M, M, M, F)
            4D array with atom properties distributed in 3D space. M is equal to
            2 * `max_dist` / `resolution` + 1
        """

        num_features = len(self.fe.FEATURE_NAMES)
        grid_shape = (
            (self.box_size, self.box_size, self.box_size, num_features)
            if self.features is not None
            else (self.box_size, self.box_size, self.box_size, 1)
        )

        # init empty grid
        grid = np.zeros(
            grid_shape,
            dtype=np.float32,
        )

        # put atoms features in it's transformed coords
        if self.features is not None:
            for (x, y, z), f in zip(self.coords, self.features):
                grid[x, y, z] += f
        else:
            # put atoms features in it's transformed coords
            for x, y, z in self.coords:
                grid[x, y, z, 0] += 1

        return grid

    def segment_grid_to_pockets(
        self,
        grid_probability_map: ndarray,
        probability_threshold: int = 0.5,
        min_pocket_size: int = 50,
    ):
        """
        Extract predicted pockets from the probability map and saves a 3D grid with labeled pockets.

        Parameters
        ----------

        grid_probability_map: ndarray
            Probability map we get from the model.
        probability_threshold: int
            Atoms are considered sites if thier probability was higher than threshold.
        min_pocket_size: int
            Predicted pockets with size smaller than min_size will be excluded.

        Returns
        ----------
        self: Mol3DGrid
            Density Transformer object after saving 3D grid with labeled predicted pockets in
            grid_labeled_pockets and pockets number.
        """

        if len(grid_probability_map) != 1:
            raise ValueError("Segmentation of more than one molecule is not supported")

        # turn every atom to 1 (site atom) or 0 (not site atom) based on it's probability
        grid_sites = (grid_probability_map[0] > probability_threshold).any(axis=-1)

        # merge close site atoms
        grid_sites = closing(grid_sites)

        # exclude most site atoms that are on grid edges/border
        grid_sites = clear_border(grid_sites)

        # label every pocket of connected site atoms in grid and get how many pockets were there
        grid_labeled_pockets, pockets_num = label(grid_sites, return_num=True)

        # voxel for 3D image is like what a pixel is for a 2D image
        voxel_size = (1 / self.scale) ** 3

        for pocket_label in range(1, pockets_num + 1):
            grid_pocket = grid_labeled_pockets == pocket_label
            scaled_pocket_size = grid_pocket.sum()

            # get number of atoms after reversing scale
            pocket_final_size = scaled_pocket_size * voxel_size

            if pocket_final_size < min_pocket_size:
                # pocket size is very small so exclude those atoms from being site atoms
                grid_labeled_pockets[np.where(grid_pocket)] = 0
                pockets_num -= 1

        self.grid_labeled_pockets = grid_labeled_pockets
        self.pockets_num = pockets_num

        return self

    def get_pockets_mols(self) -> List[pybel.Molecule]:
        """
        Extract labeled pockets from grid and save them as molecules.

        Returns
        ----------
        pockets: List[pybel.Molecule]
            Pybel molecules representing predicted pockets.
        """

        pockets = []

        for pocket_label in range(1, self.pockets_num + 1):
            pocket_atoms_coords = np.argwhere(
                self.grid_labeled_pockets == pocket_label
            ).astype("float32")

            # reverse transformations made to molecule
            pocket_atoms_coords *= self.step
            pocket_atoms_coords += self.origin

            # save pocket atoms atom by atom as a OBMol molecule
            mol = openbabel.OBMol()
            for x, y, z in pocket_atoms_coords:
                a = mol.NewAtom()
                a.SetVector(float(x), float(y), float(z))

            # convert to pybel molecule and save
            p_mol = pybel.Molecule(mol)
            pockets.append(p_mol)

        return pockets
