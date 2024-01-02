from google.protobuf import proto_builder
from openbabel import pybel
import numpy as np
from numpy import ndarray
from typing import Any, List, Dict, Tuple


class FeatureExtractor:
    def __init__(self) -> None:
        self.FEATURE_NAMES: List[str] = []
        """Names of all features in the correct order"""

        self.NAMED_PROPS: List[str] = []
        """Pybel atoms properties to save as atoms features"""

        self.ATOM_CODES: Dict[str, int] = {}
        """Atoms Codes for one-hot encoding"""

        self.NUM_ATOM_CLASSES: int = 0
        """number of atoms classes"""

        self.SMARTS: List[str] = []
        """SMARTS to look for"""

        self.__PATTERNS: List[str] = []
        """All possible Patterns of self.SMARTS"""

        self._setup_atoms_codes()
        self._setup_named_props()
        self._setup_smarts()

    def get_all_coords(self, mol: pybel.Molecule) -> ndarray:
        if not isinstance(mol, pybel.Molecule):
            raise TypeError(
                "mol should be a pybel.Molecule object, got %s " "instead" % type(mol)
            )

        coords = []

        for atom in mol:
            coords.append(atom.coords)

        return coords

    def get_feature(self, protein_mol: pybel.Molecule) -> Tuple[ndarray, ndarray]:
        if not isinstance(protein_mol, pybel.Molecule):
            raise TypeError(
                "mol should be a pybel.Molecule object, got %s "
                "instead" % type(protein_mol)
            )

        coords = []
        features = []
        heavy_atoms = []

        for i, atom in enumerate(protein_mol):
            # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
            if atom.atomicnum > 1:
                heavy_atoms.append(i)
                coords.append(atom.coords)

                atom_bin_code = self._encode_num(atom.atomicnum)
                important_properties = [
                    atom.__getattribute__(prop) for prop in self.NAMED_PROPS
                ]

                atom_features = np.concatenate((atom_bin_code, important_properties))
                features.append(atom_features)

        try:
            coords = np.asarray(coords, dtype=np.float32)

            c_shape = coords.shape
            if len(c_shape) != 2 or c_shape[1] != 3:
                raise ValueError()
        except ValueError:
            raise ValueError("coords must be an array of floats of shape (N, 3)")

        try:
            features = np.asarray(features, dtype=np.float32)

            f_shape = features.shape
            if len(f_shape) != 2 or f_shape[0] != len(coords):
                raise ValueError()
        except ValueError:
            raise ValueError("features must be an array of floats of shape (N, F)")

        features = np.hstack([features, self._find_smarts(protein_mol)[heavy_atoms]])
        if np.isnan(features).any():
            raise RuntimeError("Got NaN when calculating features")

        return coords, features

    def _setup_atoms_codes(self):
        metals_atoms_codes = (
            [3, 4, 11, 12, 13]
            + list(range(19, 32))
            + list(range(37, 51))
            + list(range(55, 84))
            + list(range(87, 104))
        )

        # List of tuples (atomic_num, class_name) with atom types to encode.
        atoms_codes_classes = [
            (5, "B"),
            (6, "C"),
            (7, "N"),
            (8, "O"),
            (15, "P"),
            (16, "S"),
            (34, "Se"),
            ([9, 17, 35, 53], "halogen"),
            (metals_atoms_codes, "metal"),
        ]

        for index, (atom_codes, class_name) in enumerate(atoms_codes_classes):
            if type(atom_codes) is list:
                for a in atom_codes:
                    self.ATOM_CODES[a] = index
            else:
                self.ATOM_CODES[atom_codes] = index
            self.FEATURE_NAMES.append(class_name)

        self.NUM_ATOM_CLASSES = len(atoms_codes_classes)

    def _setup_named_props(self):
        # pybel.Atom properties to save
        self.NAMED_PROPS = ["hyb", "heavydegree", "heterodegree", "partialcharge"]
        self.FEATURE_NAMES += self.NAMED_PROPS

    def _setup_smarts(self):
        # SMARTS definition for other properties
        self.SMARTS = [
            "[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]",
            "[a]",
            "[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]",
            "[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]",
            "[r]",
        ]
        smarts_labels = ["hydrophobic", "aromatic", "acceptor", "donor", "ring"]

        # Compile patterns
        self.__PATTERNS = []
        for smarts in self.SMARTS:
            self.__PATTERNS.append(pybel.Smarts(smarts))
        self.FEATURE_NAMES += smarts_labels

    def _encode_num(self, atomic_num):
        """One-Hot encoding for atoms types"""
        if not isinstance(atomic_num, int):
            raise TypeError(
                "Atomic number must be int, %s was given" % type(atomic_num)
            )

        encoding = np.zeros(self.NUM_ATOM_CLASSES)
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding

    def _find_smarts(self, molecule) -> ndarray:
        """Find atoms that match SMARTS patterns."""

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError(
                "molecule must be pybel.Molecule object, %s was given" % type(molecule)
            )

        features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))

        for pattern_id, pattern in enumerate(self.__PATTERNS):
            atoms_with_prop = (
                np.array(list(*zip(*pattern.findall(molecule))), dtype=int) - 1
            )
            features[atoms_with_prop, pattern_id] = 1.0
        return features
