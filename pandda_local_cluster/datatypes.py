from __future__ import annotations
from typing import *
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

import typing

import numpy as np
import gemmi
import scipy


def try_make(path: Path):
    if not path.exists():
        os.mkdir(str(path))


def mtz_to_path(mtz: gemmi.Mtz, out_dir: Path = Path("/tmp/pandda")) -> Path:
    try_make(out_dir)
    while True:

        token = secrets.token_hex()
        out_path = out_dir / f"{token}.mtz"
        if not out_path.exists():
            mtz.write_to_file(str(out_path))
            return out_path


def path_to_mtz(path: Path) -> gemmi.Mtz:
    mtz = gemmi.read_mtz_file(str(path))

    os.remove(str(path))

    return mtz


def structure_to_path(structure: gemmi.Structure, out_dir: Path = Path("/tmp/pandda")) -> Path:
    try_make(out_dir)

    while True:
        token = secrets.token_hex()

        out_path = out_dir / f"{token}.pdb"

        if not out_path.exists():
            structure.write_minimal_pdb(str(out_path))

        return out_path


def path_to_structure(path: Path) -> gemmi.Structure:
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()

    os.remove(str(path))

    return structure


@dataclass
class StructureFactors:
    f: str
    phi: str


# class StructureWrapper:
#     structure: gemmi.Structure
#
#     def __getstate__(self):
#

class Dataset:

    def __init__(self,
                 dtag: str,
                 structure: gemmi.Structure,
                 reflections: gemmi.Mtz,
                 structure_path: Path,
                 reflections_path: Path,
                 fragment_path: Optional[Path],
                 fragment_structures: Optional[MutableMapping[int, gemmi.Structure]],
                 smoothing_factor: Optional[float] = None,
                 ):
        self.dtag = dtag
        self.structure = structure
        self.reflections = reflections
        self.structure_path = structure_path
        self.reflections_path = reflections_path
        self.fragment_path = fragment_path
        self.fragment_structures = fragment_structures
        self.smoothing_factor = smoothing_factor

    def __getstate__(self):
        if self.fragment_path:
            fragment_path = self.fragment_path
        else:
            fragment_path = None

        if self.fragment_structures:
            fragment_structures = {
                idx: structure_to_path(structure)
                for idx, structure
                in self.fragment_structures.items()
            }
        else:
            fragment_structures = None

        state = {
            "dtag": self.dtag,
            "structure": structure_to_path(self.structure),
            "reflections": mtz_to_path(self.reflections),
            "structure_path": self.structure_path,
            "reflections_path": self.reflections_path,
            "fragment_path": self.fragment_path,
            "fragment_structures": fragment_structures,
            "smoothing_factor": self.smoothing_factor,
        }

        return state

    def __setstate__(self, state):
        self.dtag = state["dtag"]
        self.structure = path_to_structure(state["structure"])
        self.reflections = path_to_mtz(state["reflections"])
        self.structure_path = state["structure_path"]
        self.reflections_path = state["reflections_path"]
        self.fragment_path = state["fragment_path"]
        if state["fragment_structures"]:
            self.fragment_structures = {
                idx: path_to_structure(path)
                for idx, path
                in state["fragment_structures"].items()
            }
        else:
            self.fragment_structures = state["fragment_structures"]
        self.smoothing_factor = state["smoothing_factor"]


Datasets = MutableMapping[str, Dataset]


@dataclass()
class Data:
    datasets: Dict


@dataclass()
class Event:
    centroid: Tuple[float, float, float]
    size: int


@dataclass()
class AffinityEvent:
    dtag: str
    marker: Marker
    correlation: float


@dataclass()
class AffinityMaxima:
    index: Tuple[int, int, int]
    correlation: float
    rotation_index: Tuple[float, float, float]
    position: Tuple[float, float, float]
    bdc: float
    mean_map_correlation: float
    mean_map_max_correlation: float
    max_delta_correlation: float
    # centroid: Tuple[float, float, float]


@dataclass()
class ResidueID:
    model: str
    chain: str
    insertion: str

    @staticmethod
    def from_residue_chain(model: gemmi.Model, chain: gemmi.Chain, res: gemmi.Residue):
        return ResidueID(model.name,
                         chain.name,
                         str(res.seqid.num),
                         )

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return ((self.model, self.chain, self.insertion) ==
                    (other.model, other.chain, other.insertion))
        return NotImplemented

    def __hash__(self):
        return hash((self.model, self.chain, self.insertion))


@dataclass()
class Marker:
    x: float
    y: float
    z: float
    resid: Optional[ResidueID]

    def __hash__(self):
        if self.resid:
            return hash((self.resid.model, self.resid.chain, self.resid.insertion,))
        else:
            return hash((self.x, self.y, self.z,))


@dataclass()
class DatasetResults:
    dtag: str
    residue_id: ResidueID
    structure_path: Path
    reflections_path: Path
    fragment_path: Path
    events: Dict[int, Event] = field(default_factory=dict)
    comparators: List[Dataset] = field(default_factory=list)


@dataclass()
class DatasetAffinityResults:
    dtag: str
    residue_id: ResidueID
    structure_path: Path
    reflections_path: Path
    fragment_path: Path
    events: Dict[int, AffinityEvent] = field(default_factory=dict)
    comparators: List[Dataset] = field(default_factory=list)


# @dataclass()
# class ResidueAffinityResults(MutableMapping[str, DatasetResults]):
#     _dataset_results: Dict[str, DatasetAffinityResults] = field(default_factory=dict)


@dataclass()
class ResidueResults(MutableMapping[str, DatasetResults]):
    _dataset_results: Dict[str, DatasetResults] = field(default_factory=dict)


@dataclass()
class PanDDAResults(MutableMapping[ResidueID, ResidueResults]):
    _pandda_results: Dict[ResidueID, ResidueResults] = field(default_factory=dict)


class Transform:

    def __init__(self, transform: gemmi.Transform):
        self.transform: gemmi.Transform = transform

    def __getstate__(self):
        state = {
            "mat": self.transform.mat.tolist(),
            "vec": self.transform.vec.tolist(),
        }

        return state

    def __setstate__(self, state):
        self.transform = gemmi.Transform()
        self.transform.mat.fromlist(state["mat"])
        self.transform.vec.fromlist(state["vec"])


@dataclass()
class TransformGlobal:
    transform: gemmi.Transform
    com_reference: np.array
    com_moving: np.array

    def apply_moving_to_reference(self, positions: typing.Dict[typing.Tuple[int], gemmi.Position]) -> typing.Dict[
        typing.Tuple[int], gemmi.Position]:
        transformed_positions = {}
        for index, position in positions.items():
            rotation_frame_position = gemmi.Position(position[0] - self.com_moving[0],
                                                     position[1] - self.com_moving[1],
                                                     position[2] - self.com_moving[2])
            transformed_vector = self.transform.apply(rotation_frame_position)

            transformed_positions[index] = gemmi.Position(transformed_vector[0] + self.com_reference[0],
                                                          transformed_vector[1] + self.com_reference[1],
                                                          transformed_vector[2] + self.com_reference[2])

        return transformed_positions

    def apply_reference_to_moving(self, positions: typing.Dict[typing.Tuple[int], gemmi.Position]) -> typing.Dict[
        typing.Tuple[int], gemmi.Position]:
        inverse_transform = self.transform.inverse()
        transformed_positions = {}
        for index, position in positions.items():
            rotation_frame_position = gemmi.Position(position[0] - self.com_reference[0],
                                                     position[1] - self.com_reference[1],
                                                     position[2] - self.com_reference[2])
            transformed_vector = inverse_transform.apply(rotation_frame_position)

            transformed_positions[index] = gemmi.Position(transformed_vector[0] + self.com_moving[0],
                                                          transformed_vector[1] + self.com_moving[1],
                                                          transformed_vector[2] + self.com_moving[2])

        return transformed_positions

    @staticmethod
    def from_translation_rotation(translation, rotation, com_reference, com_moving):
        transform = gemmi.Transform()
        transform.vec.fromlist(translation.tolist())
        transform.mat.fromlist(rotation.as_matrix().tolist())

        return TransformGlobal(transform, com_reference, com_moving)

    @staticmethod
    def from_residues(previous_res, current_res, next_res, previous_ref, current_ref, next_ref):
        previous_ca_pos = previous_res["CA"][0].pos
        current_ca_pos = current_res["CA"][0].pos
        next_ca_pos = next_res["CA"][0].pos

        previous_ref_ca_pos = previous_ref["CA"][0].pos
        current_ref_ca_pos = current_ref["CA"][0].pos
        next_ref_ca_pos = next_ref["CA"][0].pos

        matrix = np.array([
            TransformGlobal.pos_to_list(previous_ca_pos),
            TransformGlobal.pos_to_list(current_ca_pos),
            TransformGlobal.pos_to_list(next_ca_pos),
        ])
        matrix_ref = np.array([
            TransformGlobal.pos_to_list(previous_ref_ca_pos),
            TransformGlobal.pos_to_list(current_ref_ca_pos),
            TransformGlobal.pos_to_list(next_ref_ca_pos),
        ])

        mean = np.mean(matrix, axis=0)
        mean_ref = np.mean(matrix_ref, axis=0)

        # vec = mean_ref - mean
        vec = np.array([0.0, 0.0, 0.0])

        de_meaned = matrix - mean
        de_meaned_ref = matrix_ref - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        com_reference = mean_ref
        com_moving = mean

        return TransformGlobal.from_translation_rotation(vec, rotation, com_reference, com_moving)

    @staticmethod
    def pos_to_list(pos: gemmi.Position):
        return [pos[0], pos[1], pos[2]]

    @staticmethod
    def from_start_residues(current_res, next_res, current_ref, next_ref):
        current_ca_pos = current_res["CA"][0].pos
        next_ca_pos = next_res["CA"][0].pos

        current_ref_ca_pos = current_ref["CA"][0].pos
        next_ref_ca_pos = next_ref["CA"][0].pos

        matrix = np.array([
            TransformGlobal.pos_to_list(current_ca_pos),
            TransformGlobal.pos_to_list(next_ca_pos),
        ])
        matrix_ref = np.array([
            TransformGlobal.pos_to_list(current_ref_ca_pos),
            TransformGlobal.pos_to_list(next_ref_ca_pos),
        ])

        mean = np.mean(matrix, axis=0)
        mean_ref = np.mean(matrix_ref, axis=0)

        # vec = mean_ref - mean
        vec = np.array([0.0, 0.0, 0.0])

        de_meaned = matrix - mean
        de_meaned_ref = matrix_ref - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        com_reference = mean_ref

        com_moving = mean

        return TransformGlobal.from_translation_rotation(vec, rotation, com_reference, com_moving)

    @staticmethod
    def from_atoms(dataset_selection,
                   reference_selection,
                   com_dataset,
                   com_reference,
                   ):

        # mean = np.mean(dataset_selection, axis=0)
        # mean_ref = np.mean(reference_selection, axis=0)
        mean = np.array(com_dataset)
        mean_ref = np.array(com_reference)

        # vec = mean_ref - mean
        vec = np.array([0.0, 0.0, 0.0])

        de_meaned = dataset_selection - mean
        de_meaned_ref = reference_selection - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        com_reference = mean_ref

        com_moving = mean

        return TransformGlobal.from_translation_rotation(vec, rotation, com_reference, com_moving)

    @staticmethod
    def from_finish_residues(previous_res, current_res, previous_ref, current_ref):
        previous_ca_pos = previous_res["CA"][0].pos
        current_ca_pos = current_res["CA"][0].pos

        previous_ref_ca_pos = previous_ref["CA"][0].pos
        current_ref_ca_pos = current_ref["CA"][0].pos

        matrix = np.array([
            TransformGlobal.pos_to_list(previous_ca_pos),
            TransformGlobal.pos_to_list(current_ca_pos),
        ])
        matrix_ref = np.array([
            TransformGlobal.pos_to_list(previous_ref_ca_pos),
            TransformGlobal.pos_to_list(current_ref_ca_pos),
        ])

        mean = np.mean(matrix, axis=0)
        mean_ref = np.mean(matrix_ref, axis=0)

        # vec = mean_ref - mean
        vec = np.array([0.0, 0.0, 0.0])

        de_meaned = matrix - mean
        de_meaned_ref = matrix_ref - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        com_reference = mean_ref

        com_moving = mean

        return TransformGlobal.from_translation_rotation(vec, rotation, com_reference, com_moving)


# @dataclass()
# class Alignment(MutableMapping[ResidueID, Transform]):
#     _residue_alignments: MutableMapping[ResidueID, Transform] = field(default_factory=dict)


Alignment = MutableMapping[Marker, Optional[Transform]]


@dataclass()
class Cluster(object):
    _indexes: Tuple[np.ndarray, np.ndarray, np.ndarray]

    def size(self):
        return len(self._indexes)


@dataclass()
class DatasetAffinityResults:
    dtag: str
    marker: Marker
    structure_path: Path
    reflections_path: Path
    fragment_path: Path
    maxima: AffinityMaxima
    # events: Dict[int, AffinityEvent] = field(default_factory=dict)
    # comparators: List[Dataset] = field(default_factory=list)


#
# @dataclass()
# class ResidueAffinityResults(MutableMapping[str, DatasetAffinityResults]):
#     _dataset_results: Dict[str, DatasetAffinityResults] = field(default_factory=dict)

MarkerAffinityResults = MutableMapping[str, DatasetAffinityResults]

# @dataclass()
# class PanDDAAffinityResults(MutableMapping[ResidueID, ResidueAffinityResults]):
#     _pandda_results: Dict[ResidueID, ResidueAffinityResults] = field(default_factory=dict)


PanDDAAffinityResults = MutableMapping[Marker, MarkerAffinityResults]


class Params:
    debug: bool = True

    # Data loading
    structure_regex: str = "*.pdb"
    reflections_regex: str = "*.mtz"
    smiles_regex: str = "*.smiles"

    # Diffraction handling
    structure_factors: StructureFactors = StructureFactors("FWT", "PHWT")
    sample_rate: float = 3.0

    # Grid sampling
    grid_size: int = 32
    grid_spacing: float = 0.5

    # Dataset clusterings
    local_cluster_cutoff: float = 0.6
    global_cluster_cutoff: float = 0.9
    min_dataset_cluster_size: int = 60

    # output
    output_smoothed_mtzs: bool = True
    database_file: str = "database.sqlite"

    def update(self, **kwargs):
        for key, value in kwargs.items():

            if key == "debug":
                self.debug = value

            # Regexes
            elif key == "structure_regex":
                self.structure_regex = value

            elif key == "reflections_regex":
                self.reflections_regex = value

            # Diffraction handling
            elif key == "structure_factors":
                self.structure_factors = StructureFactors(*value.split(","))
            elif key == "sample_rate":
                self.sample_rate = value

            # Grid samping
            elif key == "grid_size":
                self.grid_size = value
            elif key == "grid_spacing":
                self.grid_spacing = value

            # Dataset clusterings
            elif key == "strong_density_cluster_cutoff":
                self.strong_density_cluster_cutoff = value
            elif key == "min_dataset_cluster_size":
                self.min_dataset_cluster_size = value
            elif key == "cutoff":
                self.local_cluster_cutoff = value

            # Fragment searching
            elif key == "num_fragment_pose_samples":
                self.num_fragment_pose_samples = value
            elif key == "min_correlation":
                self.min_correlation = value
            elif key == "pruning_threshold":
                self.pruning_threshold = value

            # output
            elif key == "output_smoothed_mtz":
                self.output_smoothed_mtzs = value

            # Unknown argument handling
            else:
                raise Exception(f"Unknown paramater: {key} = {value}")
