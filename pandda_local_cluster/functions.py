import json

# 3rd party
import numpy as np
import scipy
from scipy import spatial as spsp, cluster as spc
import pandas as pd
import gemmi
import joblib
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import hdbscan
from scipy.cluster import hierarchy
from sklearn import decomposition
from sklearn import manifold
import umap
from bokeh.plotting import ColumnDataSource, figure, output_file, show

# Custom
from pandda_local_cluster.datatypes import *


def try_make(path: Path):
    if not path.exists():
        os.mkdir(str(path))


def mtz_to_path(mtz: gemmi.Mtz, out_dir: Path) -> Path:
    return None


def python_to_mtz(path: Path) -> gemmi.Mtz:
    return None


def structure_to_python(structure: gemmi.Structure, out_dir: Path) -> Path:
    return None


def python_to_structure(path: Path) -> gemmi.Structure:
    return None


def get_residue_id(model: gemmi.Model, chain: gemmi.Chain, insertion: str):
    return ResidueID(model.name, chain.name, str(insertion))


def get_residue(structure: gemmi.Structure, residue_id: ResidueID) -> gemmi.Residue:
    return structure[residue_id.model][residue_id.chain][residue_id.insertion][0]


def get_comparator_datasets(
        linkage: np.ndarray,
        dataset_clusters: np.ndarray,
        dataset_index: int,
        target_dtag: str,
        apo_mask: np.ndarray,
        datasets: MutableMapping[str, Dataset],
        min_cluster_size: int,
        num_datasets: int,
) -> Optional[MutableMapping[str, Dataset]]:
    #
    apo_cluster_indexes: np.ndarray = np.unique(dataset_clusters[apo_mask])

    apo_clusters: MutableMapping[int, np.ndarray] = {}
    for apo_cluster_index in apo_cluster_indexes:
        cluster: np.ndarray = dataset_clusters[dataset_clusters == apo_cluster_index]

        #
        if cluster.size > min_cluster_size:
            apo_clusters[apo_cluster_index] = cluster

    if len(apo_clusters) == 0:
        return None

    # Get the cophenetic distances
    cophenetic_distances_reduced: np.ndarray = scipy.cluster.hierarchy.cophenet(linkage)
    cophenetic_distances: np.ndarray = spsp.distance.squareform(cophenetic_distances_reduced)
    distances_from_dataset: np.ndarray = cophenetic_distances[dataset_index, :]

    # Find the closest apo cluster
    cluster_distances: MutableMapping[int, float] = {}
    for apo_cluster_index, apo_cluster in apo_clusters.items():
        distances_from_cluster: np.ndarray = distances_from_dataset[dataset_clusters == apo_cluster_index]
        mean_distance: float = np.mean(distances_from_cluster)
        cluster_distances[apo_cluster_index] = mean_distance

    # Find closest n datasets in cluster
    closest_cluster_index: int = min(cluster_distances, key=lambda x: cluster_distances[x])
    closest_cluster_dtag_array: np.ndarray = np.array(list(datasets.keys()))[dataset_clusters == closest_cluster_index]

    # Sort by resolution
    closest_cluster_dtag_resolutions = {dtag: datasets[dtag].reflections.resolution_high()
                                        for dtag in closest_cluster_dtag_array}
    print(f"Got {len(closest_cluster_dtag_resolutions)} comparatprs")
    sorted_resolution_dtags = sorted(closest_cluster_dtag_resolutions,
                                     key=lambda dtag: closest_cluster_dtag_resolutions[dtag])
    resolution_cutoff = max(datasets[target_dtag].reflections.resolution_high(),
                            datasets[sorted_resolution_dtags[
                                min(len(sorted_resolution_dtags), num_datasets)]].reflections.resolution_high()
                            )
    # sorted_resolution_dtags_cutoff = [dtag for dtag in sorted_resolution_dtags if datasets[dtag].reflections.resolution_high() < resolution_cutoff]

    # highest_resolution_dtags = sorted_resolution_dtags_cutoff[-min(len(sorted_resolution_dtags), num_datasets):]

    closest_cluster_datasets: MutableMapping[str, Dataset] = {dtag: datasets[dtag]
                                                              for dtag
                                                              in closest_cluster_dtag_array
                                                              if datasets[
                                                                  dtag].reflections.resolution_high() < resolution_cutoff
                                                              }
    print(closest_cluster_datasets)

    return closest_cluster_datasets


def iterate_residues(
        datasets: MutableMapping[str, Dataset],
        reference: Dataset,
        debug: bool = True,
) -> Iterator[Tuple[ResidueID, MutableMapping[str, Dataset]]]:
    # Get all unique ResidueIDs from all datasets
    # Order them: Sort by model, then chain, then residue insertion
    # yield them

    reference_structure: gemmi.Structure = reference.structure

    for model in reference_structure:
        for chain in model:
            for residue in chain.get_polymer():
                residue_id: ResidueID = ResidueID(model.name, chain.name, str(residue.seqid.num))

                residue_datasets: MutableMapping[str, Dataset] = {}
                for dtag, dataset in datasets.items():
                    structure: gemmi.Structure = dataset.structure

                    try:
                        res = get_residue(structure, residue_id)
                        res_ca = res["CA"][0]
                        residue_datasets[dtag] = dataset
                    except Exception as e:
                        if debug:
                            print(e)
                        continue

                yield residue_id, residue_datasets


def iterate_markers(
        datasets: MutableMapping[str, Dataset],
        markers: List[Marker],
        alignments: MutableMapping[str, Alignment],
        debug: bool = True,
) -> Iterator[Tuple[Marker, MutableMapping[str, Dataset]]]:
    for marker in markers:

        marker_datasets = {}
        for dtag, dataset in datasets.items():
            if alignments[dtag][marker] is not None:
                marker_datasets[dtag] = dataset

        yield marker, marker_datasets


def get_comparator_samples(
        sample_arrays: MutableMapping[str, np.ndarray],
        comparator_datasets: MutableMapping[str, Dataset],
) -> MutableMapping[str, np.ndarray]:
    comparator_samples: MutableMapping[str, np.ndarray] = {}
    for dtag in comparator_datasets:
        comparator_samples[dtag] = sample_arrays[dtag]

    return comparator_samples


def get_path_from_regex(directory: Path, regex: str) -> Optional[Path]:
    for path in directory.glob(f"{regex}"):
        return path

    else:
        return None


def get_structure(structure_path: Path) -> gemmi.Structure:
    structure: gemmi.Structure = gemmi.read_structure(str(structure_path))
    structure.setup_entities()
    return structure


def get_reflections(reflections_path: Path) -> gemmi.Mtz:
    reflections: gemmi.Mtz = gemmi.read_mtz_file(str(reflections_path))
    return reflections


def get_dataset_from_dir(
        directory: Path,
        structure_regex: str,
        reflections_regex: str,
        smiles_regex: str,
        pruning_threshold: float,
        debug: bool = True,
) -> Optional[Dataset]:
    if debug:
        print(f"\tChecking directoy {directory} for data...")

    if directory.is_dir():
        if debug:
            print(
                f"\t\t{directory} is a directory. Checking for regexes: {structure_regex}, {reflections_regex} and {smiles_regex}")
        dtag = directory.name
        structure_path: Optional[Path] = get_path_from_regex(directory, structure_regex)
        reflections_path: Optional[Path] = get_path_from_regex(directory, reflections_regex)
        smiles_path: Optional[Path] = get_path_from_regex(directory, smiles_regex)

        if structure_path and reflections_path:

            fragment_structures = None

            dataset: Dataset = Dataset(
                dtag=dtag,
                structure=get_structure(structure_path),
                reflections=get_reflections(reflections_path),
                structure_path=structure_path,
                reflections_path=reflections_path,
                fragment_path=smiles_path,
                fragment_structures=fragment_structures,
            )

            # if debug:
            # if (structure_factors.f not in dataset.reflections.column_labels()) or (structure_factors.phi not in dataset.reflections.column_labels()):
            #     print(f"\t\t{directory} Lacks structure factors. Skipping")
            #     return None

            return dataset

        else:
            if debug:
                print(f"\t\t{directory} Lacks either a structure or reflections. Skipping")
            return None
    else:
        return None


def get_datasets(
        data_dir: Path,
        structure_regex: str,
        reflections_regex: str,
        smiles_regex: str,
        pruning_threshold: float,
        debug: bool = True,
) -> MutableMapping[str, Dataset]:
    # Iterate over the paths
    directories = list(data_dir.glob("*"))

    datasets_list: List[Optional[Dataset]] = joblib.Parallel(
        verbose=50,
        n_jobs=-1,
        # backend="multiprocessing",
    )(
        joblib.delayed(
            get_dataset_from_dir)(
            directory,
            structure_regex,
            reflections_regex,
            smiles_regex,
            pruning_threshold,
            debug,
        )
        for directory
        in directories

    )

    datasets: MutableMapping[str, Dataset] = {dataset.dtag: dataset for dataset in datasets_list if dataset is not None}

    return datasets


def truncate_resolution(reflections: gemmi.Mtz, resolution: float) -> gemmi.Mtz:
    new_reflections = gemmi.Mtz(with_base=False)

    # Set dataset properties
    new_reflections.spacegroup = reflections.spacegroup
    new_reflections.set_cell_for_all(reflections.cell)

    # Add dataset
    new_reflections.add_dataset("truncated")

    # Add columns
    for column in reflections.columns:
        new_reflections.add_column(column.label, column.type)

    # Get data
    data_array = np.array(reflections, copy=True)
    data = pd.DataFrame(data_array,
                        columns=reflections.column_labels(),
                        )
    data.set_index(["H", "K", "L"], inplace=True)

    # add resolutions
    data["res"] = reflections.make_d_array()

    # Truncate by resolution
    data_truncated = data[data["res"] >= resolution]

    # Rem,ove res colum
    data_dropped = data_truncated.drop("res", "columns")

    # To numpy
    data_dropped_array = data_dropped.to_numpy()

    # new data
    new_data = np.hstack([data_dropped.index.to_frame().to_numpy(),
                          data_dropped_array,
                          ]
                         )

    # Update
    new_reflections.set_data(new_data)

    # Update resolution
    new_reflections.update_reso()

    return new_reflections


def get_truncated_datasets(datasets: MutableMapping[str, Dataset],
                           reference_dataset: Dataset,
                           structure_factors: StructureFactors) -> MutableMapping[str, Dataset]:
    resolution_truncated_datasets = {}

    # Get the lowest common resolution
    resolution: float = max([dataset.reflections.resolution_high() for dtag, dataset in datasets.items()])

    # Truncate by common resolution
    for dtag, dataset in datasets.items():
        dataset_reflections: gemmi.Mtz = dataset.reflections
        truncated_reflections: gemmi.Mtz = truncate_resolution(dataset_reflections, resolution)
        truncated_dataset: Dataset = Dataset(dataset.dtag,
                                             dataset.structure,
                                             truncated_reflections,
                                             dataset.structure_path,
                                             dataset.reflections_path,
                                             dataset.fragment_path,
                                             dataset.fragment_structures,
                                             dataset.smoothing_factor,
                                             )

        resolution_truncated_datasets[dtag] = truncated_dataset

    return resolution_truncated_datasets


def transform_from_translation_rotation(translation, rotation):
    transform = gemmi.Transform()
    transform.vec.fromlist(translation.tolist())
    transform.mat.fromlist(rotation.as_matrix().tolist())

    return Transform(transform)


def get_transform_from_atoms(
        moving_selection,
        reference_selection,
) -> Transform:
    """
    Get the transform FROM the moving TO the reference
    :param moving_selection:
    :param reference_selection:
    :return:
    """

    # Get the means
    mean_moving = np.mean(moving_selection, axis=0)
    mean_reference = np.mean(reference_selection, axis=0)

    # Het the transation FROM the moving TO the reference
    vec = np.array(mean_reference - mean_moving)

    de_meaned_moving = moving_selection - mean_moving
    de_meaned_referecnce = reference_selection - mean_reference

    rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned_referecnce, de_meaned_moving)

    return transform_from_translation_rotation(vec, rotation)


def get_markers(
        reference_dataset: Dataset,
        markers: Optional[List[Tuple[float, float, float]]],
        debug: bool = True,
) -> List[Marker]:
    new_markers: List[Marker] = []

    if markers:
        for marker in markers:
            new_markers.append(
                Marker(
                    marker[0],
                    marker[1],
                    marker[2],
                    None,
                )
            )
        return new_markers

    else:
        for model in reference_dataset.structure:
            for chain in model:
                for ref_res in chain.get_polymer():
                    print(f"\t\tGetting transform for residue: {ref_res}")

                    # if ref_res.name.upper() not in Constants.residue_names:
                    #     continue
                    try:

                        # Get ca pos from reference
                        current_res_id: ResidueID = get_residue_id(model, chain, ref_res.seqid.num)
                        reference_ca_pos = ref_res["CA"][0].pos
                        new_markers.append(
                            Marker(
                                reference_ca_pos.x,
                                reference_ca_pos.y,
                                reference_ca_pos.z,
                                current_res_id,
                            )
                        )

                    except Exception as e:
                        if debug:
                            print(f"\t\tAlignment exception: {e}")
                        continue

        if debug:
            print(f"Found {len(new_markers)}: {new_markers}")

        return new_markers


def get_alignment(
        reference: Dataset,
        dataset: Dataset,
        markers: List[Marker],
        debug: bool = True,
) -> Alignment:
    # Find the common atoms as an array
    dataset_pos_list = []
    reference_pos_list = []
    for model in reference.structure:
        for chain in model:
            for ref_res in chain.get_polymer():
                # if ref_res.name.upper() not in Constants.residue_names:
                #     continue
                try:

                    # Get ca pos from reference
                    print("Getting ref ca")
                    current_res_id: ResidueID = get_residue_id(model, chain, ref_res.seqid.num)
                    print(type(ref_res))
                    print(ref_res)
                    reference_ca_pos = ref_res["CA"][0].pos

                    print("Getting dataset ca")
                    # Get the ca pos from the dataset
                    dataset_res = get_residue(dataset.structure, current_res_id)
                    print(type(dataset_res))
                    print(dataset_res)
                    dataset_ca_pos = dataset_res["CA"][0].pos
                except Exception as e:
                    if debug:
                        print(f"\t\tAlignment exception: {e}")
                    continue

                # residue_id: ResidueID = get_residue_id(model, chain, ref_res.seqid.num)
                #
                # dataset_res: gemmi.Residue = get_residue(dataset.structure, residue_id)

                reference_pos_list.append([reference_ca_pos.x, reference_ca_pos.y, reference_ca_pos.z])
                dataset_pos_list.append([dataset_ca_pos.x, dataset_ca_pos.y, dataset_ca_pos.z])

                try:

                    # Get ca pos from reference
                    print("Getting ref cb")
                    current_res_id: ResidueID = get_residue_id(model, chain, ref_res.seqid.num)
                    print(type(ref_res))
                    print(ref_res)
                    reference_cb_pos = ref_res["CB"][0].pos

                    print("Getting dataset cb")
                    # Get the ca pos from the dataset
                    dataset_res = get_residue(dataset.structure, current_res_id)
                    print(type(dataset_res))
                    print(dataset_res)
                    dataset_cb_pos = dataset_res["CB"][0].pos
                except Exception as e:
                    if debug:
                        print(f"\t\tAlignment exception: {e}")
                    continue

                # residue_id: ResidueID = get_residue_id(model, chain, ref_res.seqid.num)
                #
                # dataset_res: gemmi.Residue = get_residue(dataset.structure, residue_id)

                reference_pos_list.append([reference_cb_pos.x, reference_cb_pos.y, reference_cb_pos.z])
                dataset_pos_list.append([dataset_cb_pos.x, dataset_cb_pos.y, dataset_cb_pos.z])

    dataset_atom_array = np.array(dataset_pos_list)
    reference_atom_array = np.array(reference_pos_list)

    if debug:
        print(f"\t\tdataset atom array size: {dataset_atom_array.shape}")
        print(f"\t\treference atom array size: {reference_atom_array.shape}")

    # dataset kdtree
    reference_tree = spsp.KDTree(reference_atom_array)

    # Get the transform for each

    alignment: Alignment = {}
    for marker in markers:
        # dataset selection
        if debug:
            print("\t\tQuerying")

        reference_indexes = reference_tree.query_ball_point(
            [marker.x, marker.y, marker.z],
            7.0,
        )
        dataset_selection = dataset_atom_array[reference_indexes]

        # Reference selection
        reference_selection = reference_atom_array[reference_indexes]

        if dataset_selection.shape[0] < 4:
            print(f"No matching atoms near this marker! Skipping!")
            alignment[marker] = None
            continue

        # Get transform
        if debug:
            print("\t\tGetting transform")
        alignment[marker] = get_transform_from_atoms(
            dataset_selection,
            reference_selection,
        )
        if debug:
            print(
                (
                    f"\t\t\tTransform is:\n"
                    f"\t\t\t\tMat: {alignment[marker].transform.mat}\n"
                    f"\t\t\t\tVec: {alignment[marker].transform.vec}\n"
                )
            )

    if debug:
        print("Returning alignment...")
    return alignment


def get_alignments(
        datasets: MutableMapping[str, Dataset],
        reference: Dataset,
        markers: List[Marker],
        debug: bool = True,
) -> MutableMapping[str, Alignment]:
    alignment_list: List[Alignment] = joblib.Parallel(
        verbose=50,
        n_jobs=-1,
        # backend="multiprocessing",
    )(
        joblib.delayed(get_alignment)(
            reference,
            dataset,
            markers,
        )
        for dataset
        in list(datasets.values())
    )

    alignments: MutableMapping[str, Alignment] = {
        dtag: alignment
        for dtag, alignment
        in zip(list(datasets.keys()), alignment_list)
    }

    return alignments


def sample_dataset(
        dataset: Dataset,
        transform: Transform,
        marker: Marker,
        structure_factors: StructureFactors,
        sample_rate: float,
        grid_size: int,
        grid_spacing: float,
) -> np.ndarray:
    reflections: gemmi.Mtz = dataset.reflections
    unaligned_xmap: gemmi.FloatGrid = reflections.transform_f_phi_to_map(
        structure_factors.f,
        structure_factors.phi,
        sample_rate=sample_rate,
    )

    unaligned_xmap_array = np.array(unaligned_xmap, copy=False)

    std = np.std(unaligned_xmap_array)
    unaligned_xmap_array[:, :, :] = unaligned_xmap_array[:, :, :] / std

    transform_inverse = transform.transform.inverse()

    transform_vec = -np.array(transform.transform.vec.tolist())
    print(f"transform vector: {transform_vec}")

    transform_mat = np.array(transform_inverse.mat.tolist())
    print(f"transform matrix: {transform_mat}")

    transform_mat = np.matmul(transform_mat, np.eye(3) * grid_spacing)
    print(f"transform matrix scaled: {transform_mat}")

    offset = np.array([grid_size / 2, grid_size / 2, grid_size / 2]).reshape(3, 1)
    print(f"offset: {offset}")

    rotated_offset = np.matmul(transform_mat, offset).flatten()
    print(f"rotated_offset: {rotated_offset}")

    dataset_centroid = np.array([marker.x, marker.y, marker.z]) + transform_vec
    print(f"dataset_centroid: {dataset_centroid}")

    dataset_centroid_offset = dataset_centroid - rotated_offset
    print(f"Sampling from: {dataset_centroid_offset}")

    tr = gemmi.Transform()
    tr.mat.fromlist(transform_mat.tolist())
    tr.vec.fromlist(dataset_centroid_offset.tolist())

    arr = np.zeros([grid_size, grid_size, grid_size], dtype=np.float32)

    unaligned_xmap.interpolate_values(arr, tr)

    return arr


def sample_datasets(
        truncated_datasets: MutableMapping[str, Dataset],
        marker: Marker,
        alignments: MutableMapping[str, Alignment],
        structure_factors: StructureFactors,
        sample_rate: float,
        grid_size: int,
        grid_spacing: float,
) -> MutableMapping[str, np.ndarray]:
    samples: MutableMapping[str, np.ndarray] = {}
    arrays = joblib.Parallel(
        verbose=50,
        n_jobs=-1,
        # backend="multiprocessing",
    )(
        joblib.delayed(sample_dataset)(
            dataset,
            alignments[dtag][marker],
            marker,
            structure_factors,
            sample_rate,
            grid_size,
            grid_spacing,
        )
        for dtag, dataset
        in truncated_datasets.items()
    )
    samples = {dtag: result for dtag, result in zip(truncated_datasets, arrays)}

    return samples


def sample_xmap(transform_mat, centroid, grid_size, unaligned_xmap):
    tr = gemmi.Transform()
    tr.mat.fromlist(transform_mat.tolist())
    tr.vec.fromlist(centroid.tolist())

    arr = np.zeros([grid_size, grid_size, grid_size], dtype=np.float32)

    unaligned_xmap.interpolate_values(arr, tr)

    return arr


def sample_delta_transform_only(perturbation, transform_mat, centorid, unaligned_xmap, reference_sample, grid_size):
    # transformation = perturbation[]
    offset_sectroid = perturbation + centorid

    sample = sample_xmap(transform_mat, offset_sectroid, grid_size, unaligned_xmap)

    reference_sample_mean = np.mean(reference_sample)
    reference_sample_demeaned = reference_sample - reference_sample_mean
    reference_sample_denominator = np.sqrt(np.sum(np.square(reference_sample_demeaned)))

    sample_mean = np.mean(sample)
    sample_demeaned = sample - sample_mean
    sample_denominator = np.sqrt(np.sum(np.square(sample_demeaned)))

    nominator = np.sum(reference_sample_demeaned * sample_demeaned)
    denominator = sample_denominator * reference_sample_denominator

    correlation = nominator / denominator

    return correlation


def sample_xmap_perturbed(perturbation, transform_mat, centorid, unaligned_xmap, grid_size):
    transformation_perturbation = perturbation[0:3]
    rotation_perturbation = perturbation[3:6]
    perturbed_centroid = transformation_perturbation + centorid

    rotation_perturbation_obj = scipy.spatial.transform.Rotation.from_euler(
        "xyz",
        [rotation_perturbation[0], rotation_perturbation[1], rotation_perturbation[2]], degrees=True)
    rotation_perturbation_mat = rotation_perturbation_obj.as_matrix()

    perturbed_rotation_mat = np.matmul(transform_mat, rotation_perturbation_mat)

    sample = sample_xmap(perturbed_rotation_mat, perturbed_centroid, grid_size, unaligned_xmap)

    return sample


def sample_delta(perturbation, transform_mat, centorid, unaligned_xmap, reference_sample, grid_size):
    transformation_perturbation = perturbation[0:3]
    rotation_perturbation = perturbation[3:6]
    perturbed_centroid = transformation_perturbation + centorid

    rotation_perturbation_obj = scipy.spatial.transform.Rotation.from_euler(
        "xyz",
        [rotation_perturbation[0], rotation_perturbation[1], rotation_perturbation[2]], degrees=True)
    rotation_perturbation_mat = rotation_perturbation_obj.as_matrix()

    perturbed_rotation_mat = np.matmul(transform_mat, rotation_perturbation_mat)

    sample = sample_xmap(perturbed_rotation_mat, perturbed_centroid, grid_size, unaligned_xmap)

    reference_sample_mean = np.mean(reference_sample)
    reference_sample_demeaned = reference_sample - reference_sample_mean
    reference_sample_denominator = np.sqrt(np.sum(np.square(reference_sample_demeaned)))

    sample_mean = np.mean(sample)
    sample_demeaned = sample - sample_mean
    sample_denominator = np.sqrt(np.sum(np.square(sample_demeaned)))

    nominator = np.sum(reference_sample_demeaned * sample_demeaned)
    denominator = sample_denominator * reference_sample_denominator

    correlation = nominator / denominator

    return 1 - correlation


def sample_dataset_refined(
        dataset: Dataset,
        transform: Transform,
        marker: Marker,
        reference_sample: np.ndarray,
        structure_factors: StructureFactors,
        sample_rate: float,
        grid_size: int,
        grid_spacing: float,
) -> np.ndarray:
    reflections: gemmi.Mtz = dataset.reflections
    unaligned_xmap: gemmi.FloatGrid = reflections.transform_f_phi_to_map(
        structure_factors.f,
        structure_factors.phi,
        sample_rate=sample_rate,
    )

    unaligned_xmap_array = np.array(unaligned_xmap, copy=False)

    std = np.std(unaligned_xmap_array)
    unaligned_xmap_array[:, :, :] = unaligned_xmap_array[:, :, :] / std

    transform_inverse = transform.transform.inverse()

    transform_vec = -np.array(transform.transform.vec.tolist())
    print(f"transform vector: {transform_vec}")

    transform_mat = np.array(transform_inverse.mat.tolist())
    print(f"transform matrix: {transform_mat}")

    transform_mat = np.matmul(transform_mat, np.eye(3) * grid_spacing)
    print(f"transform matrix scaled: {transform_mat}")

    offset = np.array([grid_size / 2, grid_size / 2, grid_size / 2]).reshape(3, 1)
    print(f"offset: {offset}")

    rotated_offset = np.matmul(transform_mat, offset).flatten()
    print(f"rotated_offset: {rotated_offset}")

    dataset_centroid = np.array([marker.x, marker.y, marker.z]) + transform_vec
    print(f"dataset_centroid: {dataset_centroid}")

    dataset_centroid_offset = dataset_centroid - rotated_offset
    print(f"Sampling from: {dataset_centroid_offset}")

    initial_rscc = sample_delta((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), transform_mat, dataset_centroid_offset, unaligned_xmap,
                                reference_sample, grid_size)
    print(f"Intial rscc is: {initial_rscc}")

    res = scipy.optimize.shgo(
        lambda perturbation: sample_delta(perturbation, transform_mat, dataset_centroid_offset, unaligned_xmap,
                                          reference_sample, grid_size),
        [(-3, 3), (-3, 3), (-3, 3), (-180.0, 180.0), (-180.0, 180.0), (-180.0, 180.0), ],
        n=60, iters=5, sampling_method='sobol'
    )

    # res = scipy.optimize.differential_evolution(
    #     lambda perturbation: sample_delta(perturbation, transform_mat, dataset_centroid_offset, unaligned_xmap,
    #                                       reference_sample, grid_size),
    #     [(-3, 3), (-3, 3), (-3, 3), (-180.0, 180.0), (-180.0, 180.0), (-180.0, 180.0), ]
    # )

    print(f"Initial rscc was: {initial_rscc}; Refinement is: {res}")

    # sample_arr = sample_xmap(transform_mat, dataset_centroid_offset + res.x, grid_size, unaligned_xmap)

    sample_arr = sample_xmap_perturbed(res.x, transform_mat, dataset_centroid_offset, unaligned_xmap, grid_size)

    return 1 - res.fun, sample_arr


def sample_datasets_refined(
        truncated_datasets: MutableMapping[str, Dataset],
        marker: Marker,
        alignments: MutableMapping[str, Alignment],
        reference_sample: np.ndarray,
        structure_factors: StructureFactors,
        sample_rate: float,
        grid_size: int,
        grid_spacing: float,
) -> MutableMapping[str, np.ndarray]:
    samples: MutableMapping[str, np.ndarray] = {}
    results = joblib.Parallel(
        verbose=50,
        n_jobs=-1,
    )(
        joblib.delayed(sample_dataset_refined)(
            dataset,
            alignments[dtag][marker],
            marker,
            reference_sample,
            structure_factors,
            sample_rate,
            grid_size,
            grid_spacing,
        )
        for dtag, dataset
        in truncated_datasets.items()
    )
    samples = {dtag: result[1] for dtag, result in zip(truncated_datasets, results)}

    return samples


def sample_datasets_refined_iterative(
        truncated_datasets: MutableMapping[str, Dataset],
        marker: Marker,
        alignments: MutableMapping[str, Alignment],
        apo_datatags: np.ndarray,
        structure_factors: StructureFactors,
        sample_rate: float,
        grid_size: int,
        grid_spacing: float,
        cutoff,
) -> MutableMapping[str, np.ndarray]:
    samples: MutableMapping[str, np.ndarray] = {}

    datasets_to_process = {dtag: dataset for dtag, dataset in truncated_datasets.items()}

    truncated_datasets_length = len(truncated_datasets)

    alignment_classes = {}

    while len(samples) < truncated_datasets_length:
        print(f"\tGot {len(datasets_to_process)} datasets left to align")
        dtag_array = list(datasets_to_process.keys())

        reference_dtag = dtag_array[0]

        reference_sample = sample_dataset(truncated_datasets[reference_dtag],
                                          alignments[reference_dtag][marker],
                                          marker,
                                          structure_factors,
                                          sample_rate,
                                          grid_size,
                                          grid_spacing,
                                          )
        arrays = joblib.Parallel(
            verbose=50,
            n_jobs=-1,
        )(
            joblib.delayed(sample_dataset_refined)(
                datasets_to_process[dtag],
                alignments[dtag][marker],
                marker,
                reference_sample,
                structure_factors,
                sample_rate,
                grid_size,
                grid_spacing,
            )
            for dtag
            in dtag_array
        )

        for j, dtag in enumerate(dtag_array):
            alignment_classes[dtag] = []
            rscc = arrays[j][0]
            array = arrays[j][1]

            if rscc > cutoff:
                samples[dtag] = array
                alignment_classes[dtag].append(array)
                del datasets_to_process[dtag]
            else:
                continue

    # samples = {dtag: result for dtag, result in zip(truncated_datasets, arrays)}

    return samples, alignment_classes


def get_corr(reference_sample_mask, sample_mask, diag):
    reference_mask_size = np.sum(reference_sample_mask)
    sample_mask_size = np.sum(sample_mask)

    denominator = max(sample_mask_size, reference_mask_size)

    if denominator == 0.0:
        if diag:
            corr = 1.0
        else:
            corr = 0.0

    else:

        corr = np.sum(sample_mask[reference_sample_mask == 1]) / denominator

    return corr


def get_distance_matrix(samples: MutableMapping[str, np.ndarray]) -> np.ndarray:
    # Make a pairwise matrix
    correlation_matrix = np.zeros((len(samples), len(samples)))

    for x, reference_sample in enumerate(samples.values()):

        reference_sample_mean = np.mean(reference_sample)
        reference_sample_demeaned = reference_sample - reference_sample_mean
        reference_sample_denominator = np.sqrt(np.sum(np.square(reference_sample_demeaned)))

        for y, sample in enumerate(samples.values()):
            sample_mean = np.mean(sample)
            sample_demeaned = sample - sample_mean
            sample_denominator = np.sqrt(np.sum(np.square(sample_demeaned)))

            nominator = np.sum(reference_sample_demeaned * sample_demeaned)
            denominator = sample_denominator * reference_sample_denominator

            correlation = nominator / denominator

            correlation_matrix[x, y] = correlation

    correlation_matrix = np.nan_to_num(correlation_matrix)

    # distance_matrix = np.ones(correlation_matrix.shape) - correlation_matrix

    for j in range(correlation_matrix.shape[0]):
        correlation_matrix[j, j] = 1.0

    return correlation_matrix


def get_reference(datasets: MutableMapping[str, Dataset], reference_dtag: Optional[str],
                  apo_dtags: List[str]) -> gemmi.Structure:
    # If reference dtag given, select it
    if reference_dtag:
        for dtag in datasets:
            if dtag == reference_dtag:
                return datasets[dtag]

        raise Exception("Reference dtag not in datasets!")

    # Otherwise, select highest resolution structure
    else:
        reference_dtag = min(
            apo_dtags,
            key=lambda dataset_dtag: datasets[dataset_dtag].reflections.resolution_high(),
        )

        return datasets[reference_dtag]


def get_linkage_from_correlation_matrix(correlation_matrix):
    condensed = spsp.distance.squareform(1.0 - correlation_matrix)
    linkage = spc.hierarchy.linkage(condensed, method='complete')
    # linkage = spc.linkage(condensed, method='ward')

    return linkage


def cluster_linkage(linkage, cutoff):
    idx = spc.hierarchy.fcluster(linkage, cutoff, 'distance')

    return idx


def cluster_density(linkage: np.ndarray, cutoff: float) -> np.ndarray:
    # Get the linkage matrix
    # Cluster the datasets
    clusters: np.ndarray = cluster_linkage(linkage, cutoff)
    # Determine which clusters have known apos in them

    return clusters


def get_common_reflections(
        moving_reflections: gemmi.Mtz,
        reference_reflections: gemmi.Mtz,
        structure_factors: StructureFactors,
):
    # Get own reflections
    moving_reflections_array = np.array(moving_reflections, copy=True)
    moving_reflections_table = pd.DataFrame(
        moving_reflections_array,
        columns=moving_reflections.column_labels(),
    )
    moving_reflections_table.set_index(["H", "K", "L"], inplace=True)
    dtag_flattened_index = moving_reflections_table[
        ~moving_reflections_table[structure_factors.f].isna()].index.to_flat_index()

    # Get reference
    reference_reflections_array = np.array(reference_reflections, copy=True)
    reference_reflections_table = pd.DataFrame(reference_reflections_array,
                                               columns=reference_reflections.column_labels(),
                                               )
    reference_reflections_table.set_index(["H", "K", "L"], inplace=True)
    reference_flattened_index = reference_reflections_table[
        ~reference_reflections_table[structure_factors.f].isna()].index.to_flat_index()

    running_index = dtag_flattened_index.intersection(reference_flattened_index)

    return running_index.to_list()


def get_all_common_reflections(datasets: MutableMapping[str, Dataset], structure_factors: StructureFactors,
                               tol=0.000001):
    running_index = None

    for dtag, dataset in datasets.items():
        reflections = dataset.reflections
        reflections_array = np.array(reflections, copy=True)
        reflections_table = pd.DataFrame(reflections_array,
                                         columns=reflections.column_labels(),
                                         )
        reflections_table.set_index(["H", "K", "L"], inplace=True)

        is_na = reflections_table[structure_factors.f].isna()
        is_zero = reflections_table[structure_factors.f].abs() < tol
        mask = ~(is_na | is_zero)

        flattened_index = reflections_table[mask].index.to_flat_index()
        if running_index is None:
            running_index = flattened_index
        running_index = running_index.intersection(flattened_index)
    return running_index.to_list()


def truncate_reflections(reflections: gemmi.Mtz, index=None) -> gemmi.Mtz:
    new_reflections = gemmi.Mtz(with_base=False)

    # Set dataset properties
    new_reflections.spacegroup = reflections.spacegroup
    new_reflections.set_cell_for_all(reflections.cell)

    # Add dataset
    new_reflections.add_dataset("truncated")

    # Add columns
    for column in reflections.columns:
        new_reflections.add_column(column.label, column.type)

    # Get data
    data_array = np.array(reflections, copy=True)
    data = pd.DataFrame(data_array,
                        columns=reflections.column_labels(),
                        )
    data.set_index(["H", "K", "L"], inplace=True)

    # Truncate by index
    data_indexed = data.loc[index]

    # To numpy
    data_dropped_array = data_indexed.to_numpy()

    # new data
    new_data = np.hstack([data_indexed.index.to_frame().to_numpy(),
                          data_dropped_array,
                          ]
                         )

    # Update
    new_reflections.set_data(new_data)

    # Update resolution
    new_reflections.update_reso()

    return new_reflections


def smooth(reference: Dataset, moving: Dataset, structure_factors: StructureFactors):
    # Get common set of reflections
    common_reflections = get_common_reflections(
        reference.reflections,
        moving.reflections,
        structure_factors,
    )

    # Truncate
    truncated_reference: gemmi.Mtz = truncate_reflections(reference.reflections, common_reflections)
    truncated_dataset: gemmi.Mtz = truncate_reflections(moving.reflections, common_reflections)

    # Refference array
    reference_reflections: gemmi.Mtz = truncated_reference
    reference_reflections_array: np.ndarray = np.array(reference_reflections,
                                                       copy=True,
                                                       )
    reference_reflections_table = pd.DataFrame(reference_reflections_array,
                                               columns=reference_reflections.column_labels(),
                                               )
    reference_f_array = reference_reflections_table[structure_factors.f].to_numpy()

    # Dtag array
    dtag_reflections = truncated_dataset
    dtag_reflections_array = np.array(dtag_reflections,
                                      copy=True,
                                      )
    dtag_reflections_table = pd.DataFrame(dtag_reflections_array,
                                          columns=dtag_reflections.column_labels(),
                                          )
    dtag_f_array = dtag_reflections_table[structure_factors.f].to_numpy()

    # Resolution array
    resolution_array = reference_reflections.make_1_d2_array()

    # Prepare optimisation
    x = reference_f_array
    y = dtag_f_array

    r = resolution_array

    sample_grid = np.linspace(min(r), max(r), 100)

    sorting = np.argsort(r)
    r_sorted = r[sorting]
    x_sorted = x[sorting]
    y_sorted = y[sorting]

    scales = []
    rmsds = []

    # Approximate x_f
    former_sample_point = sample_grid[0]
    x_f_list = []
    for sample_point in sample_grid[1:]:
        mask = (r_sorted < sample_point) * (r_sorted > former_sample_point)
        x_vals = x_sorted[mask]
        former_sample_point = sample_point
        x_f_list.append(np.mean(x_vals))
    x_f = np.array(x_f_list)

    # Optimise the scale factor
    for scale in np.linspace(-10, 10, 100):

        y_s_sorted = y_sorted * np.exp(scale * r_sorted)

        # approximate y_f
        former_sample_point = sample_grid[0]
        y_f_list = []
        for sample_point in sample_grid[1:]:
            mask = (r_sorted < sample_point) * (r_sorted > former_sample_point)
            y_vals = y_s_sorted[mask]
            former_sample_point = sample_point
            y_f_list.append(np.mean(y_vals))
        y_f = np.array(y_f_list)

        rmsd = np.sum(np.abs(x_f - y_f))

        scales.append(scale)
        rmsds.append(rmsd)

    min_scale = scales[np.argmin(rmsds)]

    # Get the original reflections
    original_reflections = moving.reflections

    original_reflections_array = np.array(original_reflections,
                                          copy=True,
                                          )

    original_reflections_table = pd.DataFrame(original_reflections_array,
                                              columns=reference_reflections.column_labels(),
                                              )

    f_array = original_reflections_table[structure_factors.f]

    f_scaled_array = f_array * np.exp(min_scale * original_reflections.make_1_d2_array())

    original_reflections_table[structure_factors.f] = f_scaled_array

    # New reflections
    new_reflections = gemmi.Mtz(with_base=False)

    # Set dataset properties
    new_reflections.spacegroup = original_reflections.spacegroup
    new_reflections.set_cell_for_all(original_reflections.cell)

    # Add dataset
    new_reflections.add_dataset("scaled")

    # Add columns
    for column in original_reflections.columns:
        new_reflections.add_column(column.label, column.type)

    # Update
    new_reflections.set_data(original_reflections_table.to_numpy())

    # Update resolution
    new_reflections.update_reso()

    # Create new dataset
    smoothed_dataset = Dataset(
        moving.dtag,
        moving.structure,
        new_reflections,
        moving.structure_path,
        moving.reflections_path,
        moving.fragment_path,
        moving.fragment_structures,
        min_scale,
    )

    return smoothed_dataset


def smooth_datasets(
        datasets: MutableMapping[str, Dataset],
        reference_dataset: Dataset,
        structure_factors: StructureFactors,
        debug: bool = True,
) -> MutableMapping[str, Dataset]:
    # For dataset reflections
    datasets_list: List[Optional[Dataset]] = joblib.Parallel(
        verbose=50,
        n_jobs=-1,
    )(
        joblib.delayed(smooth)(
            reference_dataset, dataset, structure_factors
        )
        for dataset
        in list(datasets.values())
    )
    smoothed_datasets = {dtag: smoothed_dataset for dtag, smoothed_dataset in zip(list(datasets.keys()), datasets_list)}

    return smoothed_datasets


def save_mtz(mtz: gemmi.Mtz, path: Path):
    mtz.write_to_file(str(path))


def save_distance_matrix(distance_matrix, path):
    np.save(str(path), distance_matrix)


def save_dtag_array(dtag_array, path):
    np.save(str(path), dtag_array)


def embed(distance_matrix):
    pca = decomposition.PCA(n_components=min(distance_matrix.shape[0], 50))
    tsne = manifold.TSNE(n_components=2)
    transform = pca.fit_transform(distance_matrix)
    transform = tsne.fit_transform(transform)
    return transform


def embed_umap(distance_matrix):
    pca = decomposition.PCA(n_components=min(distance_matrix.shape[0], 50))
    reducer = umap.UMAP()
    transform = pca.fit_transform(distance_matrix)
    transform = reducer.fit_transform(transform)
    return transform


def embed_umap_no_pca(distance_matrix):
    reducer = umap.UMAP()
    transform = reducer.fit_transform(distance_matrix)
    return transform


def get_global_distance_matrix(clustering_dict, markers, datasets):
    num_datasets = len(datasets)
    num_residues = len(markers)

    dataset_connectivity_matrix = np.zeros((num_datasets, num_datasets))
    num_residues_matrix = np.zeros((num_datasets, num_datasets))

    for marker in markers:
        # for residue_id, residue_clustering in clustering_dict.items():

        for x, dtag in enumerate(datasets):
            try:
                cluster_index_x = clustering_dict[marker][dtag]
            except:
                continue

            for y, dtag_y in enumerate(datasets):
                try:
                    cluster_index_y = clustering_dict[marker][dtag_y]
                except:
                    continue

                num_residues_matrix[x, y] = num_residues_matrix[x, y] + 1

                if cluster_index_x == cluster_index_y:
                    dataset_connectivity_matrix[x, y] = dataset_connectivity_matrix[x, y] + 1

    return dataset_connectivity_matrix / num_residues_matrix


def save_parallel_cat_plot(clustering_dict, out_file):
    dimensions = []

    for residue_id, cluster_dict in clustering_dict.items():
        dimensions.append(
            {
                "label": f"{residue_id}",
                "values": list(cluster_dict.values()),
            }
        )

    fig = go.Figure(
        go.Parcats(
            dimensions=dimensions
        )
    )

    fig.write_image(
        str(out_file),
        engine="kaleido",
        width=2000,
        height=1000,
        scale=1,
    )


def save_json(clustering_dict, path):
    clustering_dict_python = {
        "{}_{}_{}".format(marker.resid.model, marker.resid.chain, marker.resid.insertion): result
        for marker, result in clustering_dict.items()}

    with open(str(path), "w") as f:
        json.dump(clustering_dict_python, f)


def save_dendrogram_plot(linkage,
                         labels,
                         dendrogram_plot_file,
                         ):
    fig, ax = plt.subplots(figsize=(0.2 * len(labels), 40))
    dn = spc.hierarchy.dendrogram(linkage, ax=ax, labels=labels, leaf_font_size=10)
    fig.savefig(str(dendrogram_plot_file))
    fig.clear()
    plt.close(fig)


def save_num_clusters_bar_plot(clustering_dict, plot_file):
    dtag_list = list(list(clustering_dict.values())[0].keys())

    fig, ax = plt.subplots(figsize=(0.2 * len(clustering_dict), 20))

    x = np.arange(len(clustering_dict))
    y = [np.unique([cluster_id for cluster_id in cluster_dict.values()]).size for cluster_dict in
         clustering_dict.values()]
    labels = [f"{residue_id}" for residue_id in clustering_dict]

    plt.bar(x, y)
    plt.xticks(x, labels, rotation='vertical', fontsize=8)
    fig.savefig(str(plot_file))
    fig.clear()
    plt.close(fig)


def save_num_clusters_stacked_bar_plot(clustering_dict, plot_file):
    dtag_list = list(list(clustering_dict.values())[0].keys())

    cluster_idx_dict = {}
    for residue_id, cluster_dict in clustering_dict.items():

        for dtag, cluster_idx in cluster_dict.items():
            # When a new cluster is discovered
            if not cluster_idx in cluster_idx_dict:
                cluster_idx_dict[cluster_idx] = {}
                for _residue_id in clustering_dict.keys():
                    cluster_idx_dict[cluster_idx][_residue_id] = 0

            cluster_idx_dict[cluster_idx][residue_id] = cluster_idx_dict[cluster_idx][residue_id] + 1

    #
    for residue_id, cluster_dict in clustering_dict.items():

        residue_cluster_dict = {cluster_idx: cluster_idx_dict[cluster_idx][residue_id] for cluster_idx in
                                cluster_idx_dict}

        sorted_cluster_dict = {cluster_idx: sorted_cluster_val for cluster_idx, sorted_cluster_val
                               in zip(residue_cluster_dict.keys(), sorted(residue_cluster_dict.values(), reverse=True))}

        for sorted_cluster_idx, sorted_cluster_value in sorted_cluster_dict.items():
            cluster_idx_dict[sorted_cluster_idx][residue_id] = sorted_cluster_value

    fig, ax = plt.subplots(figsize=(0.2 * len(clustering_dict), 20))

    cluster_bar_plot_dict = {}
    x = np.arange(len(clustering_dict))
    y_prev = [0.0 for residue_id in clustering_dict.keys()]
    for cluster_idx, cluster_residue_dict in cluster_idx_dict.items():
        y = [num_cluster_members for residue_id, num_cluster_members in cluster_residue_dict.items()]
        print(len(x))
        print(len(y))
        print(len(y_prev))

        p = plt.bar(x, y, bottom=y_prev)
        y_prev = [y_prev_item + y_item for y_prev_item, y_item in zip(y_prev, y)]
        cluster_bar_plot_dict[cluster_idx] = p

    # labels = [f"{residue_id.chain}_{residue_id.insertion}" for residue_id in clustering_dict]
    labels = [f"{residue_id}" for residue_id in clustering_dict]

    plt.xticks(x, labels, rotation='vertical', fontsize=8)
    # plt.legend([x[0] for x in cluster_bar_plot_dict.values()], [str(x) for x in cluster_bar_plot_dict.keys()])
    fig.savefig(str(plot_file))
    fig.clear()
    plt.close(fig)


def bokeh_scatter_plot(embedding, labels, plot_file):
    output_file(str(plot_file))

    source = ColumnDataSource(
        data=dict(
            x=embedding[:, 0].tolist(),
            y=embedding[:, 1].tolist(),
            desc=labels,
        ))

    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("desc", "@desc"),
    ]

    p = figure(plot_width=1200, plot_height=1200, tooltips=TOOLTIPS,
               title="Mouse over the dots")

    p.circle('x', 'y', size=15, source=source)

    show(p)


def save_plot_tsne_bokeh(dataset_connectivity_matrix, labels, plot_file):
    embedding = embed(dataset_connectivity_matrix)
    bokeh_scatter_plot(embedding, labels, plot_file)


def save_plot_pca_umap_bokeh(dataset_connectivity_matrix, labels, plot_file):
    embedding = embed_umap(dataset_connectivity_matrix)
    bokeh_scatter_plot(embedding, labels, plot_file)


def save_plot_umap_bokeh(dataset_connectivity_matrix, labels, plot_file):
    embedding = embed_umap_no_pca(dataset_connectivity_matrix)
    bokeh_scatter_plot(embedding, labels, plot_file)


def save_embed_plot(dataset_connectivity_matrix, plot_file):
    embedding = embed(dataset_connectivity_matrix)

    fig, ax = plt.subplots(figsize=(60, 60))

    ax.scatter(embedding[:, 0], embedding[:, 1])

    fig.savefig(str(plot_file))
    fig.clear()
    plt.close(fig)


def save_embed_umap_plot(dataset_connectivity_matrix, plot_file):
    embedding = embed_umap(dataset_connectivity_matrix)

    fig, ax = plt.subplots(figsize=(60, 60))

    ax.scatter(embedding[:, 0], embedding[:, 1])

    fig.savefig(str(plot_file))
    fig.clear()
    plt.close(fig)


def save_umap_plot(dataset_connectivity_matrix, plot_file):
    embedding = embed_umap_no_pca(dataset_connectivity_matrix)

    fig, ax = plt.subplots(figsize=(60, 60))

    ax.scatter(embedding[:, 0], embedding[:, 1])

    fig.savefig(str(plot_file))
    fig.clear()
    plt.close(fig)


def save_global_cut_curve(linkage, plot_file):
    fig, ax = plt.subplots(figsize=(60, 60))

    x = np.linspace(0, 1, 100)

    cuts = hierarchy.cut_tree(linkage, height=x)

    num_clusters = [np.unique(cuts[:, i]).size for i in range(x.size)]

    ax.scatter(x, num_clusters)

    fig.savefig(str(plot_file))
    fig.clear()
    plt.close(fig)


def save_hdbscan_dendrogram(connectivity_matrix, plot_file):
    clusterer = hdbscan.HDBSCAN(metric='precomputed', allow_single_cluster=True, min_cluster_size=10)
    clusterer.fit(connectivity_matrix)
    print(clusterer.labels_)

    fig, ax = plt.subplots(figsize=(60, 60))

    clusterer.condensed_tree_.plot(
        select_clusters=True,
        axis=ax,
    )

    fig.savefig(str(plot_file))
    fig.clear()
    plt.close(fig)


def save_correlation_plot(correlation_matrix, correlation_plot_file):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(correlation_matrix)
    fig.savefig(str(correlation_plot_file))
    fig.clear()
    plt.close(fig)


def filter_sfs(dataset: Dataset, structure_factor: StructureFactors):
    columns = dataset.reflections.column_labels()

    if structure_factor.f in columns:
        return True

    else:
        return False


def filter_structure(dataset: Dataset, reference_dataset: Dataset):
    ca_self = []
    ca_other = []

    # Get CAs
    matched = 0
    total = 0

    reference = reference_dataset.structure
    monomerized = False
    other = dataset.structure

    for model in reference:
        for chain in model:
            for res_self in chain.get_polymer():
                if 'LIG' in str(res_self):
                    print('Skipping Ligand...')
                    continue

                total += 1

                try:
                    current_res_id = ResidueID.from_residue_chain(model, chain, res_self)
                    if monomerized:
                        # print(other.structure[current_res_id.model])
                        # print(len(other.structure[current_res_id.model]))
                        res_other = other[current_res_id.model][0][current_res_id.insertion][0]
                    else:
                        res_other = \
                            other[current_res_id.model][current_res_id.chain][current_res_id.insertion][0]
                    # print(f'{self.structure}|{res_self}')
                    # print(f'{other.structure}|{res_other}')
                    self_ca_pos = res_self["CA"][0].pos
                    other_ca_pos = res_other["CA"][0].pos

                    matched += 1

                except Exception as e:
                    print(f"Exception: {e}")
                    print('Skipping, Residue not found in chain')
                    continue

                ca_list_self = TransformGlobal.pos_to_list(self_ca_pos)
                ca_list_other = TransformGlobal.pos_to_list(other_ca_pos)

                ca_self.append(ca_list_self)
                ca_other.append(ca_list_other)

    if len(ca_other) > 3:
        return True

    else:
        return False


def save_ccp4(path: Path, grid: gemmi.FloatGrid):
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid

    ccp4.setup()

    ccp4.update_ccp4_header(2, True)

    ccp4.write_ccp4_map(str(path))


def make_mean_map_local(
        samples,
        reference_dataset: Dataset,
        marker: Marker,
        grid_size,
        grid_step,
        structure_factors,
        sample_rate,
):
    # Get the mean
    samples_array = np.stack(samples, axis=0)
    samples_mean = np.mean(samples, axis=0)

    # Get a grid in the sample frame (defined by size and spacing because orthogonal)
    sample_grid = gemmi.FloatGrid(grid_size, grid_size, grid_size)
    sample_unit_cell = gemmi.UnitCell(
        int((grid_size - 1) / grid_step),
        int((grid_size - 1) / grid_step),
        int((grid_size - 1) / grid_step),
        90,
        90,
        90,
    )
    sample_grid.set_unit_cell(sample_unit_cell)

    # Sample the mean onto the sample grid
    for point in sample_grid:
        u = point.u
        v = point.v
        w = point.w
        value = samples_mean[u, v, w]

        sample_grid.set_value(u, v, w, value)

    # Get a grid in the reference frame
    reflections: gemmi.Mtz = reference_dataset.reflections
    unaligned_xmap: gemmi.FloatGrid = reflections.transform_f_phi_to_map(
        structure_factors.f,
        structure_factors.phi,
        sample_rate=sample_rate,
    )
    reference_grid = gemmi.FloatGrid(unaligned_xmap.nu, unaligned_xmap.nv, unaligned_xmap.nw)
    reference_grid.spacegroup = unaligned_xmap.spacegroup
    reference_grid.set_unit_cell(unaligned_xmap.unit_cell)

    # Mask the relevant reference frame grid points
    reference_grid.set_points_around(
        gemmi.Position(marker.x, marker.y, marker.z),
        3.0,
        1.0,
    )

    # Iterate over grid points, transforming into sample space, and
    for point in reference_grid:
        if point.value != 0.0:
            pos = reference_grid.point_to_position(point)
            pos_sample_frame = gemmi.Position(
                (pos.x - marker.x) - ((grid_size - 1) / grid_step),
                (pos.y - marker.y) - ((grid_size - 1) / grid_step),
                (pos.z - marker.z) - ((grid_size - 1) / grid_step),
            )
            interpolated_value = reference_grid.interpolate_value(pos_sample_frame)
            reference_grid.set_value(
                point.u,
                point.v,
                point.w,
                interpolated_value
            )

    # Update symmetry
    reference_grid.symmeterize_max()

    # return
    return reference_grid


def output_mean_maps_local(sample_arrays, dataset_clusters, reference_dataset, marker, out_dir,
                           grid_size, grid_step, structure_factors, sample_rate):
    dtag_array = np.array(list(sample_arrays.keys()))
    cluster_dtags: MutableMapping[int, List[str]] = {
        cluster_num: [dtag for dtag in dtag_array[dataset_clusters == cluster_num]]
        for cluster_num
        in dataset_clusters
    }

    for cluster_num in cluster_dtags:
        mean_map: gemmi.FloatGrid = make_mean_map_local(
            [sample_arrays[dtag] for dtag in cluster_dtags[cluster_num]],
            reference_dataset,
            marker,
            grid_size, grid_step, structure_factors, sample_rate
        )

        save_ccp4(
            out_dir / f"mean_map_{marker.resid.model}_{marker.resid.chain}_{marker.resid.insertion}_{cluster_num}.ccp4",
            mean_map,
        )
