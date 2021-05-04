from pandda_local_cluster.datatypes import *
from pandda_local_cluster.datatypes_global_cluster import *
from pandda_local_cluster.functions import *

import numpy as np


def get_global_alignments():
    def get_global_alignment():
        def get_common_landmarks():
            #
            ...

        def get_transform_between_landmarks():
            ...

        # Get common landmarks

        # Get optimal transform between landmarks

        ...

    # For each structure, align to reference
    ...


def get_sample_region(reference_dataset: Dataset, grid_step: float, margin: float):
    def get_bounding_box(_dataset: Dataset, margin: float) -> BoundingBox:
        min_pos = np.array()
        max_pos = np.array([])

        for model in _dataset.structure:
            for chain in model:
                for residue in chain.get_polymer():
                    for atom in residue:
                        pos = atom.pos

        return BoundingBox(min_pos, max_pos)

    def get_origin():
        ...

    def get_gridding(_bounding_box, _grid_step):
        _gridding = np.floor((_bounding_box.max - _bounding_box.min) / _grid_step) + 1

        return _gridding

    # Get bounding box
    bounding_box = get_bounding_box(reference_dataset)

    # Get Origin
    origin = bounding_box.min

    # Get default rotation
    rotation = np.eye(3)

    # Get gridding
    gridding = get_gridding(bounding_box, grid_step)

    return SampleRegion(
        origin,
        rotation,
        gridding,
        None,
    )


def get_sample_region_mask():
    # Get unit cell mask

    # Get sample region mask
    ...


def sample_datasets_global(datasets: Datasets, masked_sample_region):
    def get_reference(datasets: Datasets):
        ...

    def sample_dataset(_dataset: Dataset, _sample_region: SampleRegion, ):
        ...

    def sample_dataset_global():
        def transform_sample_region():
            ...

        def optimise_sampling():
            ...

        # Transform sample region

        # Get initial Sample

        # Optimise sample

        ...

    # Define initial datasets
    datasets_to_sample = {dtag: dataset for dtag, dataset in datasets.items()}

    # While there are unaligned datasets
    num_datasets = len(datasets)
    while len(datasets_to_sample) < num_datasets:

        # Choose a reference
        reference = get_reference(datasets_to_sample)

        # Sample reference
        sample = sample_dataset(reference, masked_sample_region, alignment)

        # Sample datasets
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

        # Update datasets to align
        for j, dtag in enumerate(dtag_array):
            rscc = arrays[j][0]
            array = arrays[j][1]

            if rscc > cutoff:
                samples[dtag] = array
                del datasets_to_process[dtag]
            else:
                continue


def run_global_cluster(
        data_dir: str,
        out_dir: str,
        known_apos: List[str] = None,
        reference_dtag: Optional[str] = None,
        markers: Optional[List[Tuple[float, float, float]]] = None,
        structure_factors="FWT,PHWT",
        structure_regex="*.pdb",
        reflections_regex="*.mtz",
        cutoff=0.7,
):
    # Update the Parameters
    params: Params = Params()
    params.update(structure_factors=structure_factors, structure_regex=structure_regex,
                  reflections_regex=reflections_regex,
                  cutoff=cutoff)
    if params.debug:
        print(params)

    # Type the input
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    if params.debug:
        print(
            (
                "Input:\n"
                f"\tdata_dir: {data_dir}\n"
                f"\tout dir: {out_dir}\n"
                f"\tknown apos: {known_apos}\n"
                f"\treference_dtag: {reference_dtag}\n"
                f"\tmarkers: {markers}\n"
            )
        )

    # Load the datasets
    datasets: MutableMapping[str, Dataset] = get_datasets(
        data_dir,
        params.structure_regex,
        params.reflections_regex,
        params.smiles_regex,
        0.0,
    )
    if params.debug:
        print(datasets)

    if not known_apos:
        known_apos = [dataset.dtag for dtag, dataset in datasets.items() if not dataset.fragment_path]
        if params.debug:
            print(f"Got {len(known_apos)} known apos")

        if len(known_apos) == 0:
            print("WARNING! Did not find any known apos. Using all datasets.")
            known_apos = [dataset.dtag for dtag, dataset in datasets.items()]
    else:
        if params.debug:
            print(f"Was given {len(known_apos)} known apos")

    # Get a reference dataset against which to sample things
    reference_dataset: Dataset = get_reference(datasets, reference_dtag, known_apos)
    if params.debug:
        print(f"Reference dataset for alignment is: {reference_dataset.dtag}")

    # B factor smooth the datasets
    smoothed_datasets: MutableMapping[str, Dataset] = smooth_datasets(
        datasets,
        reference_dataset,
        params.structure_factors,
    )

    # Get the markers for alignment
    markers: List[Marker] = get_markers(reference_dataset, markers)

    # Find the alignments between the reference and all other datasets
    alignments: MutableMapping[str, Alignment] = get_global_alignment(smoothed_datasets, reference_dataset, markers)

    # Truncate the datasets to the same reflections
    truncated_datasets: MutableMapping[str, Dataset] = get_truncated_datasets(
        smoothed_datasets,
        reference_dataset,
        params.structure_factors,
    )
    print(f"After truncation have {len(truncated_datasets)} datasets")

    # resolution
    resolution: float = list(truncated_datasets.values())[0].reflections.resolution_high()
    if params.debug:
        print(f"Resolution is: {resolution}")

    # Determine the sampling region
    sample_region = get_sample_region(reference_dataset)

    # Sample the datasets to ndarrays
    if params.debug:
        print(f"Getting sample arrays...")

    sample_arrays: MutableMapping[str, np.ndarray] = sample_datasets_global(
        truncated_datasets,
        sample_region,
        alignments,
        known_apos,
        params.structure_factors,
        params.sample_rate,
        params.grid_size,
        params.grid_spacing,
        0.7
    )
    print(f"Got {len(sample_arrays)} samples")

    # Get the distance matrix
    distance_matrix: np.ndarray = get_distance_matrix(sample_arrays)
    if params.debug:
        print(f"First line of distance matrix: {distance_matrix[0, :]}")
        print(f"Last line of distance matrix: {distance_matrix[-1, :]}")

    # Get the distance matrix linkage
    linkage: np.ndarray = get_linkage_from_correlation_matrix(distance_matrix)

    # Cluster the available density
    dataset_clusters: np.ndarray = cluster_density(
        linkage,
        params.local_cluster_cutoff,
    )

    # Output
    save_distance_matrix(distance_matrix,
                         out_dir / f"distance_matrix.npy")

    save_dtag_array(np.array(list(sample_arrays.keys())),
                    out_dir / f"dtags.npy",
                    )

    save_dendrogram_plot(linkage,
                         labels=[dtag for dtag in sample_arrays.keys()],
                         dendrogram_plot_file=out_dir / f"dendrogram.png",
                         )

    save_hdbscan_dendrogram(
        distance_matrix,
        out_dir / f"hdbscan_dendrogram.png",
    )

    save_embed_plot(
        distance_matrix,
        out_dir / f"embed.png"
    )

    # Store resullts
    clusters = {
        dtag: cluster_id
        for dtag, cluster_id
        in zip(sample_arrays.keys(), dataset_clusters.flatten().tolist())
    }

    print(f"Clusters: {clusters}")
    # End loop over residues
