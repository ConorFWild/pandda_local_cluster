from pandda_local_cluster.datatypes import *
from pandda_local_cluster.datatypes_global_cluster import *
from pandda_local_cluster.functions import *

import numpy as np
import gemmi
import scipy
import joblib


def get_global_alignments(datasets, reference, ):
    def get_global_alignment(_dataset, _reference):
        def get_common_landmarks(__reference, __dataset):
            # reference markers
            reference_markers = get_markers(__reference, None)

            # datasett markers
            dataset_markers = get_markers(__dataset, None)

            # common markers
            dataset_marker_ids = {dataset_marker.resid: dataset_marker for dataset_marker in dataset_markers}
            _common_markers = {}
            for reference_marker in reference_markers:
                try:
                    partner = dataset_marker_ids[reference_marker.resid]
                    _common_markers[reference_marker] = (reference_marker, partner)
                except:
                    pass

            return _common_markers

        def get_transform_between_landmarks(_common_markers):
            reference_markers = [[markers[0].x, markers[0].y, markers[0].z] for markers in _common_markers.values()]
            dataset_markers = [[markers[1].x, markers[1].y, markers[1].z] for markers in _common_markers.values()]

            reference_marker_array = np.array(reference_markers)
            dataset_marker_array = np.array(dataset_markers)
            transform = get_transform_from_atoms(dataset_marker_array, reference_marker_array)

            return transform

        # Get common landmarks
        common_markers = get_common_landmarks(_reference, _dataset)

        # Get optimal transform between landmarks
        transform = get_transform_between_landmarks(common_markers)

        return transform

    # For each structure, align to reference
    dtag_array = list(datasets.keys())

    alignments = joblib.Parallel(
        verbose=50,
        n_jobs=-1,
    )(
        joblib.delayed(get_global_alignment)(
            datasets[dtag],
            reference,
        )
        for dtag
        in dtag_array
    )

    alignments = {dtag: alignment for dtag, alignment in zip(dtag_array, alignments)}

    return alignments


def get_sample_region(reference_dataset: Dataset, grid_step: float, margin: float):
    def get_bounding_box(_dataset: Dataset, _margin: float) -> BoundingBox:

        pos_list = []
        for model in _dataset.structure:
            for chain in model:
                for residue in chain.get_polymer():
                    for atom in residue:
                        pos = atom.pos
                        pos_list.append([pos.x, pos.y, pos.z])

        pos_array = np.array(pos_list)

        min_pos = np.array([np.min(pos_array[:, 0]), np.min(pos_array[:, 1]), np.min(pos_array[:, 2])])
        max_pos = np.array([np.max(pos_array[:, 0]), np.max(pos_array[:, 1]), np.max(pos_array[:, 2])])

        return BoundingBox(min_pos - margin, max_pos + margin)

    def get_origin():
        ...

    def get_gridding(_bounding_box, _grid_step):
        _gridding = np.floor((_bounding_box.max - _bounding_box.min) / _grid_step) + 1

        return _gridding

    # Get bounding box
    bounding_box = get_bounding_box(reference_dataset, margin)

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
        grid_step,
        None,
    )


def get_sample_region_mask():
    # Get unit cell mask

    # Get sample region mask
    ...


def get_transformed_sample_region(
        sample_region: SampleRegion,
        alignment: Transform,
):
    origin = sample_region.origin
    rotation = sample_region.rotation
    gridding = sample_region.gridding
    spacing = sample_region.spacing

    transform = alignment

    transform_inverse = transform.transform.inverse()

    transform_vec = -np.array(transform.transform.vec.tolist())
    print(f"transform vector: {transform_vec}")

    transform_mat = np.matmul(rotation, np.array(transform_inverse.mat.tolist()))
    print(f"transform matrix: {transform_mat}")

    transform_mat_scaled = np.matmul(transform_mat, np.eye(3) * spacing)
    print(f"transform matrix scaled: {transform_mat_scaled}")

    origin_to_centroid = np.matmul(gridding / 2, rotation)
    print(f"origin_to_centroid: {origin_to_centroid}")

    centroid = origin + origin_to_centroid
    print(f"Sample region centroid: {centroid}")

    centroid_to_origin = -origin_to_centroid
    print(f"centroid_to_origin: {centroid_to_origin}")

    rotated_centroid_to_origin = np.matmul(transform_mat, centroid_to_origin)
    print(f"rotated_centroid_to_origin: {rotated_centroid_to_origin}")

    # offset = (gridding / 2).reshape(3, 1)
    # print(f"offset: {offset}")
    #
    # rotated_offset = np.matmul(transform_mat, offset).flatten()
    # print(f"rotated_offset: {rotated_offset}")
    #
    # dataset_centroid = (origin + gridding / 2) + transform_vec
    # print(f"dataset_centroid: {dataset_centroid}")
    #
    # dataset_centroid_offset = dataset_centroid - rotated_offset
    # print(f"Sampling from: {dataset_centroid_offset}")

    rotated_origin = centroid + rotated_centroid_to_origin
    print(f"rotated_origin: {rotated_origin}")

    transformed_origin = rotated_origin + transform_vec
    print(f"transformed_origin: {transformed_origin}")

    transformed_sample_region = SampleRegion(
        transformed_origin,
        transform_mat,
        gridding,
        spacing,
        None,
    )

    return transformed_sample_region


def sample_dataset_global(unaligned_xmap, sample_region: SampleRegion):
    tr = gemmi.Transform()
    tr.mat.fromlist(sample_region.rotation.tolist())
    tr.vec.fromlist(sample_region.origin.tolist())

    arr = np.zeros(
        (
            int(sample_region.gridding[0]),
            int(sample_region.gridding[1]),
            int(sample_region.gridding[2]),
        ),
        dtype=np.float32,
    )

    unaligned_xmap.interpolate_values(arr, tr)

    return arr


def perturb_sample_region(transformed_sample_region, perturbation):
    # Split out the components of the perturbation
    transformation_perturbation = perturbation[0:3]
    rotation_perturbation = perturbation[3:6]

    #
    rotation_perturbation_obj = scipy.spatial.transform.Rotation.from_euler(
        "xyz",
        [rotation_perturbation[0], rotation_perturbation[1], rotation_perturbation[2]], degrees=True)
    rotation_perturbation_mat = rotation_perturbation_obj.as_matrix()

    # Package them as a transform

    transform = gemmi.Transform()
    transform.vec.fromlist(transformation_perturbation.tolist())
    transform.mat.fromlist(rotation_perturbation_mat.tolist())

    # Get the perturbed sample region
    perturbed_sample_region = get_transformed_sample_region(transformed_sample_region, Transform(transform))

    return perturbed_sample_region


def sample_dataset_global_perturbed(perturbation,
                                    transformed_sample_region,
                                    unaligned_xmap, ):
    perturbed_sample_region = perturb_sample_region(transformed_sample_region, perturbation)

    sample = sample_dataset_global(unaligned_xmap, perturbed_sample_region)

    return sample


def refine_sample_dataset_global(dataset, alignment, sample_region, reference_sample, structure_factors,
                                 sample_rate, ):
    def get_sample_rscc():
        sample = sample_dataset_global()

    def transform_sample_region():
        ...

    def optimise_sampling():
        ...

    def sample_rscc(perturbation,
                    transformed_sample_region,
                    unaligned_xmap,
                    reference_sample,
                    ):
        perturbed_sample_region = perturb_sample_region(transformed_sample_region, perturbation)

        sample = sample_dataset_global(unaligned_xmap, perturbed_sample_region)

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

    # Get the unaligned xmap
    reflections: gemmi.Mtz = dataset.reflections
    unaligned_xmap: gemmi.FloatGrid = reflections.transform_f_phi_to_map(
        structure_factors.f,
        structure_factors.phi,
        sample_rate=sample_rate,
    )
    unaligned_xmap_array = np.array(unaligned_xmap, copy=False)
    std = np.std(unaligned_xmap_array)
    unaligned_xmap_array[:, :, :] = unaligned_xmap_array[:, :, :] / std

    # Transform sample region
    transformed_sample_region = get_transformed_sample_region(
        sample_region,
        alignment,
    )

    # Get initial Sample
    reference_sample = sample_dataset_global(unaligned_xmap, transformed_sample_region)

    # Optimise sample
    res = scipy.optimize.shgo(
        lambda perturbation: sample_rscc(
            perturbation,
            transformed_sample_region,
            unaligned_xmap,
            reference_sample,
        ),
        [(-3, 3), (-3, 3), (-3, 3), (-180.0, 180.0), (-180.0, 180.0), (-180.0, 180.0), ],
        n=60, iters=5, sampling_method='sobol',
    )

    # Get the optimised sample
    sample_arr = sample_dataset_global_perturbed(res.x, transformed_sample_region,
                                                 unaligned_xmap, )

    return 1-res.fun, sample_arr


def sample_reference_dataset_global(dataset, alignment, sample_region, structure_factors,
                                    sample_rate, ):
    # Get the unaligned xmap
    reflections: gemmi.Mtz = dataset.reflections
    unaligned_xmap: gemmi.FloatGrid = reflections.transform_f_phi_to_map(
        structure_factors.f,
        structure_factors.phi,
        sample_rate=sample_rate,
    )
    unaligned_xmap_array = np.array(unaligned_xmap, copy=False)
    std = np.std(unaligned_xmap_array)
    unaligned_xmap_array[:, :, :] = unaligned_xmap_array[:, :, :] / std

    # Transform sample region
    transformed_sample_region = get_transformed_sample_region(
        sample_region,
        alignment,
    )

    # Get sample
    reference_sample = sample_dataset_global(unaligned_xmap, transformed_sample_region)

    return reference_sample


def sample_datasets_global(datasets: Datasets, alignments, structure_factors, grid_step,
                           sample_rate,
                           cutoff):
    def get_reference(datasets: Datasets):
        return list(datasets.values())[0]

    samples: MutableMapping[str, np.ndarray] = {}

    # Define initial datasets
    datasets_to_sample = {dtag: dataset for dtag, dataset in datasets.items()}

    # While there are unaligned datasets
    num_datasets = len(datasets)
    print(f"\tGot {len(datasets)} datasets to process")

    while len(samples) < num_datasets:
        dtag_array = list(datasets_to_sample.keys())

        # Choose a reference
        reference = get_reference(datasets_to_sample)
        print(f"\t\tRegerence is: {reference.dtag}")

        # get sample region
        sample_region = get_sample_region(reference, grid_step, margin=2.5)
        print(f"\t\tSample region is: {sample_region}")

        alignments = get_global_alignments(datasets_to_sample, reference)

        # Sample reference
        reference_sample = sample_reference_dataset_global(reference,
                                                           alignments[reference.dtag],
                                                           sample_region,
                                                           structure_factors,
                                                           sample_rate)

        # Sample datasets
        arrays = joblib.Parallel(
            verbose=50,
            n_jobs=-1,
        )(
            joblib.delayed(refine_sample_dataset_global)(
                datasets_to_sample[dtag],
                alignments[dtag],
                sample_region,
                reference_sample,
                structure_factors,
                sample_rate
            )
            for dtag
            in dtag_array
        )

        # Update datasets to align
        for j, dtag in enumerate(dtag_array):
            rscc = arrays[j][0]
            array = arrays[j][1]

            print(f"\tDtag rscc: {rscc}")

            if rscc > cutoff:
                samples[dtag] = array
                del datasets_to_sample[dtag]
            else:
                continue

        return samples


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
        print(f"Got: {len(datasets)} datasets")
        for dataset in datasets.values():
            print(f"\t\tStructure factors are: {dataset.reflections.column_labels()}")

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
        print(f"Reference reflection mtz columns are: {reference_dataset.reflections.column_labels()}")

    # B factor smooth the datasets
    smoothed_datasets: MutableMapping[str, Dataset] = smooth_datasets(
        datasets,
        reference_dataset,
        params.structure_factors,
    )

    # Get the markers for alignment
    markers: List[Marker] = get_markers(reference_dataset, markers)

    # Find the alignments between the reference and all other datasets
    alignments: MutableMapping[str, Alignment] = get_global_alignments(smoothed_datasets, reference_dataset, )

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

    # Sample the datasets to ndarrays
    if params.debug:
        print(f"Getting sample arrays...")

    sample_arrays: MutableMapping[str, np.ndarray] = sample_datasets_global(
        truncated_datasets,
        alignments,
        params.structure_factors,
        params.grid_spacing,
        params.sample_rate,
        cutoff=0.7,
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
