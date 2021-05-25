from pandda_local_cluster.datatypes import *
from pandda_local_cluster.datatypes_global_cluster import *
from pandda_local_cluster.functions import *

import numpy as np
import gemmi
import scipy
import joblib
import itertools


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


def mask_xmap(_xmap, _mask_grid):
    for point in _xmap:
        mask_val = _mask_grid.get_value(point.u, point.v, point.w)
        _xmap.set_value(point.u, point.v, point.w, point.value * mask_val)

    return _xmap


def refine_sample_dataset_global(dataset, alignment, sample_region, reference_sample, structure_factors,
                                 sample_rate, ):
    def sample_rscc(_perturbation,
                    _transformed_sample_region,
                    _unaligned_xmap,
                    _reference_sample,
                    ):
        perturbed_sample_region = perturb_sample_region(_transformed_sample_region, _perturbation)

        sample = sample_dataset_global(_unaligned_xmap, perturbed_sample_region)

        reference_sample_mean = np.mean(_reference_sample)
        reference_sample_demeaned = _reference_sample - reference_sample_mean
        reference_sample_denominator = np.sqrt(np.sum(np.square(reference_sample_demeaned)))

        sample_mean = np.mean(sample)
        sample_demeaned = sample - sample_mean
        sample_denominator = np.sqrt(np.sum(np.square(sample_demeaned)))

        nominator = np.sum(reference_sample_demeaned * sample_demeaned)
        denominator = sample_denominator * reference_sample_denominator

        correlation = nominator / denominator

        return 1 - correlation

    # def sample_rscc_masked(perturbation,
    #                        transformed_sample_region,
    #                        unaligned_xmap,
    #                        reference_sample,
    #                        mask,
    #                        ):
    #     perturbed_sample_region = perturb_sample_region(transformed_sample_region, perturbation)
    #
    #     sample = sample_dataset_global(unaligned_xmap, perturbed_sample_region)
    #     sample_mask = sample_dataset_global(mask, perturbed_sample_region)
    #     sample = sample * sample_mask
    #
    #     reference_sample_mean = np.mean(reference_sample)
    #     reference_sample_demeaned = reference_sample - reference_sample_mean
    #     reference_sample_denominator = np.sqrt(np.sum(np.square(reference_sample_demeaned)))
    #
    #     sample_mean = np.mean(sample)
    #     sample_demeaned = sample - sample_mean
    #     sample_denominator = np.sqrt(np.sum(np.square(sample_demeaned)))
    #
    #     nominator = np.sum(reference_sample_demeaned * sample_demeaned)
    #     denominator = sample_denominator * reference_sample_denominator
    #
    #     correlation = nominator / denominator
    #
    #     return 1 - correlation
    #

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
    # reference_sample = sample_dataset_global(unaligned_xmap, transformed_sample_region)

    # Mask
    mask_grid = get_mask(dataset,
                         alignment,
                         transformed_sample_region,
                         structure_factors,
                         sample_rate)

    masked_xmap = mask_xmap(unaligned_xmap,
                            mask_grid)

    initial_rscc = sample_rscc(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        transformed_sample_region,
        masked_xmap,
        reference_sample,
    )

    # Optimise sample
    res = scipy.optimize.shgo(
        lambda perturbation: sample_rscc(
            perturbation,
            transformed_sample_region,
            masked_xmap,
            reference_sample,
        ),
        [(-3, 3), (-3, 3), (-3, 3), (-180.0, 180.0), (-180.0, 180.0), (-180.0, 180.0), ],
        n=60, iters=5, sampling_method='sobol',
    )

    # Get the optimised sample
    sample_arr = sample_dataset_global_perturbed(res.x, transformed_sample_region,
                                                 unaligned_xmap, )

    return 1 - res.fun, sample_arr, res.x, transformed_sample_region, initial_rscc


def sample_reference_dataset_global(dataset, alignment, sample_region, structure_factors,
                                    sample_rate, mask):
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

    masked_xmap = mask_xmap(unaligned_xmap, mask)

    # Get sample
    reference_sample = sample_dataset_global(masked_xmap, transformed_sample_region)

    return reference_sample


def get_mask(dataset: Dataset, alignment, sample_region, structure_factors,
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

    # mask
    mask = gemmi.FloatGrid(unaligned_xmap.nu,
                           unaligned_xmap.nv,
                           unaligned_xmap.nw, )
    mask.set_unit_cell(unaligned_xmap.unit_cell)
    mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    for model in dataset.structure:
        for chain in model:
            for residue in chain.get_polymer():
                for atom in residue:
                    mask.set_points_around(atom.pos, 3.0, 1.0)

    # Get sample
    # reference_sample = sample_dataset_global(mask, transformed_sample_region)

    return mask


def get_alignment(reference: gemmi.Structure, other: gemmi.Structure, monomerized=False):
    ca_self = []
    ca_other = []

    # Get CAs
    matched = 0
    total = 0
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

    print(f"Aligned {matched} of {total}")

    # Make coord matricies
    matrix_self = np.array(ca_self)
    matrix_other = np.array(ca_other)

    # Find means
    mean_self = np.mean(matrix_self, axis=0)
    mean_other = np.mean(matrix_other, axis=0)

    # demaen
    de_meaned_self = matrix_self - mean_self
    de_meaned_other = matrix_other - mean_other

    # Align
    rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned_self, de_meaned_other)

    # Get transform
    vec = np.array([0.0, 0.0, 0.0])
    # Transform is from other frame to self frame
    transform = TransformGlobal.from_translation_rotation(vec,
                                                          rotation,
                                                          mean_other,
                                                          mean_self,
                                                          )

    return transform


def resample(
        reference_xmap: gemmi.FloatGrid,
        moving_xmap: gemmi.FloatGrid,
        reference_structure: gemmi.Structure,
        moving_structure: gemmi.Structure,
        monomerized=False,
):
    # Get transform: from ref to align
    transform = get_alignment(moving_structure, reference_structure, monomerized=monomerized)
    print(f"Transform: {transform}; {transform.transform.vec} {transform.transform.mat}")

    interpolated_grid = gemmi.FloatGrid(
        reference_xmap.nu,
        reference_xmap.nv,
        reference_xmap.nw,
    )
    interpolated_grid.set_unit_cell(reference_xmap.unit_cell)
    interpolated_grid.spacegroup = reference_xmap.spacegroup

    # points
    mask = gemmi.FloatGrid(reference_xmap.nu,
                           reference_xmap.nv,
                           reference_xmap.nw, )
    mask.set_unit_cell(reference_xmap.unit_cell)
    mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    for model in reference_structure:
        for chain in model:
            for residue in chain.get_polymer():
                for atom in residue:
                    mask.set_points_around(atom.pos, 3.0, 1.0)

    mask_array = np.array(mask)
    mask_indicies = np.hstack([x.reshape((len(x), 1)) for x in np.nonzero(mask)])
    print(f"Mask indicies shape: {mask_indicies.shape}")

    fractional_coords = []
    for model in reference_structure:
        for chain in model:
            for residue in chain.get_polymer():
                for atom in residue:
                    fractional = reference_xmap.unit_cell.fractionalize(atom.pos)
                    fractional_coords.append([fractional.x, fractional.y, fractional.z])

    fractional_coords_array = np.array(fractional_coords)
    max_coord = np.max(fractional_coords_array, axis=0)
    min_coord = np.min(fractional_coords_array, axis=0)

    min_index = np.floor(min_coord * np.array([interpolated_grid.nu, interpolated_grid.nv, interpolated_grid.nw]))
    max_index = np.floor(max_coord * np.array([interpolated_grid.nu, interpolated_grid.nv, interpolated_grid.nw]))

    points = itertools.product(range(int(min_index[0]), int(max_index[0])),
                               range(int(min_index[1]), int(max_index[1])),
                               range(int(min_index[2]), int(max_index[2])),
                               )

    # Unpack the points, poitions and transforms
    point_list: List[Tuple[int, int, int]] = []
    position_list: List[Tuple[float, float, float]] = []
    transform_list: List[gemmi.transform] = []
    com_moving_list: List[np.array] = []
    com_reference_list: List[np.array] = []

    transform_rotate_reference_to_moving = transform.transform
    transform_rotate_reference_to_moving.vec.fromlist([0.0, 0.0, 0.0])

    transform_reference_to_centered = gemmi.Transform()
    transform_reference_to_centered.vec.fromlist((-transform.com_reference).tolist())
    transform_reference_to_centered.mat.fromlist(np.eye(3).tolist())

    tranform_centered_to_moving = gemmi.Transform()
    tranform_centered_to_moving.vec.fromlist(transform.com_moving.tolist())
    tranform_centered_to_moving.mat.fromlist(np.eye(3).tolist())

    # indicies to positions
    for point in points:
        if mask.get_value(*point) < 1.0:
            continue

        # get position
        position = interpolated_grid.get_position(*point)

        # Tranform to origin frame
        position_origin_reference = gemmi.Position(transform_reference_to_centered.apply(position))

        # Rotate
        position_origin_moving = gemmi.Position(transform_rotate_reference_to_moving.apply(position_origin_reference))

        # Transform to moving frame
        position_moving = gemmi.Position(tranform_centered_to_moving.apply(position_origin_moving))

        # Interpolate moving map
        interpolated_map_value = moving_xmap.interpolate_value(position_moving)

        # Set original point
        interpolated_grid.set_value(point[0], point[1], point[2], interpolated_map_value)

    interpolated_grid.symmetrize_max()

    return interpolated_grid


def get_xmap(dataset: Dataset, structure_factors, sample_rate):
    reflections: gemmi.Mtz = dataset.reflections
    unaligned_xmap: gemmi.FloatGrid = reflections.transform_f_phi_to_map(
        structure_factors.f,
        structure_factors.phi,
        sample_rate=sample_rate,
    )
    unaligned_xmap_array = np.array(unaligned_xmap, copy=False)
    std = np.std(unaligned_xmap_array)
    unaligned_xmap_array[:, :, :] = unaligned_xmap_array[:, :, :] / std

    return unaligned_xmap


def get_rscc(_reference_grid, _grid, reference_structure):

    mask = gemmi.FloatGrid(_reference_grid.nu,
                           _reference_grid.nv,
                           _reference_grid.nw, )
    mask.set_unit_cell(_reference_grid.unit_cell)
    mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    for model in reference_structure:
        for chain in model:
            for residue in chain.get_polymer():
                for atom in residue:
                    mask.set_points_around(atom.pos, 3.0, 1.0)

    _reference_sample = np.array(_reference_grid, copy=False)
    _sample = np.array(_grid, copy=False)
    _mask = np.array(mask, copy=False)

    reference_flattened = _reference_sample[_mask!= 0]
    sample_flattened = _sample[_mask!=0]

    reference_sample_mean = np.mean(reference_flattened)
    reference_sample_demeaned = reference_flattened - reference_sample_mean
    reference_sample_denominator = np.sqrt(np.sum(np.square(reference_sample_demeaned)))

    sample_mean = np.mean(sample_flattened)
    sample_demeaned = sample_flattened - sample_mean
    sample_denominator = np.sqrt(np.sum(np.square(sample_demeaned)))

    nominator = np.sum(reference_sample_demeaned * sample_demeaned)
    denominator = sample_denominator * reference_sample_denominator

    correlation = nominator / denominator

    return correlation


def sample_datasets_global(
        datasets: Datasets,
        structure_factors,
        grid_step,
        sample_rate,
        cutoff,
):
    def get_reference(datasets: Datasets):
        return list(datasets.values())[0]

    samples: MutableMapping[str, np.ndarray] = {}

    reference = get_reference(datasets)

    reference_xmap = get_xmap(
        reference,
        structure_factors,
        sample_rate,
    )

    reference_structure = reference.structure

    moving_xmaps = map(lambda dataset: get_xmap(dataset, structure_factors, sample_rate), datasets.values())

    moving_structures = [dataset.structure for dataset in datasets.values()]

    grids = map(
        lambda moving: resample(
            reference_xmap,
            moving[0],
            reference_structure,
            moving[1],
        ),
        zip(
            moving_xmaps,
            moving_structures
        ),
    )

    samples = {dtag: grid for dtag, grid in zip(datasets.keys(), grids)}

    rsccs = {dtag: get_rscc(samples[reference.dtag], sample, reference_structure) for dtag, sample in samples.items()}

    for dtag, rscc in rsccs.items():
        print(f"{dtag}: {rscc}")

    return samples


def make_mean_map(grids: List[gemmi.FloatGrid]):
    reference_xmap = grids[0]

    arrays_list = [np.array(grid, copy=False) for grid in grids]

    arrays = np.stack(arrays_list, axis=0)
    mean_array = np.mean(arrays, axis=0)

    mean_grid = gemmi.FloatGrid(
        reference_xmap.nu,
        reference_xmap.nv,
        reference_xmap.nw,
    )
    mean_grid.set_unit_cell(reference_xmap.unit_cell)
    mean_grid.spacegroup = reference_xmap.spacegroup

    for point in mean_grid:
        u = point.u
        v = point.v
        w = point.w
        value = mean_array[u, v, w]

        mean_grid.set_poiint(u, v, w, value)

    return mean_grid


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
        output_mean_maps: bool=False,
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

    if not out_dir.exists():
        os.mkdir(str(out_dir))

    if params.debug:
        print(
            (
                "Input:\n"
                f"\tdata_dir: {data_dir}\n"
                f"\tout dir: {out_dir}\n"
                f"\tknown apos: {known_apos}\n"
                f"\treference_dtag: {reference_dtag}\n"
                f"\tmarkers: {markers}\n"
                f"\tStructure factors: {params.structure_factors}\n"
                f"\tMtz regex: {reflections_regex}\n"
                f"\tStructure regex: {structure_regex}\n"
                f"\tOutput mean maps: {output_mean_maps}\n"
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

    # Filter valid structure factors
    filtered_datasets_sfs = {
        dtag: datasets[dtag]
        for dtag
        in filter(lambda dtag: filter_sfs(datasets[dtag], params.structure_factors), datasets)
    }
    print(f"Filtered datasets on structure factors: {[dtag for dtag in datasets if not dtag in filtered_datasets_sfs]}")

    # Filter invalid structures
    filtered_datasets_struc = {
        dtag: filtered_datasets_sfs[dtag]
        for dtag
        in filter(lambda dtag: filter_structure(filtered_datasets_sfs[dtag], reference_dataset), filtered_datasets_sfs)
    }
    print(
        f"Filtered datasets on structure match: {[dtag for dtag in filtered_datasets_sfs if not dtag in filtered_datasets_struc]}")

    # B factor smooth the datasets
    smoothed_datasets: MutableMapping[str, Dataset] = smooth_datasets(
        filtered_datasets_struc,
        reference_dataset,
        params.structure_factors,
    )

    # Get the markers for alignment
    # markers: List[Marker] = get_markers(reference_dataset, markers)

    # Find the alignments between the reference and all other datasets
    # alignments: MutableMapping[str, Alignment] = get_global_alignments(smoothed_datasets, reference_dataset, )

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

    sample_grids: MutableMapping[str, gemmi.FloatGrid] = sample_datasets_global(
        truncated_datasets,
        params.structure_factors,
        params.grid_spacing,
        params.sample_rate,
        cutoff=0.7,
    )

    sample_arrays = {dtag: np.array(grid, copy=False) for dtag, grid in sample_grids.items()}
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

    if output_mean_maps:
        print(f"Outputting mean maps...")
        dtag_array = np.array(list(sample_grids.keys()))
        cluster_dtags: MutableMapping[int, List[str]] = {
            cluster_num: [dtag for dtag in dtag_array[dataset_clusters == cluster_num]]
            for cluster_num
            in dataset_clusters
        }

        for cluster_num in cluster_dtags:
            mean_map: gemmi.FloatGrid = make_mean_map([sample_grids[dtag] for dtag in cluster_dtags[cluster_num]])

            save_ccp4(
                out_dir / f"mean_map_{cluster_num}.ccp4",
                mean_map,
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

    save_embed_umap_plot(
        distance_matrix,
        out_dir / f"embed_umap.png"
    )

    save_umap_plot(
        distance_matrix,
        out_dir / f"umap.png"
    )

    # Interactive plots
    labels = list(sample_arrays.keys())
    save_plot_tsne_bokeh(distance_matrix, labels, out_dir / f"embed.html")
    save_plot_umap_bokeh(distance_matrix, labels, out_dir / f"umap.html")
    save_plot_pca_umap_bokeh(distance_matrix, labels, out_dir / f"pca_umap.html")

    # Store resullts
    clusters = {
        dtag: cluster_id
        for dtag, cluster_id
        in zip(sample_arrays.keys(), dataset_clusters.flatten().tolist())
    }

    print(f"Clusters: {clusters}")
    # End loop over residues
