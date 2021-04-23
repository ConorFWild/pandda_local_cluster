from pandda_local_cluster.datatypes import *
from pandda_local_cluster.datatypes_global_cluster import *

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


        ...

    ...
