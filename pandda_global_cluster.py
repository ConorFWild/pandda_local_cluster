# First party
from typing import *
from pathlib import Path
import time

# Third party
import fire
import numpy as np

# Custom
from pandda_local_cluster.datatypes import (
    Dataset,
    Alignment,
    Params,
    Marker,
)

from pandda_local_cluster.functions import (
    iterate_markers,
    get_datasets,
    get_alignments,
    get_reference,
    smooth_datasets,
    save_mtz,
    get_markers,
    get_truncated_datasets,
    sample_datasets,
    sample_dataset,
    sample_datasets_refined,
    get_distance_matrix,
    get_linkage_from_correlation_matrix,
    cluster_density,
    save_json,
    save_dendrogram_plot,
    save_hdbscan_dendrogram,
    save_embed_plot,
    get_global_distance_matrix,
    save_num_clusters_bar_plot,
    save_num_clusters_stacked_bar_plot,
    save_global_cut_curve,
    save_parallel_cat_plot,
    save_correlation_plot,
    sample_datasets_refined_iterative,
    save_distance_matrix,
    save_dtag_array,
)

from pandda_local_cluster.functions_gloval_cluster import (
get_global_alignment,
get_sample_region,
sample_datasets_global,

)


def run_local_cluster(
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


if __name__ == "__main__":
    fire.Fire(run_local_cluster)
