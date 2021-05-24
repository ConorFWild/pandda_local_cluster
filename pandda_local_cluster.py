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
    filter_sfs,
    filter_structure,
    output_mean_maps_local
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
        datasets,
        reference_dataset,
        params.structure_factors,
    )
    # if params.output_smoothed_mtzs:
    #     for dtag, smoothed_dataset in smoothed_datasets.items():
    #         save_mtz(smoothed_dataset.reflections, out_dir / f"{smoothed_dataset.dtag}_smoothed.mtz")

    # Get the markers to sample around
    markers: List[Marker] = get_markers(reference_dataset, markers)

    # Find the alignments between the reference and all other datasets
    alignments: MutableMapping[str, Alignment] = get_alignments(smoothed_datasets, reference_dataset, markers)

    # Loop over the residues, sampling the local electron density
    marker_clusters = {}
    for marker, marker_datasets in iterate_markers(datasets, markers, alignments):

        # if marker.resid.insertion != "29":
        #     continue

        print(f"Processing marker: {marker}")
        print(f"Got number of datasets: {len(marker_datasets)}")
        print(f"Datasets are: {marker_datasets}")

        if len(marker_datasets) == 0:
            print(f"No datasets at residue: {marker}!")
            continue

        if params.debug:
            print(f"Processing residue: {marker}")

        # Truncate the datasets to the same reflections
        truncated_datasets: MutableMapping[str, Dataset] = get_truncated_datasets(
            marker_datasets,
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
        # sample_arrays: MutableMapping[str, np.ndarray] = sample_datasets(
        #     truncated_datasets,
        #     marker,
        #     alignments,
        #     params.structure_factors,
        #     params.sample_rate,
        #     params.grid_size,
        #     params.grid_spacing,
        # )
        #
        reference_sample = sample_dataset(reference_dataset,
                                          alignments[reference_dataset.dtag][marker],
                                          marker,
                                          params.structure_factors,
                                          params.sample_rate,
                                          params.grid_size,
                                          params.grid_spacing,
                                          )
        sample_arrays: MutableMapping[str, np.ndarray] = sample_datasets_refined(
            truncated_datasets,
            marker,
            alignments,
            reference_sample,
            params.structure_factors,
            params.sample_rate,
            params.grid_size,
            params.grid_spacing,
        )
        # sample_arrays = sample_datasets_refined_iterative(
        #     truncated_datasets,
        #     marker,
        #     alignments,
        #     known_apos,
        #     params.structure_factors,
        #     params.sample_rate,
        #     params.grid_size,
        #     params.grid_spacing,
        #     0.7
        # )
        print(f"Got {len(sample_arrays)} samples")

        # Get the distance matrix
        distance_matrix: np.ndarray = get_distance_matrix(sample_arrays)
        if params.debug:
            print(f"First line of distance matrix: {distance_matrix[0, :]}")
            print(f"Lasy line of distance matrix: {distance_matrix[-1, :]}")

        # Get the distance matrix linkage
        linkage: np.ndarray = get_linkage_from_correlation_matrix(distance_matrix)

        # Cluster the available density
        dataset_clusters: np.ndarray = cluster_density(
            linkage,
            params.local_cluster_cutoff,
        )

        # Output
        output_mean_maps_local(
            sample_arrays, dataset_clusters, reference_dataset, marker, out_dir,
            params.grid_size, params.grid_spacing, params.structure_factors, params.sample_rate
        )

        save_distance_matrix(distance_matrix,
                             out_dir / f"{marker.resid.model}_{marker.resid.chain}_{marker.resid.insertion}.npy")

        save_dtag_array(np.array(list(sample_arrays.keys())),
                        out_dir / f"{marker.resid.model}_{marker.resid.chain}_{marker.resid.insertion}_dtags.npy",
                        )

        save_dendrogram_plot(linkage,
                             labels=[dtag for dtag in sample_arrays.keys()],
                             dendrogram_plot_file=out_dir / f"{marker.resid}_dendrogram.png",
                             )

        save_hdbscan_dendrogram(
            distance_matrix,
            out_dir / f"{marker.resid}_hdbscan_dendrogram.png",
        )

        save_embed_plot(
            distance_matrix,
            out_dir / f"{marker.resid}_embed.png"
        )

        # Store resullts
        marker_clusters[marker] = {
            dtag: cluster_id
            for dtag, cluster_id
            in zip(sample_arrays.keys(), dataset_clusters.flatten().tolist())
        }

        print(f"Marker clusters: {marker_clusters}")

    # End loop over residues

    save_num_clusters_bar_plot(marker_clusters, out_dir / f"global_residue_cluster_bar.png")

    save_num_clusters_stacked_bar_plot(marker_clusters, out_dir / f"global_residue_cluster_stacked_bar.png")

    # Perform global clustering
    global_distance_matrix = get_global_distance_matrix(marker_clusters, markers, smoothed_datasets)
    print(global_distance_matrix)
    print(global_distance_matrix[0, :])
    print(global_distance_matrix[-1, :])

    time.sleep(60)

    global_linkage: np.ndarray = get_linkage_from_correlation_matrix(global_distance_matrix)
    global_clusters: np.ndarray = cluster_density(
        global_linkage,
        params.global_cluster_cutoff,
    )

    # Output
    # Save json
    save_json(marker_clusters,
              out_dir / f"clusterings.json",
              )

    # # Get dtag list
    dtag_list = list(list(marker_clusters.values())[0].keys())

    # # Connectivity
    # save_correlation_plot(global_distance_matrix,
    #                       out_dir / f"global_connectivity_correlation.png",
    #                       )
    #
    # # # Summary plots
    # save_dendrogram_plot(global_linkage,
    #                      labels=[dtag for dtag in dtag_list],
    #                      dendrogram_plot_file=out_dir / f"global_connectivity_dendrogram.png",
    #                      )

    save_embed_plot(global_distance_matrix, out_dir / f"global_embed_scatter.png")

    save_global_cut_curve(global_linkage, out_dir / f"global_cut_curve.png")

    save_hdbscan_dendrogram(
        global_distance_matrix,
        out_dir / f"global_hdbscan_dendrogram.png",
    )

    save_parallel_cat_plot(
        marker_clusters,
        out_dir / f"parallel_cat_plot.png",
    )


if __name__ == "__main__":
    fire.Fire(run_local_cluster)
