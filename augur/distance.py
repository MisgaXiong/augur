"""Calculate the distance between amino acid sequences across entire genes or at a predefined subset of sites.
"""
import argparse
import Bio
import Bio.Phylo
from collections import defaultdict
import copy
import json
import numpy as np
import os
import sys

from .reconstruct_sequences import load_alignments


def read_masks(mask_file):
    ha_masks = {}
    with open(mask_file) as f:
        for line in f:
            (key, value) = line.strip().split()
            ha_masks[key] = np.frombuffer(value.encode(), 'S1').astype(int).astype(bool)

    return ha_masks


def read_distance_map(map_file):
    """Read a distance map JSON into a dictionary and assert that the JSON follows
    the correct format. Coordinates should be one-based in the JSON and are
    converted to zero-based coordinates on load.

    Parameters
    ----------
    map_file : str
        name of a JSON file containing a valid distance map

    Returns
    -------
    dict :
        Python representation of the distance map JSON

    >>> sorted(read_distance_map("tests/data/distance_map_weight_per_site.json").items())
    [('default', 0), ('map', {'HA1': {144: 1}})]

    """
    # Load the JSON.
    with open(map_file, "r") as fh:
        json_distance_map = json.load(fh)

    # Confirm that all required fields are present.
    assert "default" in json_distance_map
    assert "map" in json_distance_map

    # Convert coordinate numbers from strings to integers.
    distance_map = copy.deepcopy(json_distance_map)
    for gene, site_weights in json_distance_map["map"].items():
        for site, weights in site_weights.items():
            # Convert each one-based site string to a zero-based integer.
            distance_map["map"][gene][int(site) - 1] = weights
            del distance_map["map"][gene][site]

    # Return the distance map.
    return distance_map


def get_distance_between_nodes(node_a_sequences, node_b_sequences, distance_map):
    """Calculate distance between the two given nodes using the given distance map.

    In cases where the distance map between sequences is asymmetric, the first
    node is interpreted as the "ancestral" sequence and the second node is
    interpreted as the "derived" sequence.

    Parameters
    ----------
    node_a_sequences, node_b_sequences : dict
        sequences by gene name for two nodes (samples) in a tree

    distance_map : dict
        definition of site-specific and, optionally, sequence-specific distances
        per gene

    Returns
    -------
    int or float :
        distance between node sequences based on the given map where the
        returned type matches the type of the default value

    >>> node_a_sequences = {"gene": "ACTG"}
    >>> node_b_sequences = {"gene": "ACGG"}
    >>> distance_map = {"default": 0, "map": {}}
    >>> get_distance_between_nodes(node_a_sequences, node_b_sequences, distance_map)
    0
    >>> distance_map = {"default": 1, "map": {}}
    >>> get_distance_between_nodes(node_a_sequences, node_b_sequences, distance_map)
    1
    >>> distance_map = {"default": 0.0, "map": {"gene": {3: 1.0}}}
    >>> get_distance_between_nodes(node_a_sequences, node_b_sequences, distance_map)
    0.0
    >>> distance_map = {"default": 0.0, "map": {"gene": {2: 0.5, 3: 1.0}}}
    >>> get_distance_between_nodes(node_a_sequences, node_b_sequences, distance_map)
    0.5

    """
    distance_type = type(distance_map["default"])
    distance = distance_type(0)

    for gene in node_a_sequences:
        gene_length = len(node_a_sequences[gene])

        for site in range(gene_length):
            if node_a_sequences[gene][site] != node_b_sequences[gene][site]:
                if gene in distance_map["map"] and site in distance_map["map"][gene]:
                    # Assume distances are site- and sequence-specific
                    # first. Failing that, distances must be site-specific.
                    seq_ancestral = node_a_sequences[gene][site]
                    seq_derived = node_b_sequences[gene][site]

                    try:
                        distance += distance_map["map"][gene][site][(seq_ancestral, seq_derived)]
                    except TypeError:
                        distance += distance_map["map"][gene][site]
                else:
                    distance += distance_map["default"]

    return distance_type(distance)


def get_distances_to_root(tree, sequences_by_node_and_gene, distance_map):
    """Calculate distances between all samples in the given sequences and the node
    of the given tree using the given distance map.

    Parameters
    ----------
    tree : Bio.Phylo
        a rooted tree whose node names match the given dictionary of sequences
        by node and gene

    sequences_by_node_and_gene : dict
        nucleotide or amino acid sequences by node name and gene

    distance_map : dict
        site-specific and, optionally, sequence-specific distances between two
        sequences

    Returns
    -------
    dict :
        distances calculated between the root sequence and each sample in the
        tree and indexed by node name

    """
    distances_by_node = {}

    # Find the root node's sequences.
    root_node_sequences = sequences_by_node_and_gene[tree.root.name]

    # Calculate distance between root and all other nodes.
    for node_name, node_sequences in sequences_by_node_and_gene.items():
        distances_by_node[node_name] = get_distance_between_nodes(
            root_node_sequences,
            node_sequences,
            distance_map
        )

    return distances_by_node


def mask_sites(aa, mask):
    return aa[mask[:len(aa)]]


def mask_distance(aaA, aaB, mask):
    """Return distance of sequences aaA and aaB by comparing sites in the given binary mask.

    >>> aaA = np.array(["A", "B", "C"], dtype="S1")
    >>> aaB = np.array(["A", "B", "D"], dtype="S1")
    >>> mask = np.array([0, 1, 1], dtype=np.bool)
    >>> mask_distance(aaA, aaB, mask)
    1
    >>> aaB = np.array(["A", "B", "X"], dtype="S1")
    >>> mask_distance(aaA, aaB, mask)
    0
    """
    sites_A = mask_sites(aaA, mask)
    sites_B = mask_sites(aaB, mask)

    # Count sites that differ between sequences excluding undetermined residues.
    distance = int(np.sum((sites_A != sites_B) & (sites_A != b"X") & (sites_B != b"X")))

    return distance


def register_arguments(parser):
    parser.add_argument("--tree", help="Newick tree", required=True)
    parser.add_argument("--alignment", nargs="+", help="sequence(s) to be used, supplied as FASTA files", required=True)
    parser.add_argument('--gene-names', nargs="+", type=str, help="names of the sequences in the alignment, same order assumed", required=True)
    parser.add_argument("--compare-to", choices=["root", "ancestor", "pairwise"], help="type of comparison between samples in the given tree", required=True)
    parser.add_argument("--attribute-name", help="name to store distances associated with the given distance map", required=True)
    parser.add_argument("--map", help="JSON providing the distance map between sites and, optionally, amino acids at those sites", required=True)
    parser.add_argument("--latest-date", help="latest date to consider samples for last ancestor or pairwise comparisons (e.g., 2019-01-01); defaults to the current date")
    parser.add_argument("--output", help="JSON file with calculated distances stored by node name and attribute name", required=True)


def run(args):
    # Load tree.
    tree = Bio.Phylo.read(args.tree, "newick")

    # Load sequences.
    alignments = load_alignments(args.alignment, args.gene_names)

    # Index sequences by node name and gene.
    sequences_by_node_and_gene = defaultdict(dict)
    for gene, alignment in alignments.items():
        for record in alignment:
            sequences_by_node_and_gene[record.name][gene] = str(record.seq)

    # Load the given distance map.
    distance_map = read_distance_map(args.map)

    # Use the distance map to calculate distances between all samples in the
    # given tree and the desired target(s).
    if args.compare_to == "root":
        # Calculate distance between the root and all samples.
        distances_by_node = get_distances_to_root(
            tree,
            sequences_by_node_and_gene,
            distance_map
        )
    elif args.compare_to == "ancestor":
        # Calculate distance between the last ancestor for each sample in a
        # previous season.
        distances_by_node = get_distances_to_last_ancestor(
            tree,
            sequences_by_node_and_gene,
            distance_map,
            args.latest_date
        )
    elif args.compare_to == "pairwise":
        # Calculate distance between each sample and all other samples in a
        # previous season.
        distances_by_node = get_pairwise_distances(
            tree,
            sequences_by_node_and_gene,
            distance_map,
            args.latest_date
        )
    else:
        pass

    # Map distances to the requested attribute name.
    # Convert data like:
    # {
    #   "A/AbuDhabi/24/2017": 1
    # }
    # to data like:
    #
    # {
    #   "A/AbuDhabi/24/2017": {
    #     "ep": 1
    #   }
    # }
    #
    final_distances_by_node = {}
    for node_name, values in distances_by_node.items():
        final_distances_by_node[node_name] = {
            args.attribute_name: values
        }

    # Prepare params for export.
    params = {
        "attribute": args.attribute_name,
        "compare_to": args.compare_to,
        "map_name": distance_map.get("name", args.map),
        "latest_date": args.latest_date
    }

    # Export distances to JSON.
    with open(args.output, "w") as oh:
        json.dump({"params": params, "nodes": final_distances_by_node}, oh, indent=1, sort_keys=True)
