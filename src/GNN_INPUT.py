import numpy as np
import pandas as pd
import torch

from GRAPH_CONSTRUCTION import graph_construction
from LLM_EMBEDDING import generate_chunk_embeddings


def load_gnn_inputs(nodes_csv, edges_csv, embeddings_npy):
    """
    Load graph nodes, graph edges, and chunk embeddings,
    then convert them into GNN-ready tensors.
    """
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    embeddings = np.load(embeddings_npy)

    num_nodes = len(nodes_df)

    if embeddings.shape[0] != num_nodes:
        raise ValueError(
            f"Number of embeddings ({embeddings.shape[0]}) does not match "
            f"number of nodes ({num_nodes})."
        )

    X = torch.tensor(embeddings, dtype=torch.float)
    y = torch.tensor(nodes_df["author_label"].values, dtype=torch.long)

    edge_index = torch.tensor(
        edges_df[["source", "target"]].values.T,
        dtype=torch.long
    )

    edge_weight = torch.tensor(
        edges_df["weight"].values,
        dtype=torch.float
    )

    return X, edge_index, edge_weight, y


def prepare_gnn_inputs(
    input_folder,
    chunked_dataset_file,
    nodes_csv,
    edges_csv,
    embeddings_npy,
    function_words,
    chunk_size=5000,
    D=10,
    alpha=0.75,
    epsilon=1e-12,
    distance_type="kl",
    model_name="gpt2",
    batch_size=4,
    max_length=256
):
    """
    Full pipeline before GNN:

    1. Build chunked dataset + graph files
    2. Generate embeddings
    3. Load X, edge_index, edge_weight, y
    """

    # -----------------------------------
    # 1. Build graph (this already builds dataset inside)
    # -----------------------------------
    print("\n[1] Building graph...")
    graph_construction(
        input_folder=input_folder,
        chunked_dataset_file=chunked_dataset_file,
        nodes_output_file=nodes_csv,
        edges_output_file=edges_csv,
        function_words=function_words,
        chunk_size=chunk_size,
        D=D,
        alpha=alpha,
        epsilon=epsilon,
        distance_type=distance_type
    )

    # -----------------------------------
    # 2. Generate embeddings
    # -----------------------------------
    print("\n[2] Generating embeddings...")
    generate_chunk_embeddings(
        input_csv=chunked_dataset_file,
        output_npy=embeddings_npy,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length
    )

    # -----------------------------------
    # 3. Load GNN tensors
    # -----------------------------------
    print("\n[3] Loading GNN tensors...")
    X, edge_index, edge_weight, y = load_gnn_inputs(
        nodes_csv=nodes_csv,
        edges_csv=edges_csv,
        embeddings_npy=embeddings_npy
    )

    return X, edge_index, edge_weight, y


if __name__ == "__main__":
    input_folder = "data/test_plays"
    chunked_dataset_file = "data/test_plays.csv"
    nodes_csv = "data/test_graph_nodes.csv"
    edges_csv = "data/test_graph_edges.csv"
    embeddings_npy = "data/test_chunk_embeddings.npy"

    function_words = [
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on",
        "for", "with", "as", "at", "by", "from", "that", "this", "it",
        "he", "she", "i", "you", "we", "they", "is", "was", "be", "been",
        "are", "were", "not", "do", "does", "did", "have", "has", "had"
    ]

    X, edge_index, edge_weight, y = prepare_gnn_inputs(
        input_folder=input_folder,
        chunked_dataset_file=chunked_dataset_file,
        nodes_csv=nodes_csv,
        edges_csv=edges_csv,
        embeddings_npy=embeddings_npy,
        function_words=function_words,
        chunk_size=5000,
        D=10,
        alpha=0.75,
        epsilon=1e-12,
        distance_type="kl",
        model_name="gpt2",
        batch_size=4,
        max_length=256
    )

    print("\nFinished preparing GNN inputs.")
    print("X shape:", X.shape)
    print("edge_index shape:", edge_index.shape)
    print("edge_weight shape:", edge_weight.shape)
    print("y shape:", y.shape)