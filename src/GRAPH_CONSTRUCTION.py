import pandas as pd

from build_dataset import build_dataset
from WAN.WAN_pipeline import WAN_distance_pipeline


def build_author_label_map(df):
    """
    Create a mapping from author name to integer label.
    """
    authors = sorted(df["author"].unique())
    author_to_label = {}

    for i in range(len(authors)):
        author_to_label[authors[i]] = i

    return author_to_label


def distance_to_similarity(distance):
    """
    Convert WAN distance to similarity weight for graph edges.
    """
    return 1.0 / (1.0 + distance)


def create_nodes_dataframe(df):
    """
    Create node dataframe from chunked dataset.
    Each chunk becomes one node.
    """
    author_to_label = build_author_label_map(df)

    rows = []

    for i in range(len(df)):
        row = df.iloc[i]

        rows.append({
            "node_id": i,
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "author": row["author"],
            "author_label": author_to_label[row["author"]],
            "play": row["play"],
            "chunk_index": row["chunk_index"],
            "source_file": row["source_file"],
            "num_words": row["num_words"]
        })

    nodes_df = pd.DataFrame(rows)
    return nodes_df, author_to_label


def build_all_pairs_edges(nodes_df,
                          function_words,
                          D=10,
                          alpha=0.75,
                          epsilon=1e-12,
                          distance_type="kl"):
    """
    Build all-pairs graph edges.

    For every pair of nodes (i, j), compute WAN distance
    and convert it to a similarity weight.
    """
    edge_rows = []
    n = len(nodes_df)

    for i in range(n):
        text_i = nodes_df.iloc[i]["text"]

        for j in range(i + 1, n):
            text_j = nodes_df.iloc[j]["text"]

            try:
                distance = WAN_distance_pipeline(
                    chunk_text_1=text_i,
                    chunk_text_2=text_j,
                    function_words=function_words,
                    D=D,
                    alpha=alpha,
                    epsilon=epsilon,
                    distance_type=distance_type
                )

                weight = distance_to_similarity(distance)

                # add i -> j
                edge_rows.append({
                    "source": i,
                    "target": j,
                    "distance": distance,
                    "weight": weight
                })

                # add j -> i
                edge_rows.append({
                    "source": j,
                    "target": i,
                    "distance": distance,
                    "weight": weight
                })

            except Exception as e:
                print("Error on pair:", i, j)
                print(e)

    edges_df = pd.DataFrame(edge_rows)
    return edges_df


def graph_construction(input_folder,
                       chunked_dataset_file,
                       nodes_output_file,
                       edges_output_file,
                       function_words,
                       chunk_size=5000,
                       D=10,
                       alpha=0.75,
                       epsilon=1e-12,
                       distance_type="kl"):
    """
    Full graph construction pipeline.

    Step 1: build chunked dataset from raw play files
    Step 2: create node table
    Step 3: build all-pairs edges using WAN distance
    Step 4: save nodes and edges
    """

    # 1. Build chunked dataset
    build_dataset(
        input_folder=input_folder,
        output_file=chunked_dataset_file,
        chunk_size=chunk_size
    )
    df = pd.read_csv(chunked_dataset_file)

    # 2. Create nodes
    nodes_df, author_to_label = create_nodes_dataframe(df)

    print("\nAuthor to label mapping:")
    print(author_to_label)

    # 3. Build all-pairs edges
    edges_df = build_all_pairs_edges(
        nodes_df=nodes_df,
        function_words=function_words,
        D=D,
        alpha=alpha,
        epsilon=epsilon,
        distance_type=distance_type
    )

    # 4. Save outputs
    nodes_df.to_csv(nodes_output_file, index=False)
    edges_df.to_csv(edges_output_file, index=False)

    print("\nGraph construction finished.")
    print("Nodes saved to:", nodes_output_file)
    print("Edges saved to:", edges_output_file)
    print("Number of nodes:", len(nodes_df))
    print("Number of edges:", len(edges_df))


if __name__ == "__main__":
    input_folder = "data/test_plays"
    chunked_dataset_file = "data/chunked_plays.csv"
    nodes_output_file = "data/graph_nodes.csv"
    edges_output_file = "data/graph_edges.csv"

    function_words = [
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on",
        "for", "with", "as", "at", "by", "from", "that", "this", "it",
        "he", "she", "i", "you", "we", "they", "is", "was", "be", "been",
        "are", "were", "not", "do", "does", "did", "have", "has", "had"
    ]

    graph_construction(
        input_folder=input_folder,
        chunked_dataset_file=chunked_dataset_file,
        nodes_output_file=nodes_output_file,
        edges_output_file=edges_output_file,
        function_words=function_words,
        chunk_size=5000,
        D=10,
        alpha=0.75,
        epsilon=1e-12,
        distance_type="kl"
    )