import torch
import numpy as np

from GNN_MODELS import GCN, SAGE, GIN, GAT
from GNN_INPUT import prepare_gnn_inputs


def create_data_splits(y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Create boolean masks for train / validation / test nodes.

    Inputs:
    - y: label tensor of shape [num_nodes]
    - train_ratio: fraction of nodes for training
    - val_ratio: fraction of nodes for validation
    - test_ratio: fraction of nodes for testing
    - seed: random seed

    Returns:
    - train_mask, val_mask, test_mask
    """

    num_nodes = len(y)

    torch.manual_seed(seed)
    perm = torch.randperm(num_nodes)

    train_end = int(train_ratio * num_nodes)
    val_end = train_end + int(val_ratio * num_nodes)

    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def build_model(model_name, in_channels, hidden_channels, out_channels):
    """
    Build one GNN model by name.
    """
    if model_name == "GCN":
        model = GCN(
            in_channels=in_channels,
            all_hidden_channels=hidden_channels,
            out_channels=out_channels,
            activation=None
        )
    elif model_name == "SAGE":
        model = SAGE(
            in_channels=in_channels,
            all_hidden_channels=hidden_channels,
            out_channels=out_channels,
            activation=None
        )
    elif model_name == "GIN":
        model = GIN(
            in_channels=in_channels,
            all_hidden_channels=hidden_channels,
            out_channels=out_channels,
            activation=None
        )
    elif model_name == "GAT":
        model = GAT(
            in_channels=in_channels,
            all_hidden_channels=hidden_channels,
            out_channels=out_channels,
            activation=None
        )
    else:
        raise ValueError("Invalid model_name. Choose from 'GCN', 'SAGE', 'GIN', 'GAT'.")

    return model


def evaluate(model, X, edge_index, edge_weight, y, mask, model_name):
    """
    Compute loss and accuracy on the nodes selected by mask.
    """
    model.eval()

    with torch.no_grad():
        if model_name == "GCN":
            out = model(X, edge_index, edge_weight)
        else:
            out = model(X, edge_index)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(out[mask], y[mask])

        pred = out.argmax(dim=1)
        correct = (pred[mask] == y[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0.0

    return loss.item(), acc


def train_gnn(
    model_name,
    X,
    edge_index,
    edge_weight,
    y,
    train_mask,
    val_mask,
    test_mask,
    hidden_channels=[128],
    learning_rate=0.01,
    weight_decay=5e-4,
    num_epochs=100
):
    """
    Train one GNN model and report train/val/test results.
    """
    in_channels = X.shape[1]
    out_channels = len(torch.unique(y))

    model = build_model(
        model_name=model_name,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        if model_name == "GCN":
            out = model(X, edge_index, edge_weight)
        else:
            out = model(X, edge_index)

        loss = loss_fn(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        train_loss, train_acc = evaluate(
            model, X, edge_index, edge_weight, y, train_mask, model_name
        )
        val_loss, val_acc = evaluate(
            model, X, edge_index, edge_weight, y, val_mask, model_name
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {
                key: value.clone()
                for key, value in model.state_dict().items()
            }

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_acc = evaluate(
        model, X, edge_index, edge_weight, y, test_mask, model_name
    )

    print("\nBest validation accuracy:", round(best_val_acc, 4))
    print("Test loss:", round(test_loss, 4))
    print("Test accuracy:", round(test_acc, 4))

    return model


if __name__ == "__main__":
    # -----------------------------------
    # 1. Choose model here
    # -----------------------------------
    model_name = "GCN"

    # -----------------------------------
    # 2. File paths
    # -----------------------------------
    input_folder = "data/test_plays"
    chunked_dataset_file = "data/chunked_plays.csv"
    nodes_csv = "data/graph_nodes.csv"
    edges_csv = "data/graph_edges.csv"
    embeddings_npy = "data/chunk_embeddings.npy"

    # -----------------------------------
    # 3. Function words
    # -----------------------------------
    function_words = [
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on",
        "for", "with", "as", "at", "by", "from", "that", "this", "it",
        "he", "she", "i", "you", "we", "they", "is", "was", "be", "been",
        "are", "were", "not", "do", "does", "did", "have", "has", "had"
    ]

    # -----------------------------------
    # 4. Prepare graph tensors
    # -----------------------------------
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
        distance_type="bhattacharyya",
        model_name="gpt2",
        batch_size=4,
        max_length=256,
        graph_mode="annoy",
        k=3,
        num_trees=20,
        search_k=-1,
        renyi_alpha=0.5
    )

    print("\nData loaded.")
    print("X shape:", X.shape)
    print("edge_index shape:", edge_index.shape)
    print("edge_weight shape:", edge_weight.shape)
    print("y shape:", y.shape)

    # -----------------------------------
    # 5. Create train / val / test masks
    # -----------------------------------
    train_mask, val_mask, test_mask = create_data_splits(
        y,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42
    )

    print("\nSplit sizes:")
    print("Train nodes:", train_mask.sum().item())
    print("Val nodes:", val_mask.sum().item())
    print("Test nodes:", test_mask.sum().item())

    # -----------------------------------
    # 6. Train model
    # -----------------------------------
    model = train_gnn(
        model_name=model_name,
        X=X,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        hidden_channels=[128],
        learning_rate=0.01,
        weight_decay=5e-4,
        num_epochs=100
    )