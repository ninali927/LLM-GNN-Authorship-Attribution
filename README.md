# LLM + GNN for Authorship Attribution

## Overview

This project implements an authorship attribution pipeline that combines:

- **LLM-based embeddings** (semantic features)  
- **Word Adjacency Networks (WANs)** (stylistic features)  
- **Graph Neural Networks (GNNs)** (for classification)  

Each text chunk is treated as a node in a graph. Edges between nodes are constructed using WAN-based distances, and node features are obtained from a language model. A GNN is then trained to predict the author of each text chunk.

---

## Pipeline

The full pipeline consists of the following steps:

### 1. Dataset Construction
- Raw play texts are loaded from `data/raw_texts_plays/` or ``data/test_plays/`
- Each play is split into fixed-length chunks
- Output: `chunked_plays.csv`

### 2. Graph Construction (WAN-based)
- Each chunk becomes a node
- For each pair of chunks:
  - Build WAN (Word Adjacency Network)
  - Compute distance (e.g., KL divergence)
  - Convert distance to similarity weight
- Output:
  - `graph_nodes.csv`
  - `graph_edges.csv`

### 3. LLM Embeddings
- Each chunk is converted into a vector using a language model (e.g., GPT-2)
- Output:
  - `chunk_embeddings.npy`

### 4. GNN Input Preparation
- Load:
  - node features (embeddings)
  - edge_index (graph structure)
  - edge_weight
  - labels (author)
- Output:
  - tensors ready for training

### 5. GNN Training
- Supported models:
  - GCN
  - GraphSAGE
  - GIN
  - GAT
- Currently using:
  - **GCN with edge weights**
- Task:
  - node classification (predict author)

---

## Project Structure

```text
project/
├── data/
│   ├── test_plays/
│   └── raw_texts_plays/
│
├── src/
│   ├── preprocess/
│   │   ├── remove_extra_spaces.py
│   │   ├── annotate_and_mask.py
│   │   ├── split_sentences_from_annotation.py
│   │   ├── preprocess_pipeline.py
│   │   └── test_preprocess.py
│   │
│   ├── WAN/
│   │   ├── function_words.py
│   │   ├── wan_matrix.py
│   │   ├── markov_normalization.py
│   │   ├── wan_distance.py
│   │   ├── WAN_pipeline.py
│   │   ├── relative_entropy/
│   │   │   ├── Bhattacharyya_Distance.py
│   │   │   ├── Hellinger_Distance.py
│   │   │   ├── Jensen_Shannon_Divergence.py
│   │   │   ├── Kullback_Leibler_Divergence.py
│   │   │   ├── Renyi_Divergence.py
│   │   │   └── Total_Variation_Distance.py
│   │   └── test_WAN.py
│   │
│   ├── build_dataset.py
│   ├── GNN_INPUT.py
│   ├── GNN_MODELS.py
│   ├── GRAPH_CONSTRUCTION_PAIRWISE.py
│   ├── LLM_EMBEDDING.py
│   └── TRAIN_GNN.py
│
└── README.md
```

---

## Run full pipeline + training
```bash
python src/TRAIN_GNN.py
```

This will:
	1.	Build dataset
	2.	Construct graph (WAN)
	3.	Generate embeddings
	4.	Train GNN
	5.	Print accuracy