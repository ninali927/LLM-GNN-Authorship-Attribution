import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def mean_pooling(last_hidden_state, attention_mask):
    """
    Mean-pool token embeddings, ignoring padding tokens.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_hidden = last_hidden_state * mask
    summed = masked_hidden.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def generate_chunk_embeddings(
    input_csv,
    output_npy,
    model_name="gpt2",
    batch_size=4,
    max_length=256
):
    """
    Read chunked plays CSV and generate one embedding per chunk using
    the same Hugging Face base model family as Method 1.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    df = pd.read_csv(input_csv)

    required_cols = ["chunk_id", "text", "author", "play"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    texts = df["text"].fillna("").tolist()

    # load tokenizer + base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)

            batch_embeddings = mean_pooling(
                outputs.last_hidden_state,
                encoded["attention_mask"]
            )

            all_embeddings.append(batch_embeddings.cpu().numpy())

            print(f"Processed {min(start + batch_size, len(texts))} / {len(texts)} chunks")

    embeddings = np.vstack(all_embeddings)

    output_dir = os.path.dirname(output_npy)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.save(output_npy, embeddings)

    print("Embeddings saved to:", output_npy)
    print("Embedding shape:", embeddings.shape)

    return embeddings
