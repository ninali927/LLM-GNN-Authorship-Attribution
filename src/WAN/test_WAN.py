import os
import sys
import pandas as pd

sys.path.append("src")

from build_dataset import build_dataset
from preprocess.preprocess_pipeline import preprocess_chunk_text
from WAN.function_words import get_function_word_to_idx, FUNCTION_WORDS
from WAN.wan_matrix import build_wan_from_sentences
from WAN.markov_normalization import markov_normalization, compute_stationary_distribution
from WAN.wan_distance import compute_chunk_distance


def print_divider(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# --------------------------------------------------
# 0. Build dataset (only runs if CSV not present)
# --------------------------------------------------
input_folder = "data/test_plays"
csv_path = "data/chunked_plays.csv"

if not os.path.exists(csv_path):
    build_dataset(input_folder, csv_path)


# --------------------------------------------------
# 1. Read chunked dataset
# --------------------------------------------------
df = pd.read_csv(csv_path)

fw_to_idx = get_function_word_to_idx()
all_chunks = []

for i in range(len(df)):
    chunk_id = df.loc[i, "chunk_id"]
    raw_text = df.loc[i, "text"]

    print_divider("CHUNK: " + chunk_id)

    cleaned_text, annotation, masked_text, sentences = preprocess_chunk_text(raw_text)
    all_chunks.append((chunk_id, cleaned_text))

    # --------------------------------------------------
    # 2. WAN matrix
    # --------------------------------------------------
    A = build_wan_from_sentences(sentences)

    F = sorted(list(FUNCTION_WORDS))
    A_df = pd.DataFrame(A, index=F, columns=F)

    print("\n[WAN matrix]")
    print(A_df)

    # --------------------------------------------------
    # 3. Markov normalization
    # --------------------------------------------------
    P = markov_normalization(A)

    P_df = pd.DataFrame(P, index=F, columns=F)

    print("\n[Markov matrix]")
    print(P_df)

    # --------------------------------------------------
    # 4. Stationary distribution
    # --------------------------------------------------
    pi = compute_stationary_distribution(P)

    pi_series = pd.Series(pi, index=F)

    print("\n[Stationary distribution]")
    print(pi_series)


# --------------------------------------------------
# 5. WAN distance between first two chunks
# --------------------------------------------------
print_divider("WAN DISTANCE TEST")

chunk_id_1, chunk_text_1 = all_chunks[0]
chunk_id_2, chunk_text_2 = all_chunks[1]

distance = compute_chunk_distance(
    chunk_text_1,
    chunk_text_2,
    function_words=FUNCTION_WORDS,
    D=10,
    alpha=0.75,
    epsilon=1e-12,
    distance_type="kl"
)

print("Chunk 1:", chunk_id_1)
print("Chunk 2:", chunk_id_2)
print("Distance:", distance)