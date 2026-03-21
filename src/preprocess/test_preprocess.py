import os
import sys

sys.path.append("src")

from preprocess.remove_extra_spaces import remove_extra_spaces
from preprocess.annotate_and_mask import annotate_and_mask
from preprocess.split_sentences_from_annotation import split_sentences_from_annotation


def print_divider(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def sentence_to_text(sentence):
    text = ""
    for tok, pos, ent in sentence:
        if pos != "SPACE":
            text += tok + " "
    return text.strip()


folder_path = os.path.join("data", "test_plays")

for fname in os.listdir(folder_path):
    if not fname.endswith(".txt"):
        continue

    file_path = os.path.join(folder_path, fname)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    print_divider("FILE: " + fname)

    # --------------------------------------------------
    # 1. Test remove_extra_spaces
    # --------------------------------------------------
    cleaned_text = remove_extra_spaces(raw_text)

    print("\n[1] remove_extra_spaces")
    print("\nOriginal text preview:")
    print(repr(raw_text[:500]))

    print("\nCleaned text preview:")
    print(repr(cleaned_text[:500]))

    # --------------------------------------------------
    # 2. Test annotate_and_mask
    # --------------------------------------------------
    annotation, masked_text = annotate_and_mask(cleaned_text)

    print("\n[2] annotate_and_mask")

    print("\nFirst 30 annotated tokens:")
    for item in annotation[:30]:
        print(item)

    print("\nMasked text preview:")
    print(masked_text[:500])
    
    print("\nTokens that should be masked:")
    found_masked = False

    for tok, pos, ent in annotation:
        if ent in ["PERSON", "GPE", "ORG"]:
            print((tok, pos, ent))
            found_masked = True

    if not found_masked:
        print("No PERSON/GPE/ORG entities found.")

    # --------------------------------------------------
    # 3. Test split_sentences_from_annotation
    # --------------------------------------------------
    sentences = split_sentences_from_annotation(annotation)

    print("\n[3] split_sentences_from_annotation")
    print("\nNumber of sentences:", len(sentences))

    print("\nFirst 5 sentences as text:")
    for i, sentence in enumerate(sentences[:5]):
        print("Sentence", i + 1, ":", sentence_to_text(sentence))

    print("\nDone testing:", fname)