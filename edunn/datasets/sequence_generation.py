import numpy as np
from . import basepath

classification_basepath = basepath / "sequence_data"


def load_sequence_dataset(filename, vocabulary_size):
    unk_token = "<UNK>"
    pad_token = "<PAD>"
    sos_token = "<SOS>"
    eos_token = "<EOS>"

    raw = np.genfromtxt(classification_basepath / filename, dtype=str, delimiter="\n", encoding="utf-8")
    raw_lower = np.char.lower(raw)

    # basic segmentation
    temp = np.char.replace(raw_lower, "!", ".")
    temp = np.char.replace(temp, "?", ".")
    all_text = " ".join(temp.tolist())
    parts = [s.strip() for s in all_text.split(".") if s.strip()]

    # add special tokens
    sentences = np.array([f"{sos_token} {s} {eos_token}" for s in parts], dtype=object)

    # tokenization
    tokenized = np.array([s.split() for s in sentences], dtype=object)
    tokenized = np.array([t for t in tokenized if len(t) > 3], dtype=object)

    # vocabulary
    all_tokens = np.concatenate(tokenized)
    words, counts = np.unique(all_tokens, return_counts=True)
    idx_sorted = np.argsort(-counts)
    words = words[idx_sorted]
    counts = counts[idx_sorted]

    # filter vocabulary size (-2 for unk_token and pad_token)
    vocab = words[: vocabulary_size - 2]
    index_to_word = np.concatenate([[unk_token], [pad_token], vocab])
    word_to_index = {w: i for i, w in enumerate(index_to_word)}

    # set unk for out of vocabulary words
    mapper = np.vectorize(lambda w: w if w in word_to_index else unk_token)
    tokenized = np.array([mapper(sent) for sent in tokenized], dtype=object)

    # X_train, y_train (arrays of lists)
    X_train = np.array([[word_to_index[w] for w in sent[:-1]] for sent in tokenized], dtype=object)
    y_train = np.array([[word_to_index[w] for w in sent[1:]] for sent in tokenized], dtype=object)

    return X_train, y_train, word_to_index, index_to_word


reddit_comments = lambda: load_sequence_dataset("reddit_comments.csv", vocabulary_size=8000)

loaders = {
    "reddit_comments": reddit_comments,
}
