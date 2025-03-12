# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike


def sample_seqs(seqs: List[str], labels: List[bool], seed: int) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    seqs, labels = np.array(seqs), np.array(labels)
    labels = np.array(labels, dtype=bool)

    # Find indices where labels are True (positive) and False (negative)
    pos_indices = np.where(labels == True)[0]
    neg_indices = np.where(labels == False)[0]

    # Ensure both classes are present in dataset
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        raise ValueError("Dataset must contain both classes.")

    # Determine maximum class size
    max_size = max(len(pos_indices), len(neg_indices))

    # Resampling
    pos_sampled = np.random.choice(pos_indices, max_size, replace=True)
    neg_sampled = np.random.choice(neg_indices, max_size, replace=True)

    final_indices = np.concatenate([pos_sampled, neg_sampled])
    np.random.shuffle(final_indices)

    # Return sampled sequences and labels
    return list(seqs[final_indices]), list(labels[final_indices])


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # Dictionary for encoding
    encode_dict = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1]
    }

    # Create a list of one-hot encodings directly
    output_encodings = []

    # Loop through the sequences
    for seq in seq_arr:
        # Using a list comprehension to map each nucleotide to its encoding
        encoded_seq = [encode_dict[nucleotide] for nucleotide in seq]

        # Convert the list of lists into a 1D array and append to the output list
        output_encodings.append(np.array(encoded_seq).flatten())

    # Convert the list of numpy arrays to a single numpy array
    return np.array(output_encodings)
