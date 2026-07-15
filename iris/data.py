import random

from sklearn.datasets import load_iris

SPECIES_COUNT = 3
TEST_RATIO = 0.2
SPLIT_SEED = 42


def prepare_iris(split='train'):
    """
    Loads and prepares the Iris dataset for use with the NeuralNetwork train function.

    Features are min-max normalized to 0.0-1.0 and labels are one-hot encoded.
    The train/test split is stratified per species (80/20) using a fixed seed,
    so callers requesting 'train' and 'test' always see the same 120/30 split.

    Args:
        split (str): Which split to return. 'train' for training data, 'test' for test data.

    Returns:
        List of tuples (inputs, expected_outputs) where:
            - inputs is a 1D list of 4 normalized feature values (0.0 to 1.0)
            - expected_outputs is a one-hot encoded list of length 3
    """
    iris = load_iris()
    features = iris.data
    labels = iris.target

    mins = features.min(axis=0)
    maxs = features.max(axis=0)
    normalized = (features - mins) / (maxs - mins)

    by_class = {label: [] for label in range(SPECIES_COUNT)}
    for idx, label in enumerate(labels):
        by_class[label].append(idx)

    rng = random.Random(SPLIT_SEED)
    train_indices = []
    test_indices = []
    for label in range(SPECIES_COUNT):
        indices = by_class[label].copy()
        rng.shuffle(indices)
        split_point = int(len(indices) * (1 - TEST_RATIO))
        train_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    chosen = train_indices if split == 'train' else test_indices

    prepared = []
    for idx in chosen:
        inputs = normalized[idx].tolist()
        expected_outputs = [0.0] * SPECIES_COUNT
        expected_outputs[labels[idx]] = 1.0
        prepared.append((inputs, expected_outputs))

    return prepared
