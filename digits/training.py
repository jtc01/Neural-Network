import random

from network import NeuralNetwork
import math

# EPOCHS 1 and 2 work, 3, 4, and 5 do not. Need to retrain using leaky relu.

def main():

    train_data = prepare_mnist(split='train')   # 60,000 samples
    test_data = prepare_mnist(split='test')     # 10,000 samples

    network = NeuralNetwork(
        hidden_layers=[512, 256, 128],
        input_size=784,
        output_size=10,
        hidden_activation='leaky_relu',
        output_activation='softmax',
        random_seed=67,
        cost_function='cross-entropy'
    )

    network.load("network_state_epoch_2.json")

    k = 0
    t=0
    e=2

    lr=0.001

    streak = 0

    correct = 0
    testing = False

    while True:

        network.train_adam(train_data, epochs=1, initial_learning_rate=lr, print_rate=1000, weight_clip_value=5.0, bias_clip_value=10.0, momentum=0.9, squared_gradient_term=0.999, batch_size=32, dropout_rate=0.2)
        network.save(create_file_name(e))


def create_file_name(epoch):
    return f"network_state_epoch_{epoch}.json"

def cross_entropy_loss(softmax_outputs, true_label):
    # Add small epsilon to prevent log(0)
    epsilon = 1e-10
    return -math.log(softmax_outputs[true_label] + epsilon)

def prepare_mnist(split='train'):
    """
    Loads and prepares the MNIST dataset for use with the NeuralNetwork train function.

    Args:
        split (str): Which split of the dataset to load. 'train' for training data, 'test' for test data.

    Returns:
        List of tuples (inputs, expected_outputs) where:
            - inputs is a 1D list of 784 normalized pixel values (0.0 to 1.0)
            - expected_outputs is a one-hot encoded list of length 10
    """
    from datasets import load_dataset

    dataset = load_dataset('mnist', split=split)

    prepared = []
    for sample in dataset:
        # Flatten 28x28 image to 1D list of 784 pixels and normalize to 0-1
        inputs = [pixel / 255.0 for pixel in sample['image'].getdata()]

        # One-hot encode the label
        expected_outputs = [0.0] * 10
        expected_outputs[sample['label']] = 1.0

        prepared.append((inputs, expected_outputs))

    return prepared

main()