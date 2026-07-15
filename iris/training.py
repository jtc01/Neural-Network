import random
import math

from network import NeuralNetwork
from iris.data import prepare_iris

MAX_EPOCHS = 300
TARGET_CORRECT = 28


def main():

    train_data = prepare_iris(split='train')  # 120 samples
    test_data = prepare_iris(split='test')    # 30 samples

    network = NeuralNetwork(
        hidden_layers=[],
        input_size=4,
        output_size=3,
        hidden_activation='relu',
        output_activation='softmax',
        random_seed=42,
        cost_function='cross-entropy'
    )

    for e in range(MAX_EPOCHS):
        random.seed(e)
        random.shuffle(train_data)
        network.train(train_data, epochs=1, optimizer='adam', initial_learning_rate=0.05, learning_rate_decay=1.0, batch_size=8, weight_clip_value=5.0, bias_clip_value=10.0, momentum=0.9, squared_gradient_term=0.999, print_rate=0)

        correct, total_loss = evaluate(network, test_data)
        print(f"Epoch {e} - Test Accuracy: {correct}/{len(test_data)}, Avg Loss: {total_loss / len(test_data):.4f}")

        if correct >= TARGET_CORRECT:
            print(f"Reached target of {TARGET_CORRECT}/{len(test_data)} after {e + 1} epoch(s).")
            break

    network.save("iris_state.json")


def evaluate(network, data):
    correct = 0
    total_loss = 0.0
    for inputs, expected_outputs in data:
        output = network.forward(inputs)
        result = output.index(max(output))
        label = expected_outputs.index(1.0)

        if result == label:
            correct += 1

        total_loss += cross_entropy_loss(output, label)

    return correct, total_loss


def cross_entropy_loss(softmax_outputs, true_label):
    # Add small epsilon to prevent log(0)
    epsilon = 1e-10
    return -math.log(softmax_outputs[true_label] + epsilon)


main()
