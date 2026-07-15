import math

from network import NeuralNetwork
from iris.data import prepare_iris

SPECIES_NAMES = ['setosa', 'versicolor', 'virginica']


def main():

    test_data = prepare_iris(split='test')  # 30 samples

    network = NeuralNetwork(
        hidden_layers=[],
        input_size=4,
        output_size=3,
        hidden_activation='relu',
        output_activation='softmax',
        random_seed=42,
        cost_function='cross-entropy'
    )

    network.load("iris_state.json")

    total_correct = 0
    total_loss = 0

    for k, (inputs, expected_outputs) in enumerate(test_data):
        label = expected_outputs.index(1.0)

        output = network.forward(inputs)
        result = output.index(max(output))

        if label == result:
            total_correct += 1

        loss = cross_entropy_loss(output, label)
        total_loss += loss

        print(f"Test {k}: Label={SPECIES_NAMES[label]}, Predicted={SPECIES_NAMES[result]}")
        for i in range(3):
            if label == i:
                print(f"  {SPECIES_NAMES[i]}: {output[i]:.2f} <--")
            else:
                print(f"  {SPECIES_NAMES[i]}: {output[i]:.2f}")
        print(f"  Loss: {loss:.4f}\n")

    print("Testing completed.")
    print(f"Total Accuracy: {total_correct}/{len(test_data)} ({total_correct / len(test_data):.2%})")
    print(f"Average Cross-Entropy Loss: {total_loss / len(test_data):.4f}")


def cross_entropy_loss(softmax_outputs, true_label):
    # Add small epsilon to prevent log(0)
    epsilon = 1e-10
    return -math.log(softmax_outputs[true_label] + epsilon)


main()
