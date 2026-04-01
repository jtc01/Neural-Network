from network import NeuralNetwork
from dataset import load_dataset
import random

class DataPoint:
    def __init__(self, input=None, output=0):
        if self.input is None:
            raise ValueError("Input cannot be None.")
        elif len(input) != 784:
            raise ValueError("Input must be a list of length 784.")
        else:
            self.input = input
        
        if output < 0 or output > 9:
            raise ValueError("Output must be an integer between 0 and 9.")
        else:
            self.output = output


def main():

    dataset = load_dataset("ylecun/mnist")
    train_set = dataset["train"]

    network = NeuralNetwork(
        hidden_layers=[128, 64],
        input_size=784,
        output_size=10,
        hidden_activation='relu',
        output_activation='softmax',
        random_seed=67
    )

    i = random.randint(0, len(train_set)-1)

    streak = 0

    while True:
        inputs = []
        
        numeral = train_set[i]
        label = numeral["label"]

        image = numeral["image"]
        for i in range(28):
            for j in range(28):
                pixel = image.getpixel((j, i))/255
                inputs.append(pixel)
        
        expected_outputs = [0]*10
        expected_outputs[label] = 1

        output = network.forward(inputs)
        result = output.index(max(output))
        print(f"{label} | {result}")

        if label == result:
            streak += 1
        else:
            streak = 0
        
        if streak >= 10:
            network.print_network_structure()
            break

        network.backpropagate_output_layer(expected_outputs, 0.01)
        network.backpropagate_hidden_layers(0.01)


main()