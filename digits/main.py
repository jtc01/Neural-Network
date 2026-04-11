from network import NeuralNetwork
from network import Neuron
from datasets import load_dataset
import random

class DataPoint:
    def __init__(self, input=None, output=0):
        if input is None:
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
        hidden_layers=[512, 256, 128],
        input_size=784,
        output_size=10,
        hidden_activation='relu',
        output_activation='softmax',
        random_seed=67
    )

    k = 0
    e=0

    streak = 0

    correct = 0

    while True:
        inputs = []
        
        numeral = train_set[k]
        label = numeral["label"]

        image = numeral["image"]
        for i in range(28):
            for j in range(28):
                pixel = image.getpixel((j, i))/255
                inputs.append(pixel)
        
        print(inputs)

        index = int(label)
        expected_outputs = [0.0]*10
        expected_outputs[index] = 1

        output = network.forward(inputs)
        result = output.index(max(output))

        if label == result:
            streak += 1
            correct += 1

        else:
            streak = 0

        k+=1

        if k%100 == 0:
            break

        print(f"{k} | {label} | {result}")
        results = []
        for neuron in network.output_layer:
            results.append(neuron.last_output)
        print(f"network results: {results}")
        print(f"expected_outputs: {expected_outputs}")

        if k >= len(train_set):
            k = 0
            e+=1
            print(f"{e} epochs completed.")
            print(f"Epoch accuracy: {correct/(len(train_set)):.2%}")
            correct = 0

        network.backpropagate_output_layer(expected_outputs, 0.00001)
        network.backpropagate_hidden_layers(0.00001)
    network.print_network_structure()


main()
#24579