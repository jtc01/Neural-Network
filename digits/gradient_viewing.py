from datasets import load_dataset
from network import NeuralNetwork
import math

def main():

    dataset = load_dataset("ylecun/mnist")
    train_set = dataset["train"]
    test_set = dataset["test"]

    network = NeuralNetwork(
        hidden_layers=[512, 256, 128],
        input_size=784,
        output_size=10,
        hidden_activation='relu',
        output_activation='softmax',
        random_seed=67,
        cost_function='cross-entropy'
    )

    network.load("network_state_epoch_2.json")

    dataset = load_dataset("ylecun/mnist")
    test_set = dataset["test"]
    digit = test_set[6]
    image = digit["image"]
    label = digit["label"]
    inputs = []

    for i in range(28):
        for j in range(28):
            pixel = image.getpixel((j, i))/255
            inputs.append(pixel)
    
    output = network.forward(inputs)
    result = output.index(max(output))
    gradients = []
    for layer in network.hidden_layers:
        for neuron in layer:
            for idx, weight in enumerate(neuron.weights):
                gradients.append(weight * neuron.last_inputs[idx])
    
    for neuron in network.output_layer:
        for idx, weight in enumerate(neuron.weights):
            gradients.append(weight * neuron.last_inputs[idx])
    
    gradients.sort(key=abs, reverse=False)
    for gradient in gradients:
        print(gradient)

main()