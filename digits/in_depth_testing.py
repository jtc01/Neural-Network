from network import NeuralNetwork
from datasets import load_dataset
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

    k = 0
    while True:
        inputs = []
        
        numeral = train_set[k]
        label = numeral["label"]

        image = numeral["image"]
        for i in range(28):
            for j in range(28):
                pixel = image.getpixel((j, i))/255
                inputs.append(pixel)
        
        index = int(label)
        expected_outputs = [0.0]*10
        expected_outputs[index] = 1

        output = network.forward(inputs)
        result = output.index(max(output))

        if k % 1000 == 0:
            print(f"Test {k}: Label={label}, Predicted={result}")
            for i in range(10):
                if (int(label) == i):
                    print(f"  Class {i}: {output[i]:.2f} <--")
                else:
                    print(f"  Class {i}: {output[i]:.2f}")
            loss = cross_entropy_loss(output, index)
            print(f"  Loss: {loss:.4f}\n")
            k+=1
            
        if k >= len(train_set):
            print("Testing completed.")
            break

def cross_entropy_loss(softmax_outputs, true_label):
    # Add small epsilon to prevent log(0)
    epsilon = 1e-10
    return -math.log(softmax_outputs[true_label] + epsilon)

main()