import random
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
        hidden_activation='leaky_relu',
        output_activation='softmax',
        random_seed=67,
        cost_function='cross-entropy'
    )

    network.load("network_state_epoch_1.json")

    k = 0
    batch_correct = 0
    batch_loss = 0
    total_correct = 0
    total_loss = 0

    while True:
        

        inputs = []
        
        numeral = test_set[k]
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
        if index == result:
            batch_correct += 1
            total_correct += 1

        loss = cross_entropy_loss(output, index)
        batch_loss += loss
        total_loss += loss

        if k % 100 == 0:
            print(f"{k} | {label} | {result}")

        if k % 1000 == 0:
            if k > 0:
                print(f"Accuracy from sample {k-1000} to {k}: {batch_correct/1000:.2%}")
                print(f"Average Cross-Entropy Loss from sample {k-1000} to {k}: {batch_loss/1000:.4f}")
            batch_correct = 0
            batch_loss = 0
            print(f"Test {k}: Label={label}, Predicted={result}")
            for i in range(10):
                if (int(label) == i):
                    print(f"  Class {i}: {output[i]:.2f} <--")
                else:
                    print(f"  Class {i}: {output[i]:.2f}")
            loss = cross_entropy_loss(output, index)
            print(f"  Loss: {loss:.4f}\n")
        k+=1
            
        if k >= len(test_set):
            print("Testing completed.")
            print(f"Total Accuracy: {total_correct/len(test_set):.2%}")
            print(f"Average Cross-Entropy Loss: {total_loss/len(test_set):.4f}")
            break

def cross_entropy_loss(softmax_outputs, true_label):
    # Add small epsilon to prevent log(0)
    epsilon = 1e-10
    return -math.log(softmax_outputs[true_label] + epsilon)

main()