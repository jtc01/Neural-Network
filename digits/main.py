from network import NeuralNetwork
from datasets import load_dataset
import math

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
        
        #print(inputs)

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
            print(f"{k} | {label} | {result}")


        #print(f"network results: {results}")
        #print(f"expected_outputs: {expected_outputs}")

        if k % 1000 == 0:
            print(f"{k} Iterations - performing evaluation on test set...")
            g = (e) * 100
            correct = 0
            total_loss = 0
            for i in range(100):
                numeral = test_set[g + i]
                label = int(numeral["label"])

                image = numeral["image"]
                inputs = []
                for i in range(28):
                    for j in range(28):
                        pixel = image.getpixel((j, i))/255
                        inputs.append(pixel)

                output = network.forward(inputs)
                result = output.index(max(output))
                if label == result:
                    correct +=1
                loss = cross_entropy_loss(output, label)
                total_loss += loss
            print(f"{k} Iterations - Accuracy: {correct/100:.2%}, Average Loss: {total_loss/100:.4f}")
            e+=1

        network.backpropagate_output_layer(expected_outputs, 0.001)
        network.backpropagate_hidden_layers(0.001)
    network.print_network_structure()

def cross_entropy_loss(softmax_outputs, true_label):
    # Add small epsilon to prevent log(0)
    epsilon = 1e-10
    return -math.log(softmax_outputs[true_label] + epsilon)



main()
#24579