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
    t=0
    e=2

    streak = 0

    correct = 0
    testing = False

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

        network.backpropagate_output_layer(expected_outputs, 0.0007)
        network.backpropagate_hidden_layers(0.0007)


        #print(f"network results: {results}")
        #print(f"expected_outputs: {expected_outputs}")


        if k % 1000 == 0 and testing == True:
            print(f"{k} Iterations - performing evaluation on test set...")
            g = (t) * 100
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
            t+=1

        if k >= len(train_set):
            e+=1
            k=0
            print(f"Epoch {e} completed.")
            network.save(create_file_name(e))

def create_file_name(epoch):
    return f"network_state_epoch_{epoch}.json"

def cross_entropy_loss(softmax_outputs, true_label):
    # Add small epsilon to prevent log(0)
    epsilon = 1e-10
    return -math.log(softmax_outputs[true_label] + epsilon)

main()