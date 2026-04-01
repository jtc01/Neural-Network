from network import NeuralNetwork
from testing.data import DataPoint
import random

def generate_data_two_inputs(numSamples):
    data=[]
    for i in range(numSamples):
        x = random.uniform(-5,5)
        y = random.uniform(-5,5)
        dataPoint = DataPoint(x,y)
        data.append(dataPoint)
    return data


def main():

    network = NeuralNetwork(
        hidden_layers=[4],
        input_size=2,
        output_size=2,
        hidden_activation='sigmoid',
        output_activation='sigmoid',
        random_seed=42
    )

    counter=0
    total_forwards=0

    while True:
        inputs=[]
        data_points=[]
        expected_outputs=[]

        data_points=generate_data_two_inputs(50).copy()

        for point in data_points:
            input=[point.x, point.y]
            inputs.append(input)
            expected_output=[point.t, point.f]
            expected_outputs.append(expected_output)
        incorrect=0
        for i in range(len(data_points)):
            output=network.forward(inputs[i])
            if (expected_outputs[i][0]>expected_outputs[i][1] and output[0]<output[1]) or (expected_outputs[i][0]<expected_outputs[i][1] and output[0]>output[1]):
                incorrect+=1
            network.backpropagate_output_layer(expected_outputs[i], 0.05)
            network.backpropagate_hidden_layers(0.05)
        if incorrect==0:
            network.print_network_structure
            break
        else:
            counter+=1
            total_forwards+=50
            print(f"{counter} | {total_forwards} | {incorrect}")
    

main()