"""
Numerical gradient checking for gradwave.NeuralNetwork.

This is the most important test file in the suite: it verifies that
compute_output_node_values() / compute_hidden_node_values() actually compute
the correct gradient of the network's own cost function, by comparing the
analytic node_value-derived gradient against a central-difference numerical
approximation. If this file passes, every optimizer built on top of these
gradients (which is all of them) is at least consuming a mathematically
correct signal, whatever else might be wrong with the optimizer itself.
"""
import math

import pytest

from gradwave import NeuralNetwork


EPSILON = 1e-5
REL_TOLERANCE = 1e-3
ABS_TOLERANCE = 1e-6


def _loss(network, outputs, expected):
    """
    Compute the scalar loss exactly matching the gradient formula used
    inside compute_output_node_values(), so the numerical and analytic
    gradients being compared are checking the same quantity.
    """
    if network.cost_function == 'mse':
        return 0.5 * sum((o - e) ** 2 for o, e in zip(outputs, expected))
    else:  # cross-entropy
        total = 0.0
        for o, e in zip(outputs, expected):
            o_clamped = min(max(o, 1e-12), 1 - 1e-12)  # avoid log(0)
            if network.using_softmax:
                total += -e * math.log(o_clamped)
            else:
                total += -(e * math.log(o_clamped) + (1 - e) * math.log(1 - o_clamped))
        return total


def _get_neuron(network, layer, layer_idx, neuron_idx):
    if layer == 'hidden':
        return network.hidden_layers[layer_idx][neuron_idx]
    return network.output_layer[neuron_idx]


def _numeric_weight_gradient(network, inputs, expected, layer, layer_idx, neuron_idx, weight_idx):
    neuron = _get_neuron(network, layer, layer_idx, neuron_idx)
    original = neuron.weights[weight_idx]

    neuron.weights[weight_idx] = original + EPSILON
    loss_plus = _loss(network, network.forward(inputs), expected)

    neuron.weights[weight_idx] = original - EPSILON
    loss_minus = _loss(network, network.forward(inputs), expected)

    neuron.weights[weight_idx] = original  # restore

    return (loss_plus - loss_minus) / (2 * EPSILON)


def _numeric_bias_gradient(network, inputs, expected, layer, layer_idx, neuron_idx):
    neuron = _get_neuron(network, layer, layer_idx, neuron_idx)
    original = neuron.bias

    neuron.bias = original + EPSILON
    loss_plus = _loss(network, network.forward(inputs), expected)

    neuron.bias = original - EPSILON
    loss_minus = _loss(network, network.forward(inputs), expected)

    neuron.bias = original  # restore

    return (loss_plus - loss_minus) / (2 * EPSILON)


def _run_backward(network, inputs, expected):
    """One real forward + backward pass, leaving analytic node_values in place."""
    network.forward(inputs)
    network.compute_output_node_values(expected)
    network.compute_hidden_node_values()


# Every (hidden_activation, output_activation, cost_function) combination
# worth checking. mse is activation-agnostic; cross-entropy is restricted
# to sigmoid/softmax output activations by the constructor.
CONFIGS = [
    pytest.param('relu', 'sigmoid', 'mse', id='relu-sigmoid-mse'),
    pytest.param('tanh', 'linear', 'mse', id='tanh-linear-mse'),
    pytest.param('leaky_relu', 'linear', 'mse', id='leaky_relu-linear-mse'),
    pytest.param('sigmoid', 'tanh', 'mse', id='sigmoid-tanh-mse'),
    pytest.param('sigmoid', 'sigmoid', 'cross-entropy', id='sigmoid-sigmoid-crossent'),
    pytest.param('relu', 'softmax', 'cross-entropy', id='relu-softmax-crossent'),
    pytest.param('tanh', 'softmax', 'cross-entropy', id='tanh-softmax-crossent'),
]

INPUTS = [0.3, -0.6, 0.9]


def _expected_for(cost_function, one_hot_index=0, size=3):
    if cost_function == 'cross-entropy':
        vec = [0.0] * size
        vec[one_hot_index] = 1.0
        return vec
    # Arbitrary fixed non-trivial mse targets.
    base = [0.2, -0.4, 0.6, -0.1, 0.5]
    return base[:size]


@pytest.mark.parametrize("hidden_activation, output_activation, cost_function", CONFIGS)
def test_output_layer_weight_gradients(hidden_activation, output_activation, cost_function):
    net = NeuralNetwork(
        hidden_layer_sizes=[4], input_size=3, output_size=3,
        hidden_activation=hidden_activation, output_activation=output_activation,
        cost_function=cost_function, random_seed=1
    )
    expected = _expected_for(cost_function, one_hot_index=0, size=3)

    _run_backward(net, INPUTS, expected)

    for neuron_idx in range(net.output_size):
        neuron = net.output_layer[neuron_idx]
        for weight_idx in range(len(neuron.weights)):
            analytic = neuron.node_value * neuron.last_inputs[weight_idx]
            numeric = _numeric_weight_gradient(net, INPUTS, expected, 'output', None, neuron_idx, weight_idx)
            assert numeric == pytest.approx(analytic, rel=REL_TOLERANCE, abs=ABS_TOLERANCE)


@pytest.mark.parametrize("hidden_activation, output_activation, cost_function", CONFIGS)
def test_output_layer_bias_gradients(hidden_activation, output_activation, cost_function):
    net = NeuralNetwork(
        hidden_layer_sizes=[4], input_size=3, output_size=3,
        hidden_activation=hidden_activation, output_activation=output_activation,
        cost_function=cost_function, random_seed=2
    )
    expected = _expected_for(cost_function, one_hot_index=1, size=3)

    _run_backward(net, INPUTS, expected)

    for neuron_idx in range(net.output_size):
        neuron = net.output_layer[neuron_idx]
        analytic = neuron.node_value
        numeric = _numeric_bias_gradient(net, INPUTS, expected, 'output', None, neuron_idx)
        assert numeric == pytest.approx(analytic, rel=REL_TOLERANCE, abs=ABS_TOLERANCE)


@pytest.mark.parametrize("hidden_activation, output_activation, cost_function", CONFIGS)
def test_hidden_layer_weight_gradients(hidden_activation, output_activation, cost_function):
    net = NeuralNetwork(
        hidden_layer_sizes=[4], input_size=3, output_size=3,
        hidden_activation=hidden_activation, output_activation=output_activation,
        cost_function=cost_function, random_seed=3
    )
    expected = _expected_for(cost_function, one_hot_index=2, size=3)

    _run_backward(net, INPUTS, expected)

    for neuron_idx in range(4):
        neuron = net.hidden_layers[0][neuron_idx]
        for weight_idx in range(len(neuron.weights)):
            analytic = neuron.node_value * neuron.last_inputs[weight_idx]
            numeric = _numeric_weight_gradient(net, INPUTS, expected, 'hidden', 0, neuron_idx, weight_idx)
            assert numeric == pytest.approx(analytic, rel=REL_TOLERANCE, abs=ABS_TOLERANCE)


@pytest.mark.parametrize("hidden_activation, output_activation, cost_function", CONFIGS)
def test_hidden_layer_bias_gradients(hidden_activation, output_activation, cost_function):
    net = NeuralNetwork(
        hidden_layer_sizes=[4], input_size=3, output_size=3,
        hidden_activation=hidden_activation, output_activation=output_activation,
        cost_function=cost_function, random_seed=4
    )
    expected = _expected_for(cost_function, one_hot_index=0, size=3)

    _run_backward(net, INPUTS, expected)

    for neuron_idx in range(4):
        neuron = net.hidden_layers[0][neuron_idx]
        analytic = neuron.node_value
        numeric = _numeric_bias_gradient(net, INPUTS, expected, 'hidden', 0, neuron_idx)
        assert numeric == pytest.approx(analytic, rel=REL_TOLERANCE, abs=ABS_TOLERANCE)


def test_two_hidden_layers_gradient_chain_is_correct():
    """
    A deeper check specifically for the multi-hidden-layer backprop chain --
    compute_hidden_node_values() summing weight_to_next * next_node_value
    across every neuron in the *next* layer. The single-hidden-layer tests
    above never exercise a hidden-to-hidden connection at all, only
    hidden-to-output, so this is a distinct and necessary check.
    """
    net = NeuralNetwork(
        hidden_layer_sizes=[4, 3], input_size=3, output_size=2,
        hidden_activation='tanh', output_activation='sigmoid',
        cost_function='mse', random_seed=5
    )
    expected = [0.6, 0.1]

    _run_backward(net, INPUTS, expected)

    # Check every weight in the FIRST hidden layer specifically -- its
    # gradient depends on the chain running through the second hidden layer.
    for neuron_idx in range(4):
        neuron = net.hidden_layers[0][neuron_idx]
        for weight_idx in range(len(neuron.weights)):
            analytic = neuron.node_value * neuron.last_inputs[weight_idx]
            numeric = _numeric_weight_gradient(net, INPUTS, expected, 'hidden', 0, neuron_idx, weight_idx)
            assert numeric == pytest.approx(analytic, rel=REL_TOLERANCE, abs=ABS_TOLERANCE)


def test_dropped_neuron_gets_zero_node_value_regardless_of_downstream_weights():
    """
    Regression test for the dropout/backprop consistency fix:
    compute_hidden_node_values() must zero out node_value for any neuron
    whose last_dropout_mask is 0.0, regardless of what the chain-rule sum
    over the next layer's weights and node values would otherwise produce.
    """
    net = NeuralNetwork(
        hidden_layer_sizes=[4], input_size=3, output_size=2,
        hidden_activation='relu', output_activation='sigmoid',
        cost_function='mse', random_seed=6
    )
    net.forward(INPUTS)
    net.compute_output_node_values([0.5, 0.5])

    # Force this neuron to look like it was dropped on the forward pass.
    net.hidden_layers[0][0].last_dropout_mask = 0.0

    net.compute_hidden_node_values()

    assert net.hidden_layers[0][0].node_value == 0.0
