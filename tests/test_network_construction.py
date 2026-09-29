"""
Unit tests for deepnode.NeuralNetwork construction, weight initialization,
and forward()-level edge cases.

Backprop/gradient correctness lives in test_gradient_checking.py; optimizer
behavior lives in test_optimizers.py; train()/save()/load() integration
tests live in their own files. This file covers everything needed to trust
that a NeuralNetwork is built correctly and produces a sane forward pass
before any of those later layers are tested.
"""
import math
import statistics

import pytest

from deepnode import NeuralNetwork


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_invalid_cost_function_raises(self):
        with pytest.raises(ValueError):
            NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2,
                           cost_function='bogus')

    def test_cross_entropy_requires_sigmoid_or_softmax_output(self):
        with pytest.raises(ValueError):
            NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2,
                           output_activation='tanh', cost_function='cross-entropy')

    @pytest.mark.parametrize("output_activation", ['sigmoid', 'softmax'])
    def test_cross_entropy_accepts_valid_output_activations(self, output_activation):
        # Should not raise.
        NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2,
                       output_activation=output_activation, cost_function='cross-entropy')

    @pytest.mark.parametrize(
        "output_activation",
        ['sigmoid', 'relu', 'tanh', 'linear', 'softmax', 'exponential']
    )
    def test_mse_accepts_any_output_activation(self, output_activation):
        # Should not raise -- mse has no activation restriction.
        NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2,
                       output_activation=output_activation, cost_function='mse')


# ---------------------------------------------------------------------------
# Architecture storage
# ---------------------------------------------------------------------------

class TestArchitectureStorage:
    def test_hidden_layer_sizes_stored_as_independent_copy(self):
        sizes = [3, 4]
        net = NeuralNetwork(hidden_layer_sizes=sizes, input_size=2, output_size=2, random_seed=1)
        sizes.append(99)  # mutate the original list after construction
        assert net.hidden_layer_sizes == [3, 4]

    def test_num_hidden_layers_matches_length(self):
        net = NeuralNetwork(hidden_layer_sizes=[3, 4, 5], input_size=2, output_size=2, random_seed=1)
        assert net.num_hidden_layers == 3

    def test_layer_shapes_match_requested_sizes(self):
        net = NeuralNetwork(hidden_layer_sizes=[3, 4], input_size=2, output_size=5, random_seed=1)
        assert [len(layer) for layer in net.hidden_layers] == [3, 4]
        assert len(net.output_layer) == 5
        # Fan-in of each layer should match the previous layer's size.
        assert all(len(n.weights) == 2 for n in net.hidden_layers[0])
        assert all(len(n.weights) == 3 for n in net.hidden_layers[1])
        assert all(len(n.weights) == 4 for n in net.output_layer)


class TestZeroHiddenLayers:
    def test_architecture(self):
        net = NeuralNetwork(hidden_layer_sizes=[], input_size=3, output_size=2, random_seed=1)
        assert net.num_hidden_layers == 0
        assert net.hidden_layers == []
        assert len(net.output_layer) == 2
        # Output layer's fan-in should fall back to input_size directly.
        assert all(len(n.weights) == 3 for n in net.output_layer)

    def test_forward_pass_runs_and_skips_hidden_outputs(self):
        net = NeuralNetwork(hidden_layer_sizes=[], input_size=3, output_size=2,
                             output_activation='linear', random_seed=1)
        output = net.forward([1.0, 2.0, 3.0])
        assert len(output) == 2
        assert net.last_hidden_outputs == []


# ---------------------------------------------------------------------------
# Softmax setup (internal 'linear' rewrite + honest external reporting)
# ---------------------------------------------------------------------------

class TestSoftmaxSetup:
    def test_using_softmax_flag_and_internal_rewrite(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=3,
                             output_activation='softmax', cost_function='cross-entropy',
                             random_seed=1)
        assert net.using_softmax is True
        assert net.output_activation == 'linear'  # internal implementation detail
        assert all(n.activation == 'linear' for n in net.output_layer)

    def test_non_softmax_output_activation_untouched(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2,
                             output_activation='sigmoid', random_seed=1)
        assert net.using_softmax is False
        assert net.output_activation == 'sigmoid'

    def test_get_network_info_reports_softmax_not_linear(self):
        # Regression test: get_network_info() previously leaked the internal
        # 'linear' rewrite instead of reporting what the caller asked for.
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=3,
                             output_activation='softmax', cost_function='cross-entropy',
                             random_seed=1)
        assert net.get_network_info()['output_activation'] == 'softmax'

    def test_get_network_info_reports_non_softmax_normally(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2,
                             output_activation='relu', random_seed=1)
        assert net.get_network_info()['output_activation'] == 'relu'


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_produces_identical_weights(self):
        net1 = NeuralNetwork(hidden_layer_sizes=[5, 4], input_size=3, output_size=2, random_seed=123)
        net2 = NeuralNetwork(hidden_layer_sizes=[5, 4], input_size=3, output_size=2, random_seed=123)
        for layer1, layer2 in zip(net1.hidden_layers, net2.hidden_layers):
            for n1, n2 in zip(layer1, layer2):
                assert n1.weights == n2.weights
                assert n1.bias == n2.bias
        for n1, n2 in zip(net1.output_layer, net2.output_layer):
            assert n1.weights == n2.weights

    def test_different_seeds_produce_different_weights(self):
        net1 = NeuralNetwork(hidden_layer_sizes=[5], input_size=3, output_size=2, random_seed=1)
        net2 = NeuralNetwork(hidden_layer_sizes=[5], input_size=3, output_size=2, random_seed=2)
        assert net1.hidden_layers[0][0].weights != net2.hidden_layers[0][0].weights


# ---------------------------------------------------------------------------
# _weight_init_std — exact formula for each activation group
# ---------------------------------------------------------------------------

class TestWeightInitStd:
    @pytest.mark.parametrize("activation", ['relu', 'leaky_relu'])
    def test_he_group_uses_fan_in_only(self, activation):
        std = NeuralNetwork._weight_init_std(activation, fan_in=10, fan_out=5)
        assert std == pytest.approx(math.sqrt(2 / 10))

    @pytest.mark.parametrize("activation", ['sigmoid', 'tanh', 'linear', 'exponential'])
    def test_glorot_group_uses_fan_in_and_fan_out(self, activation):
        std = NeuralNetwork._weight_init_std(activation, fan_in=10, fan_out=5)
        assert std == pytest.approx(math.sqrt(2 / (10 + 5)))

    def test_unknown_activation_falls_back_to_he_with_warning(self, capsys):
        std = NeuralNetwork._weight_init_std('bogus', fan_in=8, fan_out=4)
        assert std == pytest.approx(math.sqrt(2 / 8))
        captured = capsys.readouterr()
        assert "Warning" in captured.out


class TestWeightInitIntegration:
    """
    Statistical checks that the constructor actually calls _weight_init_std
    with the right activation and fan-in/fan-out for real layers, not just
    that the static method's formula is correct in isolation.
    """

    def test_relu_hidden_layer_weight_std_matches_he_formula(self):
        # A single neuron with a large fan-in gives enough weight samples
        # to estimate its std reasonably well.
        net = NeuralNetwork(hidden_layer_sizes=[1], input_size=200, output_size=1,
                             hidden_activation='relu', random_seed=1)
        weights = net.hidden_layers[0][0].weights
        expected_std = math.sqrt(2 / 200)
        assert statistics.pstdev(weights) == pytest.approx(expected_std, rel=0.3)

    def test_sigmoid_hidden_layer_weight_std_matches_glorot_formula(self):
        net = NeuralNetwork(hidden_layer_sizes=[1], input_size=200, output_size=1,
                             hidden_activation='sigmoid', random_seed=1)
        weights = net.hidden_layers[0][0].weights
        expected_std = math.sqrt(2 / (200 + 1))  # fan_out is this layer's size: 1
        assert statistics.pstdev(weights) == pytest.approx(expected_std, rel=0.3)

    def test_relu_and_sigmoid_produce_meaningfully_different_std(self):
        # fan_out needs to actually diverge from fan_in for He and Glorot to
        # disagree noticeably -- a small fan_in (10) with a large layer size
        # (1000, i.e. fan_out) makes sqrt(2/10) and sqrt(2/1010) an order of
        # magnitude apart, unlike a fan_out of 1 where they're nearly equal.
        net_relu = NeuralNetwork(hidden_layer_sizes=[1000], input_size=10, output_size=1,
                                  hidden_activation='relu', random_seed=7)
        net_sigmoid = NeuralNetwork(hidden_layer_sizes=[1000], input_size=10, output_size=1,
                                     hidden_activation='sigmoid', random_seed=7)
        std_relu = statistics.pstdev(net_relu.hidden_layers[0][0].weights)
        std_sigmoid = statistics.pstdev(net_sigmoid.hidden_layers[0][0].weights)
        assert std_relu != pytest.approx(std_sigmoid, rel=0.2)


# ---------------------------------------------------------------------------
# forward() — input validation
# ---------------------------------------------------------------------------

class TestForwardValidation:
    def test_raises_on_input_length_mismatch(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=4, output_size=2, random_seed=1)
        with pytest.raises(ValueError):
            net.forward([1.0, 2.0])

    def test_matching_length_does_not_raise(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        net.forward([1.0, 2.0])


# ---------------------------------------------------------------------------
# forward() — recorded debug state
# ---------------------------------------------------------------------------

class TestForwardRecording:
    def test_last_inputs_is_independent_copy(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        inputs = [1.0, 2.0]
        net.forward(inputs)
        inputs[0] = 999.0  # mutate after the call
        assert net.last_inputs == [1.0, 2.0]

    def test_last_outputs_matches_return_value(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        output = net.forward([1.0, 2.0])
        assert net.last_outputs == output

    def test_last_hidden_outputs_shape_matches_layer_sizes(self):
        net = NeuralNetwork(hidden_layer_sizes=[3, 5], input_size=2, output_size=2, random_seed=1)
        net.forward([1.0, 2.0])
        assert len(net.last_hidden_outputs) == 2
        assert len(net.last_hidden_outputs[0]) == 3
        assert len(net.last_hidden_outputs[1]) == 5


# ---------------------------------------------------------------------------
# forward() — softmax normalization and numerical stability
# ---------------------------------------------------------------------------

class TestSoftmaxForward:
    def test_outputs_sum_to_one_and_bounded(self):
        net = NeuralNetwork(hidden_layer_sizes=[4], input_size=3, output_size=5,
                             output_activation='softmax', cost_function='cross-entropy',
                             random_seed=1)
        output = net.forward([0.5, -0.2, 0.9])
        assert sum(output) == pytest.approx(1.0)
        assert all(0.0 <= o <= 1.0 for o in output)

    def test_numerically_stable_with_extreme_logits(self):
        net = NeuralNetwork(hidden_layer_sizes=[2], input_size=2, output_size=3,
                             output_activation='softmax', cost_function='cross-entropy',
                             random_seed=1)
        # Force extreme pre-softmax logits directly, independent of input,
        # by zeroing the weights and setting extreme biases.
        net.output_layer[0].set_weights([0.0, 0.0])
        net.output_layer[0].set_bias(1000.0)
        net.output_layer[1].set_weights([0.0, 0.0])
        net.output_layer[1].set_bias(-1000.0)
        net.output_layer[2].set_weights([0.0, 0.0])
        net.output_layer[2].set_bias(0.0)

        output = net.forward([0.0, 0.0])

        assert not any(math.isnan(o) for o in output)
        assert sum(output) == pytest.approx(1.0)
        assert output[0] == pytest.approx(1.0, abs=1e-6)   # dominant logit
        assert output[1] == pytest.approx(0.0, abs=1e-6)   # crushed logit


# ---------------------------------------------------------------------------
# forward() — dropout is scoped to hidden layers only
# ---------------------------------------------------------------------------

class TestDropoutScope:
    def test_dropout_never_applies_to_output_layer(self):
        net = NeuralNetwork(hidden_layer_sizes=[10], input_size=3, output_size=2, random_seed=1)
        net.forward([0.5, 0.5, 0.5], dropout_rate=1.0)  # drop every hidden neuron
        assert all(n.last_dropout_mask == 1.0 for n in net.output_layer)

    def test_dropout_rate_zero_leaves_hidden_masks_at_one(self):
        net = NeuralNetwork(hidden_layer_sizes=[10], input_size=3, output_size=2, random_seed=1)
        net.forward([0.5, 0.5, 0.5], dropout_rate=0.0)
        assert all(n.last_dropout_mask == 1.0 for n in net.hidden_layers[0])
