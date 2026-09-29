"""
Tests for every optimizer in deepnode.NeuralNetwork: SGD, Adam, AdaGrad, and
RMSprop, in both their single-sample (update_weights_*) and batched
(apply_gradient_accumulations_*) forms, plus the shared machinery around
them (_apply_optimizer_update dispatch, accumulate_gradients,
reset_accumulated_gradients).

These tests bypass forward()/backward() entirely and set node_value,
last_inputs, and the gradient accumulators directly on hand-picked neurons.
That's intentional: it isolates "does the optimizer math do what it claims"
from "is the gradient it's given correct" (already covered by
test_gradient_checking.py), and it makes every expected value exactly
hand-computable.
"""
import math

import pytest

from deepnode import NeuralNetwork


def _single_output_neuron_network(weights, bias=0.0, input_size=None):
    """
    A network with zero hidden layers and one output neuron, so the neuron
    under test is reachable directly at net.output_layer[0] with fully
    controlled starting weights/bias and zeroed optimizer state.
    """
    if input_size is None:
        input_size = len(weights)
    net = NeuralNetwork(hidden_layer_sizes=[], input_size=input_size, output_size=1, random_seed=1)
    neuron = net.output_layer[0]
    neuron.weights = list(weights)
    neuron.bias = bias
    return net, neuron


def _primed(neuron, last_inputs, node_value):
    """Set a neuron up as if a forward+backward pass had just run."""
    neuron.last_inputs = list(last_inputs)
    neuron.node_value = node_value
    return neuron


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------

class TestSGDSingleSample:
    def test_zero_momentum_matches_plain_gradient_descent(self):
        net, neuron = _single_output_neuron_network(weights=[1.0, 2.0], bias=0.5)
        _primed(neuron, last_inputs=[3.0, 4.0], node_value=0.1)

        net.update_weights_sgd(learning_rate=0.1, weight_clip_value=None,
                                bias_clip_value=None, momentum=0.0)

        # weight_gradient_i = node_value * input_i; velocity = -lr * gradient (momentum=0)
        assert neuron.weights[0] == pytest.approx(1.0 - 0.1 * (0.1 * 3.0))
        assert neuron.weights[1] == pytest.approx(2.0 - 0.1 * (0.1 * 4.0))
        assert neuron.bias == pytest.approx(0.5 - 0.1 * 0.1)

    def test_momentum_accumulates_across_successive_calls(self):
        net, neuron = _single_output_neuron_network(weights=[1.0], bias=0.0)
        lr, momentum = 0.1, 0.9
        gradient_input = 2.0
        node_value = 0.5  # weight_gradient = 0.5 * 2.0 = 1.0 each call

        # Reference recurrence computed independently of the implementation.
        ref_velocity, ref_weight = 0.0, 1.0
        for _ in range(3):
            gradient = node_value * gradient_input
            ref_velocity = momentum * ref_velocity - lr * gradient
            ref_weight += ref_velocity

            _primed(neuron, last_inputs=[gradient_input], node_value=node_value)
            net.update_weights_sgd(learning_rate=lr, weight_clip_value=None,
                                    bias_clip_value=None, momentum=momentum)

        assert neuron.weight_velocities[0] == pytest.approx(ref_velocity)
        assert neuron.weights[0] == pytest.approx(ref_weight)


class TestSGDBatched:
    def test_averages_accumulated_gradient_over_batch_size(self):
        net, neuron = _single_output_neuron_network(weights=[1.0], bias=0.0)
        # Simulate 4 accumulated samples with gradients 1, 2, 3, 4 -> sum 10.
        neuron.weight_gradient_accumulations = [10.0]
        neuron.bias_gradient_accumulation = 10.0

        net.apply_gradient_accumulations_sgd(learning_rate=0.1, batch_size=4,
                                              weight_clip_value=None, bias_clip_value=None,
                                              momentum=0.0)

        avg_gradient = 10.0 / 4
        assert neuron.weights[0] == pytest.approx(1.0 - 0.1 * avg_gradient)
        assert neuron.bias == pytest.approx(0.0 - 0.1 * avg_gradient)

    def test_batch_size_one_matches_single_sample_update(self):
        net_a, neuron_a = _single_output_neuron_network(weights=[1.0, -2.0], bias=0.3)
        net_b, neuron_b = _single_output_neuron_network(weights=[1.0, -2.0], bias=0.3)
        inputs, node_value = [0.5, 1.5], 0.2

        _primed(neuron_a, inputs, node_value)
        net_a.update_weights_sgd(learning_rate=0.05, weight_clip_value=None,
                                  bias_clip_value=None, momentum=0.9)

        _primed(neuron_b, inputs, node_value)
        net_b.accumulate_gradients()
        net_b.apply_gradient_accumulations_sgd(learning_rate=0.05, batch_size=1,
                                                weight_clip_value=None, bias_clip_value=None,
                                                momentum=0.9)

        assert neuron_a.weights == pytest.approx(neuron_b.weights)
        assert neuron_a.bias == pytest.approx(neuron_b.bias)


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class TestAdamBiasCorrection:
    def test_step_counter_increments_once_per_call(self):
        net, neuron = _single_output_neuron_network(weights=[1.0], bias=0.0)
        assert net.adam_step == 0
        for expected_step in range(1, 4):
            _primed(neuron, last_inputs=[1.0], node_value=0.1)
            net.update_weights_adam(learning_rate=0.001, weight_clip_value=None, bias_clip_value=None)
            assert net.adam_step == expected_step

    def test_matches_reference_adam_over_many_steps(self):
        """
        Runs 100 update steps with a constant gradient and compares the
        network's resulting velocity, squared-gradient accumulator, and
        weight against an independently written reference implementation of
        Adam's bias-corrected update rule. This specifically guards against
        the previously-fixed bug where bias correction used a constant
        divisor (1 - momentum) instead of (1 - momentum**t).
        """
        net, neuron = _single_output_neuron_network(weights=[0.5], bias=0.0)
        lr, momentum, beta2 = 0.01, 0.9, 0.999
        gradient_input = 1.0
        node_value = 0.3  # weight_gradient = 0.3 each call

        ref_velocity = ref_sq_grad = 0.0
        ref_weight = 0.5
        for t in range(1, 101):
            gradient = node_value * gradient_input
            ref_velocity = momentum * ref_velocity + (1 - momentum) * gradient
            ref_sq_grad = beta2 * ref_sq_grad + (1 - beta2) * (gradient ** 2)
            v_hat = ref_velocity / (1 - momentum ** t)
            s_hat = ref_sq_grad / (1 - beta2 ** t)
            ref_weight -= lr * v_hat / (math.sqrt(s_hat) + 1e-8)

            _primed(neuron, last_inputs=[gradient_input], node_value=node_value)
            net.update_weights_adam(learning_rate=lr, weight_clip_value=None, bias_clip_value=None,
                                     momentum=momentum, squared_gradient_term=beta2)

        assert net.adam_step == 100
        assert neuron.weight_velocities[0] == pytest.approx(ref_velocity, rel=1e-9)
        assert neuron.weight_squared_gradient_accumulations[0] == pytest.approx(ref_sq_grad, rel=1e-9)
        assert neuron.weights[0] == pytest.approx(ref_weight, rel=1e-6)


class TestAdamBatched:
    def test_batch_size_one_matches_single_sample_update(self):
        net_a, neuron_a = _single_output_neuron_network(weights=[1.0, -2.0], bias=0.3)
        net_b, neuron_b = _single_output_neuron_network(weights=[1.0, -2.0], bias=0.3)
        inputs, node_value = [0.5, 1.5], 0.2

        _primed(neuron_a, inputs, node_value)
        net_a.update_weights_adam(learning_rate=0.001, weight_clip_value=None, bias_clip_value=None,
                                   momentum=0.9, squared_gradient_term=0.999)

        _primed(neuron_b, inputs, node_value)
        net_b.accumulate_gradients()
        net_b.apply_gradient_accumulations_adam(learning_rate=0.001, batch_size=1,
                                                 weight_clip_value=None, bias_clip_value=None,
                                                 momentum=0.9, squared_gradient_term=0.999)

        assert neuron_a.weights == pytest.approx(neuron_b.weights)
        assert neuron_a.bias == pytest.approx(neuron_b.bias)
        assert net_a.adam_step == net_b.adam_step == 1

    def test_step_counter_shared_across_batched_calls(self):
        net, neuron = _single_output_neuron_network(weights=[1.0], bias=0.0)
        for expected_step in range(1, 4):
            neuron.weight_gradient_accumulations = [0.5]
            neuron.bias_gradient_accumulation = 0.5
            net.apply_gradient_accumulations_adam(learning_rate=0.001, batch_size=1,
                                                   weight_clip_value=None, bias_clip_value=None)
            assert net.adam_step == expected_step


# ---------------------------------------------------------------------------
# AdaGrad
# ---------------------------------------------------------------------------

class TestAdaGradSingleSample:
    def test_first_call_matches_exact_formula(self):
        net, neuron = _single_output_neuron_network(weights=[1.0], bias=0.0)
        _primed(neuron, last_inputs=[2.0], node_value=0.5)  # gradient = 1.0

        net.update_weights_adagrad(learning_rate=0.1, weight_clip_value=None, bias_clip_value=None)

        expected_sq_grad = 1.0 ** 2
        expected_weight = 1.0 - 0.1 / (math.sqrt(expected_sq_grad) + 1e-8) * 1.0
        assert neuron.weight_squared_gradient_accumulations[0] == pytest.approx(expected_sq_grad)
        assert neuron.weights[0] == pytest.approx(expected_weight)

    def test_squared_gradient_accumulator_never_decays(self):
        # Unlike Adam/RMSprop's exponential decay, AdaGrad's accumulator
        # should only ever grow (or stay flat if the gradient is exactly 0),
        # since it's a plain running sum with no decay term.
        net, neuron = _single_output_neuron_network(weights=[1.0], bias=0.0)
        accumulated = []
        for _ in range(5):
            _primed(neuron, last_inputs=[1.0], node_value=0.3)
            net.update_weights_adagrad(learning_rate=0.1, weight_clip_value=None, bias_clip_value=None)
            accumulated.append(neuron.weight_squared_gradient_accumulations[0])
        assert accumulated == sorted(accumulated)  # monotonically non-decreasing
        assert accumulated[-1] > accumulated[0]  # and strictly grew overall


class TestAdaGradBatched:
    def test_batch_size_one_matches_single_sample_update(self):
        net_a, neuron_a = _single_output_neuron_network(weights=[1.0, -2.0], bias=0.3)
        net_b, neuron_b = _single_output_neuron_network(weights=[1.0, -2.0], bias=0.3)
        inputs, node_value = [0.5, 1.5], 0.2

        _primed(neuron_a, inputs, node_value)
        net_a.update_weights_adagrad(learning_rate=0.01, weight_clip_value=None, bias_clip_value=None)

        _primed(neuron_b, inputs, node_value)
        net_b.accumulate_gradients()
        net_b.apply_gradient_accumulations_adagrad(learning_rate=0.01, batch_size=1,
                                                    weight_clip_value=None, bias_clip_value=None)

        assert neuron_a.weights == pytest.approx(neuron_b.weights)
        assert neuron_a.bias == pytest.approx(neuron_b.bias)


# ---------------------------------------------------------------------------
# RMSprop
# ---------------------------------------------------------------------------

class TestRMSpropSingleSample:
    def test_first_call_matches_exact_formula(self):
        net, neuron = _single_output_neuron_network(weights=[1.0], bias=0.0)
        _primed(neuron, last_inputs=[2.0], node_value=0.5)  # gradient = 1.0
        beta2 = 0.999

        net.update_weights_rmsprop(learning_rate=0.001, weight_clip_value=None,
                                    bias_clip_value=None, squared_gradient_term=beta2)

        expected_sq_grad = beta2 * 0.0 + (1 - beta2) * (1.0 ** 2)
        expected_weight = 1.0 - 0.001 * 1.0 / (math.sqrt(expected_sq_grad) + 1e-8)
        assert neuron.weight_squared_gradient_accumulations[0] == pytest.approx(expected_sq_grad)
        assert neuron.weights[0] == pytest.approx(expected_weight)

    def test_squared_gradient_accumulator_decays_unlike_adagrad(self):
        # With a decaying running average, repeatedly feeding a *shrinking*
        # gradient should let old (larger) squared-gradient history decay
        # away, unlike AdaGrad's ever-growing sum.
        net, neuron = _single_output_neuron_network(weights=[1.0], bias=0.0)
        _primed(neuron, last_inputs=[1.0], node_value=10.0)  # one big gradient
        net.update_weights_rmsprop(learning_rate=0.001, weight_clip_value=None, bias_clip_value=None)
        after_big = neuron.weight_squared_gradient_accumulations[0]

        for _ in range(50):
            _primed(neuron, last_inputs=[1.0], node_value=0.0)  # gradient = 0 from now on
            net.update_weights_rmsprop(learning_rate=0.001, weight_clip_value=None, bias_clip_value=None)

        assert neuron.weight_squared_gradient_accumulations[0] < after_big


class TestRMSpropBatched:
    def test_batch_size_one_matches_single_sample_update(self):
        net_a, neuron_a = _single_output_neuron_network(weights=[1.0, -2.0], bias=0.3)
        net_b, neuron_b = _single_output_neuron_network(weights=[1.0, -2.0], bias=0.3)
        inputs, node_value = [0.5, 1.5], 0.2

        _primed(neuron_a, inputs, node_value)
        net_a.update_weights_rmsprop(learning_rate=0.001, weight_clip_value=None, bias_clip_value=None)

        _primed(neuron_b, inputs, node_value)
        net_b.accumulate_gradients()
        net_b.apply_gradient_accumulations_RMSprop(learning_rate=0.001, batch_size=1,
                                                     weight_clip_value=None, bias_clip_value=None)

        assert neuron_a.weights == pytest.approx(neuron_b.weights)
        assert neuron_a.bias == pytest.approx(neuron_b.bias)


# ---------------------------------------------------------------------------
# Gradient clipping — one check per optimizer
# ---------------------------------------------------------------------------

class TestGradientClipping:
    def test_sgd_clips_large_weight_gradient(self):
        net, neuron = _single_output_neuron_network(weights=[0.0], bias=0.0)
        _primed(neuron, last_inputs=[1.0], node_value=1000.0)  # huge gradient
        net.update_weights_sgd(learning_rate=1.0, weight_clip_value=5.0,
                                bias_clip_value=None, momentum=0.0)
        # velocity = -lr * clipped_gradient = -1.0 * 5.0 = -5.0 exactly
        assert neuron.weights[0] == pytest.approx(-5.0)

    def test_sgd_clip_none_lets_full_gradient_through(self):
        net, neuron = _single_output_neuron_network(weights=[0.0], bias=0.0)
        _primed(neuron, last_inputs=[1.0], node_value=1000.0)
        net.update_weights_sgd(learning_rate=1.0, weight_clip_value=None,
                                bias_clip_value=None, momentum=0.0)
        assert neuron.weights[0] == pytest.approx(-1000.0)

    def test_adam_clips_large_weight_gradient_before_moment_update(self):
        net, neuron = _single_output_neuron_network(weights=[0.0], bias=0.0)
        _primed(neuron, last_inputs=[1.0], node_value=1000.0)
        net.update_weights_adam(learning_rate=0.001, weight_clip_value=5.0, bias_clip_value=None)
        # First moment should reflect the clipped gradient (5.0), not 1000.0.
        assert neuron.weight_velocities[0] == pytest.approx((1 - 0.9) * 5.0)

    def test_adagrad_clips_large_weight_gradient(self):
        net, neuron = _single_output_neuron_network(weights=[0.0], bias=0.0)
        _primed(neuron, last_inputs=[1.0], node_value=1000.0)
        net.update_weights_adagrad(learning_rate=0.01, weight_clip_value=5.0, bias_clip_value=None)
        assert neuron.weight_squared_gradient_accumulations[0] == pytest.approx(5.0 ** 2)

    def test_rmsprop_clips_large_weight_gradient(self):
        net, neuron = _single_output_neuron_network(weights=[0.0], bias=0.0)
        _primed(neuron, last_inputs=[1.0], node_value=1000.0)
        net.update_weights_rmsprop(learning_rate=0.001, weight_clip_value=5.0, bias_clip_value=None)
        expected_sq_grad = 0.999 * 0.0 + 0.001 * (5.0 ** 2)
        assert neuron.weight_squared_gradient_accumulations[0] == pytest.approx(expected_sq_grad)


# ---------------------------------------------------------------------------
# No ZeroDivisionError on the very first call (relies on the 1e-8 epsilon)
# ---------------------------------------------------------------------------

class TestFirstCallNoZeroDivision:
    @pytest.mark.parametrize("method_name, kwargs", [
        ("update_weights_sgd", {}),
        ("update_weights_adam", {}),
        ("update_weights_adagrad", {}),
        ("update_weights_rmsprop", {}),
    ])
    def test_fresh_network_first_update_does_not_raise(self, method_name, kwargs):
        net, neuron = _single_output_neuron_network(weights=[0.5], bias=0.0)
        _primed(neuron, last_inputs=[1.0], node_value=0.3)
        getattr(net, method_name)(**kwargs)  # should not raise ZeroDivisionError


# ---------------------------------------------------------------------------
# _apply_optimizer_update dispatch
# ---------------------------------------------------------------------------

class TestApplyOptimizerUpdateDispatch:
    @pytest.mark.parametrize("optimizer", ["sgd", "adam", "adagrad", "rmsprop"])
    def test_valid_optimizer_updates_weights_and_resets_accumulators(self, optimizer):
        net, neuron = _single_output_neuron_network(weights=[1.0], bias=0.0)
        neuron.weight_gradient_accumulations = [2.0]
        neuron.bias_gradient_accumulation = 2.0

        net._apply_optimizer_update(optimizer, learning_rate=0.01, batch_size=1,
                                     weight_clip_value=5.0, bias_clip_value=10.0,
                                     momentum=0.9, squared_gradient_term=0.999)

        assert neuron.weights[0] != 1.0  # an update actually happened
        assert neuron.weight_gradient_accumulations == [0.0]  # accumulators reset
        assert neuron.bias_gradient_accumulation == 0.0

    def test_unknown_optimizer_raises_and_does_not_reset_accumulators(self):
        net, neuron = _single_output_neuron_network(weights=[1.0], bias=0.0)
        neuron.weight_gradient_accumulations = [2.0]
        neuron.bias_gradient_accumulation = 2.0

        with pytest.raises(ValueError):
            net._apply_optimizer_update("not_a_real_optimizer", learning_rate=0.01, batch_size=1,
                                         weight_clip_value=5.0, bias_clip_value=10.0,
                                         momentum=0.9, squared_gradient_term=0.999)

        # The raise happens before reset_accumulated_gradients() is reached,
        # so the (unapplied) accumulated gradients should still be intact --
        # confirming this really does fail loudly rather than silently
        # discarding the batch's gradients.
        assert neuron.weight_gradient_accumulations == [2.0]
        assert neuron.bias_gradient_accumulation == 2.0
        assert neuron.weights[0] == 1.0  # unchanged


# ---------------------------------------------------------------------------
# accumulate_gradients() / reset_accumulated_gradients()
# ---------------------------------------------------------------------------

class TestAccumulateGradients:
    def test_repeated_calls_without_reset_are_additive(self):
        net, neuron = _single_output_neuron_network(weights=[1.0, 2.0], bias=0.0)
        _primed(neuron, last_inputs=[3.0, 4.0], node_value=0.5)

        net.accumulate_gradients()
        net.accumulate_gradients()  # same node_value/last_inputs, no reset in between

        assert neuron.weight_gradient_accumulations[0] == pytest.approx(2 * (0.5 * 3.0))
        assert neuron.weight_gradient_accumulations[1] == pytest.approx(2 * (0.5 * 4.0))
        assert neuron.bias_gradient_accumulation == pytest.approx(2 * 0.5)


class TestResetAccumulatedGradients:
    def test_only_zeroes_gradient_accumulators_not_other_state(self):
        net, neuron = _single_output_neuron_network(weights=[1.0, 2.0], bias=0.5)
        neuron.weight_gradient_accumulations = [9.0, 9.0]
        neuron.bias_gradient_accumulation = 9.0
        neuron.weight_velocities = [7.0, 7.0]
        neuron.bias_velocity = 7.0
        neuron.weight_squared_gradient_accumulations = [3.0, 3.0]

        net.reset_accumulated_gradients()

        assert neuron.weight_gradient_accumulations == [0.0, 0.0]
        assert neuron.bias_gradient_accumulation == 0.0
        # Everything else must be untouched.
        assert neuron.weights == [1.0, 2.0]
        assert neuron.bias == 0.5
        assert neuron.weight_velocities == [7.0, 7.0]
        assert neuron.bias_velocity == 7.0
        assert neuron.weight_squared_gradient_accumulations == [3.0, 3.0]
