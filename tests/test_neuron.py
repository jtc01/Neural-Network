"""
Unit tests for gradwave.Neuron.

These tests cover construction defaults, every activation function
(including the unknown-activation fallback), forward-pass edge cases
(input-length validation, dropout at rate 0.0 and 1.0, statistical
dropout behavior), and setters/get_info().
"""
import math
import random

import pytest

from gradwave import Neuron


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        neuron = Neuron()
        assert neuron.weights == []
        assert neuron.bias == 0.0
        assert neuron.activation == 'sigmoid'

    def test_weight_velocities_initialized_to_zero_and_matches_weight_count(self):
        neuron = Neuron(weights=[0.1, 0.2, 0.3])
        assert neuron.weight_velocities == [0.0, 0.0, 0.0]
        assert neuron.weight_gradient_accumulations == [0.0, 0.0, 0.0]
        assert neuron.weight_squared_gradient_accumulations == [0.0, 0.0, 0.0]

    def test_bias_optimizer_state_initialized_to_zero(self):
        neuron = Neuron(weights=[0.1], bias=0.5)
        assert neuron.bias == 0.5
        assert neuron.bias_velocity == 0.0
        assert neuron.bias_gradient_accumulation == 0.0
        assert neuron.bias_squared_gradient_accumulation == 0.0

    def test_zero_weight_neuron_is_bias_only(self):
        # A neuron with no weights should still work: forward([]) just
        # returns activation(bias).
        neuron = Neuron(weights=[], bias=2.0, activation='linear')
        assert neuron.forward([]) == 2.0

    def test_last_dropout_mask_defaults_to_one(self):
        neuron = Neuron()
        assert neuron.last_dropout_mask == 1.0

    def test_node_value_defaults_to_zero(self):
        neuron = Neuron()
        assert neuron.node_value == 0.0


# ---------------------------------------------------------------------------
# apply_activation() — exact formulas for every supported activation
# ---------------------------------------------------------------------------

class TestApplyActivation:
    def test_sigmoid(self):
        neuron = Neuron(activation='sigmoid')
        assert neuron.apply_activation(0.0) == pytest.approx(0.5)
        assert neuron.apply_activation(2.0) == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))

    def test_sigmoid_clips_large_positive_input(self):
        neuron = Neuron(activation='sigmoid')
        assert neuron.apply_activation(1000) == 1.0

    def test_sigmoid_clips_large_negative_input(self):
        neuron = Neuron(activation='sigmoid')
        assert neuron.apply_activation(-1000) == 0.0

    def test_sigmoid_near_clip_boundary_does_not_raise(self):
        # math.exp(-500) and math.exp(500) sit right at the clip boundary;
        # confirm no OverflowError right around the edge.
        neuron = Neuron(activation='sigmoid')
        neuron.apply_activation(500.0001)   # just past the clip -> should be 1.0
        neuron.apply_activation(-500.0001)  # just past the clip -> should be 0.0
        neuron.apply_activation(499.9999)   # just inside -> exercises math.exp directly

    def test_relu_positive(self):
        neuron = Neuron(activation='relu')
        assert neuron.apply_activation(3.0) == 3.0

    def test_relu_negative_clamps_to_zero(self):
        neuron = Neuron(activation='relu')
        assert neuron.apply_activation(-3.0) == 0.0

    def test_relu_at_zero(self):
        neuron = Neuron(activation='relu')
        assert neuron.apply_activation(0.0) == 0.0

    def test_leaky_relu_positive(self):
        neuron = Neuron(activation='leaky_relu')
        assert neuron.apply_activation(3.0) == 3.0

    def test_leaky_relu_negative(self):
        neuron = Neuron(activation='leaky_relu')
        assert neuron.apply_activation(-2.0) == pytest.approx(-0.02)

    def test_tanh(self):
        neuron = Neuron(activation='tanh')
        assert neuron.apply_activation(0.0) == pytest.approx(0.0)
        assert neuron.apply_activation(1.0) == pytest.approx(math.tanh(1.0))

    def test_tanh_bounded(self):
        neuron = Neuron(activation='tanh')
        assert -1.0 <= neuron.apply_activation(1000.0) <= 1.0
        assert -1.0 <= neuron.apply_activation(-1000.0) <= 1.0

    def test_linear(self):
        neuron = Neuron(activation='linear')
        assert neuron.apply_activation(7.5) == 7.5
        assert neuron.apply_activation(-3.2) == -3.2

    def test_exponential(self):
        neuron = Neuron(activation='exponential')
        assert neuron.apply_activation(0.0) == pytest.approx(1.0)
        assert neuron.apply_activation(2.0) == pytest.approx(math.exp(2.0))

    def test_exponential_overflows_on_large_input(self):
        # exponential() has no overflow guard (unlike sigmoid). Documenting
        # this as a known edge case: a large enough input raises OverflowError
        # rather than saturating. If exponential() is ever given a clip guard,
        # this test should be updated to assert the new saturating behavior.
        neuron = Neuron(activation='exponential')
        with pytest.raises(OverflowError):
            neuron.apply_activation(10000.0)

    def test_unknown_activation_falls_back_to_sigmoid(self, capsys):
        neuron = Neuron(activation='not_a_real_activation')
        result = neuron.apply_activation(0.0)
        assert result == pytest.approx(0.5)  # sigmoid(0) == 0.5
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "not_a_real_activation" in captured.out


# ---------------------------------------------------------------------------
# forward() — input validation
# ---------------------------------------------------------------------------

class TestForwardInputValidation:
    def test_raises_on_too_few_inputs(self):
        neuron = Neuron(weights=[0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            neuron.forward([1.0, 2.0])

    def test_raises_on_too_many_inputs(self):
        neuron = Neuron(weights=[0.1, 0.2])
        with pytest.raises(ValueError):
            neuron.forward([1.0, 2.0, 3.0])

    def test_error_message_reports_both_counts(self):
        neuron = Neuron(weights=[0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="2.*3|3.*2"):
            neuron.forward([1.0, 2.0])

    def test_matching_length_does_not_raise(self):
        neuron = Neuron(weights=[0.1, 0.2])
        neuron.forward([1.0, 2.0])  # should not raise


# ---------------------------------------------------------------------------
# forward() — core computation
# ---------------------------------------------------------------------------

class TestForwardComputation:
    def test_weighted_sum_and_bias(self):
        neuron = Neuron(weights=[2.0, 3.0], bias=1.0, activation='linear')
        # weighted_sum = 2*1 + 3*2 + 1 = 9
        output = neuron.forward([1.0, 2.0])
        assert output == pytest.approx(9.0)
        assert neuron.last_weighted_sum == pytest.approx(9.0)

    def test_last_inputs_recorded_and_independent_copy(self):
        neuron = Neuron(weights=[1.0], activation='linear')
        inputs = [5.0]
        neuron.forward(inputs)
        assert neuron.last_inputs == [5.0]
        inputs[0] = 999.0  # mutate the original list after the call
        assert neuron.last_inputs == [5.0]  # last_inputs must not have changed

    def test_last_output_matches_return_value(self):
        neuron = Neuron(weights=[1.0, 1.0], bias=0.0, activation='relu')
        output = neuron.forward([-5.0, 1.0])
        assert neuron.last_output == output


# ---------------------------------------------------------------------------
# forward() — dropout behavior
# ---------------------------------------------------------------------------

class TestDropout:
    def test_dropout_rate_zero_never_drops(self):
        neuron = Neuron(weights=[1.0], activation='linear')
        for _ in range(50):
            output = neuron.forward([1.0], dropout_rate=0.0)
            assert output == pytest.approx(1.0)
            assert neuron.last_dropout_mask == pytest.approx(1.0)

    def test_dropout_rate_one_always_drops_and_never_divides_by_zero(self):
        neuron = Neuron(weights=[1.0], activation='linear')
        for _ in range(50):
            output = neuron.forward([1.0], dropout_rate=1.0)
            assert output == 0.0
            assert neuron.last_dropout_mask == 0.0

    def test_dropout_scales_kept_outputs_by_inverse_keep_probability(self):
        random.seed(0)
        neuron = Neuron(weights=[1.0], activation='linear')
        # Force a "kept" pass by using a rate low enough that a fixed
        # seed's first draw keeps it, then check the exact scale factor.
        output = neuron.forward([1.0], dropout_rate=0.5)
        if neuron.last_dropout_mask != 0.0:
            assert neuron.last_dropout_mask == pytest.approx(2.0)
            assert output == pytest.approx(2.0)  # linear(1.0) * 2.0

    def test_dropout_statistically_drops_roughly_expected_fraction(self):
        random.seed(42)
        neuron = Neuron(weights=[1.0], activation='linear')
        n = 2000
        dropped = sum(
            1 for _ in range(n)
            if neuron.forward([1.0], dropout_rate=0.3) == 0.0
        )
        # Expect ~30% dropped; allow a generous statistical tolerance
        # since this is a randomized test.
        assert 0.22 < dropped / n < 0.38

    def test_dropout_mask_is_zero_or_exact_inverse_keep_probability(self):
        random.seed(1)
        neuron = Neuron(weights=[1.0], activation='linear')
        dropout_rate = 0.4
        expected_scale = 1.0 / (1.0 - dropout_rate)
        for _ in range(200):
            neuron.forward([1.0], dropout_rate=dropout_rate)
            assert neuron.last_dropout_mask in (0.0, pytest.approx(expected_scale))


# ---------------------------------------------------------------------------
# Setters
# ---------------------------------------------------------------------------

class TestSetters:
    def test_set_weights_replaces_weights_with_a_copy(self):
        neuron = Neuron(weights=[1.0, 2.0])
        new_weights = [5.0, 6.0]
        neuron.set_weights(new_weights)
        assert neuron.weights == [5.0, 6.0]
        new_weights[0] = 999.0  # mutate original after setting
        assert neuron.weights == [5.0, 6.0]  # must be unaffected (was copied)

    def test_set_bias(self):
        neuron = Neuron(bias=0.0)
        neuron.set_bias(3.5)
        assert neuron.bias == 3.5


# ---------------------------------------------------------------------------
# get_info()
# ---------------------------------------------------------------------------

class TestGetInfo:
    def test_returns_expected_keys(self):
        neuron = Neuron(weights=[1.0], bias=0.5, activation='relu')
        info = neuron.get_info()
        expected_keys = {
            'weights', 'bias', 'activation', 'last_inputs',
            'last_weighted_sum', 'last_output', 'last_dropout_mask', 'node_value'
        }
        assert set(info.keys()) == expected_keys

    def test_reflects_state_after_forward(self):
        neuron = Neuron(weights=[2.0], bias=0.0, activation='linear')
        neuron.forward([3.0])
        info = neuron.get_info()
        assert info['last_output'] == pytest.approx(6.0)
        assert info['last_weighted_sum'] == pytest.approx(6.0)
        assert info['last_inputs'] == [3.0]

    def test_reports_current_activation_not_construction_time_snapshot(self):
        neuron = Neuron(activation='relu')
        neuron.activation = 'tanh'
        assert neuron.get_info()['activation'] == 'tanh'
