"""
Tests for deepnode.NeuralNetwork.save() / NeuralNetwork.load().

load() is a @staticmethod factory: it builds a brand-new network from the
saved "architecture" section, then overwrites its weights/biases/optimizer
state. These tests cover round trips across activation/cost-function
combinations, round trips after real training (so optimizer state like
adam_step and weight_velocities is actually exercised, not just
freshly-initialized zeros), the zero-hidden-layer edge case, and every
failure mode load() can hit: a missing file, malformed JSON, a missing
"architecture" section, a missing per-neuron key, and mismatched
architecture/weight counts.
"""
import json
import math

import pytest

from deepnode import NeuralNetwork


TEST_INPUT = [0.3, -0.6, 0.9]


def _outputs_match(net_a, net_b, inputs=TEST_INPUT):
    return net_a.forward(inputs) == pytest.approx(net_b.forward(inputs))


# ---------------------------------------------------------------------------
# Round trips across activation / cost-function combinations
# ---------------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize(
        "hidden_activation, output_activation, cost_function",
        [
            ('relu', 'sigmoid', 'mse'),
            ('tanh', 'linear', 'mse'),
            ('leaky_relu', 'tanh', 'mse'),
            ('sigmoid', 'exponential', 'mse'),
            ('sigmoid', 'sigmoid', 'cross-entropy'),
            ('relu', 'softmax', 'cross-entropy'),
        ],
    )
    def test_forward_output_matches_after_round_trip(self, tmp_path, hidden_activation,
                                                       output_activation, cost_function):
        path = tmp_path / "model.json"
        net = NeuralNetwork(hidden_layer_sizes=[4, 3], input_size=3, output_size=3,
                             hidden_activation=hidden_activation, output_activation=output_activation,
                             cost_function=cost_function, random_seed=1)
        net.forward(TEST_INPUT)  # populate debug state, not required but realistic

        net.save(str(path))
        loaded = NeuralNetwork.load(str(path))

        assert loaded is not None
        assert _outputs_match(net, loaded)

    @pytest.mark.parametrize(
        "hidden_activation, output_activation, cost_function",
        [
            ('relu', 'sigmoid', 'mse'),
            ('relu', 'softmax', 'cross-entropy'),
        ],
    )
    def test_architecture_metadata_matches_after_round_trip(self, tmp_path, hidden_activation,
                                                              output_activation, cost_function):
        path = tmp_path / "model.json"
        net = NeuralNetwork(hidden_layer_sizes=[5, 2], input_size=3, output_size=4,
                             hidden_activation=hidden_activation, output_activation=output_activation,
                             cost_function=cost_function, random_seed=1)
        net.save(str(path))
        loaded = NeuralNetwork.load(str(path))

        assert loaded.input_size == net.input_size
        assert loaded.output_size == net.output_size
        assert loaded.hidden_layer_sizes == net.hidden_layer_sizes
        assert loaded.hidden_activation == net.hidden_activation
        assert loaded.cost_function == net.cost_function
        # get_network_info()'s reporting (softmax vs internal 'linear') must
        # also survive the round trip, not just the raw attribute.
        assert loaded.get_network_info()['output_activation'] == net.get_network_info()['output_activation']

    def test_softmax_flag_and_internal_rewrite_survive_round_trip(self, tmp_path):
        path = tmp_path / "model.json"
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=3,
                             output_activation='softmax', cost_function='cross-entropy',
                             random_seed=1)
        net.save(str(path))
        loaded = NeuralNetwork.load(str(path))

        assert loaded.using_softmax is True
        assert loaded.output_activation == 'linear'  # internal rewrite, same as the original
        assert loaded.get_network_info()['output_activation'] == 'softmax'


class TestZeroHiddenLayerRoundTrip:
    def test_round_trips_correctly(self, tmp_path):
        path = tmp_path / "model.json"
        net = NeuralNetwork(hidden_layer_sizes=[], input_size=3, output_size=2,
                             output_activation='sigmoid', random_seed=1)
        net.save(str(path))
        loaded = NeuralNetwork.load(str(path))

        assert loaded.hidden_layer_sizes == []
        assert loaded.hidden_layers == []
        assert _outputs_match(net, loaded)


# ---------------------------------------------------------------------------
# Round trips after real training — exercises non-zero optimizer state
# ---------------------------------------------------------------------------

class TestRoundTripAfterTraining:
    @pytest.mark.parametrize("optimizer", ["sgd", "adam", "adagrad", "rmsprop"])
    def test_optimizer_state_and_output_survive_after_training(self, tmp_path, optimizer):
        path = tmp_path / "model.json"
        net = NeuralNetwork(hidden_layer_sizes=[4], input_size=2, output_size=2,
                             hidden_activation='relu', output_activation='sigmoid',
                             cost_function='mse', random_seed=1)
        data = [([0.1, 0.2], [1.0, 0.0]), ([0.3, 0.4], [0.0, 1.0]),
                ([0.5, 0.1], [1.0, 0.0]), ([0.2, 0.6], [0.0, 1.0])]
        net.train(data, epochs=5, optimizer=optimizer, batch_size=2, print_rate=0)

        net.save(str(path))
        loaded = NeuralNetwork.load(str(path))

        assert _outputs_match(net, loaded, inputs=[0.15, 0.35])
        assert loaded.adam_step == net.adam_step

        for layer_orig, layer_loaded in zip(net.hidden_layers, loaded.hidden_layers):
            for n_orig, n_loaded in zip(layer_orig, layer_loaded):
                assert n_loaded.weights == pytest.approx(n_orig.weights)
                assert n_loaded.bias == pytest.approx(n_orig.bias)
                assert n_loaded.weight_velocities == pytest.approx(n_orig.weight_velocities)
                assert n_loaded.weight_squared_gradient_accumulations == pytest.approx(
                    n_orig.weight_squared_gradient_accumulations)
                assert n_loaded.bias_velocity == pytest.approx(n_orig.bias_velocity)
                assert n_loaded.bias_squared_gradient_accumulation == pytest.approx(
                    n_orig.bias_squared_gradient_accumulation)

    def test_adam_step_is_nonzero_and_exact_after_adam_training(self, tmp_path):
        path = tmp_path / "model.json"
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        data = [([0.1, 0.2], [1.0, 0.0])] * 8
        net.train(data, epochs=1, optimizer='adam', batch_size=2, print_rate=0)
        assert net.adam_step > 0  # sanity: training actually happened

        net.save(str(path))
        loaded = NeuralNetwork.load(str(path))
        assert loaded.adam_step == net.adam_step

    def test_resuming_adam_training_after_load_continues_step_count(self, tmp_path):
        """
        Regression test for the reason adam_step is persisted at all: if it
        weren't, resuming Adam training after a load() would restart bias
        correction from step 1 and over-correct already-warmed-up moments.
        """
        path = tmp_path / "model.json"
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        data = [([0.1, 0.2], [1.0, 0.0])] * 8
        net.train(data, epochs=1, optimizer='adam', batch_size=2, print_rate=0)
        step_at_save = net.adam_step

        net.save(str(path))
        loaded = NeuralNetwork.load(str(path))
        loaded.train(list(data), epochs=1, optimizer='adam', batch_size=2, print_rate=0)

        assert loaded.adam_step == step_at_save * 2  # continued counting, didn't reset to 0


# ---------------------------------------------------------------------------
# load() failure modes
# ---------------------------------------------------------------------------

class TestLoadFailureModes:
    def test_nonexistent_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            NeuralNetwork.load(str(tmp_path / "does_not_exist.json"))

    def test_malformed_json_raises_json_decode_error(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{this is not valid json")
        with pytest.raises(json.JSONDecodeError):
            NeuralNetwork.load(str(path))

    def test_missing_architecture_section_returns_none(self, tmp_path, capsys):
        # Simulates a file saved by the pre-architecture version of save().
        path = tmp_path / "old_format.json"
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        net.save(str(path))
        data = json.loads(path.read_text())
        del data["architecture"]
        path.write_text(json.dumps(data))

        result = NeuralNetwork.load(str(path))

        assert result is None
        assert "architecture" in capsys.readouterr().out

    def test_missing_adam_step_key_defaults_to_zero(self, tmp_path):
        # An architecture-only file (e.g. hand-crafted, or from a version
        # that added "architecture" before "adam_step" existed) should still
        # load successfully rather than raising a KeyError.
        path = tmp_path / "no_adam_step.json"
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        net.save(str(path))
        data = json.loads(path.read_text())
        del data["adam_step"]
        path.write_text(json.dumps(data))

        loaded = NeuralNetwork.load(str(path))

        assert loaded is not None
        assert loaded.adam_step == 0

    def test_missing_neuron_key_raises_key_error(self, tmp_path):
        path = tmp_path / "missing_key.json"
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        net.save(str(path))
        data = json.loads(path.read_text())
        del data["hidden_layers"][0][0]["weights"]
        path.write_text(json.dumps(data))

        with pytest.raises(KeyError):
            NeuralNetwork.load(str(path))

    def test_mismatched_hidden_layer_count_returns_none(self, tmp_path, capsys):
        path = tmp_path / "bad_layers.json"
        net = NeuralNetwork(hidden_layer_sizes=[3, 2], input_size=2, output_size=2, random_seed=1)
        net.save(str(path))
        data = json.loads(path.read_text())
        del data["hidden_layers"][1]  # drop the second hidden layer's data entirely
        # architecture still claims 2 hidden layers -- data now has only 1
        path.write_text(json.dumps(data))

        result = NeuralNetwork.load(str(path))

        assert result is None
        assert "hidden layers" in capsys.readouterr().out

    def test_mismatched_neurons_per_layer_returns_none(self, tmp_path, capsys):
        path = tmp_path / "bad_neuron_count.json"
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        net.save(str(path))
        data = json.loads(path.read_text())
        del data["hidden_layers"][0][0]  # drop one neuron's data from the layer
        path.write_text(json.dumps(data))

        result = NeuralNetwork.load(str(path))

        assert result is None
        assert "neurons in hidden layer" in capsys.readouterr().out

    def test_mismatched_weight_count_in_hidden_layer_returns_none(self, tmp_path, capsys):
        path = tmp_path / "bad_weight_count.json"
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        net.save(str(path))
        data = json.loads(path.read_text())
        data["hidden_layers"][0][0]["weights"].append(999.0)  # one extra weight
        path.write_text(json.dumps(data))

        result = NeuralNetwork.load(str(path))

        assert result is None
        assert "weights for neuron" in capsys.readouterr().out

    def test_mismatched_output_layer_count_returns_none(self, tmp_path, capsys):
        path = tmp_path / "bad_output_count.json"
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        net.save(str(path))
        data = json.loads(path.read_text())
        del data["output_layer"][0]
        path.write_text(json.dumps(data))

        result = NeuralNetwork.load(str(path))

        assert result is None
        assert "output neurons" in capsys.readouterr().out

    def test_mismatched_weight_count_in_output_layer_returns_none(self, tmp_path, capsys):
        path = tmp_path / "bad_output_weight_count.json"
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        net.save(str(path))
        data = json.loads(path.read_text())
        data["output_layer"][0]["weights"].pop()  # one fewer weight than expected
        path.write_text(json.dumps(data))

        result = NeuralNetwork.load(str(path))

        assert result is None
        assert "weights for output neuron" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# save() basics
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_creates_valid_json_file(self, tmp_path):
        path = tmp_path / "model.json"
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        net.save(str(path))

        assert path.exists()
        data = json.loads(path.read_text())  # should not raise
        assert "architecture" in data
        assert "hidden_layers" in data
        assert "output_layer" in data

    def test_save_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "model.json"
        net_a = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        net_a.save(str(path))

        net_b = NeuralNetwork(hidden_layer_sizes=[5], input_size=2, output_size=2, random_seed=2)
        net_b.save(str(path))  # overwrite with a different architecture

        loaded = NeuralNetwork.load(str(path))
        assert loaded.hidden_layer_sizes == [5]  # reflects net_b, not net_a
