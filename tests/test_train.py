"""
Tests for deepnode.NeuralNetwork.train(): batching/epoch edge cases, the
learning-rate decay schedule, optimizer-name validation, dropout wiring, and
a convergence smoke test for every optimizer.

Update-count tests use optimizer='adam' throughout and read net.adam_step
as an exact counter of how many times _apply_optimizer_update actually
fired, since adam_step increments by precisely 1 per optimizer step
regardless of batch_size -- this makes batching/flush behavior countable
without needing to inspect weight deltas.
"""
import math

import pytest

from deepnode import NeuralNetwork


def _toy_data(n):
    """n arbitrary but fixed (input, one-hot expected) pairs, 2 classes."""
    data = []
    for i in range(n):
        x = (i % 5) / 5.0
        y = ((i + 2) % 5) / 5.0
        label = i % 2
        expected = [1.0, 0.0] if label == 0 else [0.0, 1.0]
        data.append(([x, y], expected))
    return data


# ---------------------------------------------------------------------------
# Batch / epoch edge cases, counted via adam_step
# ---------------------------------------------------------------------------

class TestBatchingUpdateCounts:
    def test_batch_size_exactly_divides_data_no_double_update(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        data = _toy_data(8)
        net.train(data, epochs=1, optimizer='adam', batch_size=4, print_rate=0)
        # 8 / 4 = 2 in-loop updates, remainder 8 % 4 == 0 -> no flush.
        assert net.adam_step == 2

    def test_leftover_partial_batch_is_flushed(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        data = _toy_data(10)
        net.train(data, epochs=1, optimizer='adam', batch_size=4, print_rate=0)
        # 2 in-loop updates (samples 4 and 8) + 1 flush for the trailing 2.
        assert net.adam_step == 3

    def test_batch_size_one_updates_every_sample(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        data = _toy_data(5)
        net.train(data, epochs=1, optimizer='adam', batch_size=1, print_rate=0)
        assert net.adam_step == 5  # remainder is always 0 when batch_size == 1

    def test_batch_size_larger_than_dataset_flushes_once(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        data = _toy_data(3)
        net.train(data, epochs=1, optimizer='adam', batch_size=10, print_rate=0)
        # The in-loop trigger never fires (idx+1 never reaches 10), so the
        # only update is the end-of-epoch flush of all 3 samples.
        assert net.adam_step == 1

    def test_update_counts_compound_across_epochs(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        data = _toy_data(8)
        net.train(data, epochs=3, optimizer='adam', batch_size=4, print_rate=0)
        assert net.adam_step == 2 * 3

    def test_zero_batch_size_raises_zero_division_with_data(self):
        # Documenting current behavior rather than asserting it's "correct":
        # batch_size=0 is not validated anywhere, so (idx + 1) % batch_size
        # inside the per-sample loop raises ZeroDivisionError immediately on
        # the very first sample when data is non-empty.
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        with pytest.raises(ZeroDivisionError):
            net.train(_toy_data(3), epochs=1, optimizer='adam', batch_size=0, print_rate=0)

    def test_zero_batch_size_raises_zero_division_with_empty_data(self):
        # With empty data the per-sample loop never runs, so the
        # ZeroDivisionError instead comes from the end-of-epoch
        # `len(data) % batch_size` remainder calculation (0 % 0).
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        with pytest.raises(ZeroDivisionError):
            net.train([], epochs=1, optimizer='adam', batch_size=0, print_rate=0)


# ---------------------------------------------------------------------------
# Empty data / zero epochs — network must be left untouched
# ---------------------------------------------------------------------------

class TestNoOpTraining:
    def test_empty_data_does_not_raise_and_leaves_network_unchanged(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        original_weights = [w for n in net.output_layer for w in n.weights]

        net.train([], epochs=3, optimizer='adam', batch_size=4, print_rate=0)

        after_weights = [w for n in net.output_layer for w in n.weights]
        assert after_weights == original_weights
        assert net.adam_step == 0

    def test_zero_epochs_leaves_network_unchanged(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        original_weights = [w for n in net.output_layer for w in n.weights]

        net.train(_toy_data(10), epochs=0, optimizer='adam', batch_size=4, print_rate=0)

        after_weights = [w for n in net.output_layer for w in n.weights]
        assert after_weights == original_weights
        assert net.adam_step == 0


# ---------------------------------------------------------------------------
# Optimizer validation happens before any training occurs
# ---------------------------------------------------------------------------

class TestOptimizerValidation:
    def test_invalid_optimizer_raises_before_touching_the_network(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        original_weights = [w for n in net.output_layer for w in n.weights]

        with pytest.raises(ValueError, match="optimizer"):
            net.train(_toy_data(10), epochs=5, optimizer='not_a_real_optimizer', print_rate=0)

        after_weights = [w for n in net.output_layer for w in n.weights]
        assert after_weights == original_weights  # nothing was touched

    @pytest.mark.parametrize("optimizer", ["sgd", "adam", "adagrad", "rmsprop"])
    def test_all_four_optimizer_names_are_accepted(self, optimizer):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        net.train(_toy_data(4), epochs=1, optimizer=optimizer, batch_size=4, print_rate=0)


# ---------------------------------------------------------------------------
# print_rate=0 must never divide by zero
# ---------------------------------------------------------------------------

class TestPrintRate:
    def test_print_rate_zero_trains_silently_without_crashing(self, capsys):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        net.train(_toy_data(20), epochs=1, optimizer='adam', batch_size=4, print_rate=0)
        captured = capsys.readouterr()
        assert captured.out == ""  # no progress output at all

    def test_nonzero_print_rate_does_print_progress(self, capsys):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        net.train(_toy_data(5), epochs=1, optimizer='adam', batch_size=4, print_rate=1)
        captured = capsys.readouterr()
        assert "Epoch" in captured.out


# ---------------------------------------------------------------------------
# Learning-rate decay schedule
# ---------------------------------------------------------------------------

class TestLearningRateDecay:
    def test_decay_schedule_applied_per_epoch(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        recorded_lrs = []
        original_apply = net._apply_optimizer_update

        def spy(optimizer, learning_rate, batch_size, weight_clip_value,
                bias_clip_value, momentum, squared_gradient_term):
            recorded_lrs.append(learning_rate)
            return original_apply(optimizer, learning_rate, batch_size, weight_clip_value,
                                   bias_clip_value, momentum, squared_gradient_term)

        net._apply_optimizer_update = spy

        data = [([0.1, 0.2], [1.0, 0.0]), ([0.3, 0.4], [0.0, 1.0])]
        net.train(data, epochs=3, optimizer='sgd', initial_learning_rate=1.0,
                  learning_rate_decay=0.5, batch_size=1, print_rate=0)

        # 2 samples/epoch, batch_size=1 -> 2 calls per epoch, each at that
        # epoch's decayed rate: 1.0, 1.0*0.5, 1.0*0.5^2.
        assert recorded_lrs == pytest.approx([1.0, 1.0, 0.5, 0.5, 0.25, 0.25])

    def test_no_decay_keeps_learning_rate_constant(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        recorded_lrs = []
        original_apply = net._apply_optimizer_update

        def spy(optimizer, learning_rate, batch_size, weight_clip_value,
                bias_clip_value, momentum, squared_gradient_term):
            recorded_lrs.append(learning_rate)
            return original_apply(optimizer, learning_rate, batch_size, weight_clip_value,
                                   bias_clip_value, momentum, squared_gradient_term)

        net._apply_optimizer_update = spy

        data = [([0.1, 0.2], [1.0, 0.0])]
        net.train(data, epochs=4, optimizer='sgd', initial_learning_rate=0.2,
                  learning_rate_decay=1.0, batch_size=1, print_rate=0)

        assert recorded_lrs == pytest.approx([0.2, 0.2, 0.2, 0.2])


# ---------------------------------------------------------------------------
# dropout_rate actually reaches forward() during training
# ---------------------------------------------------------------------------

class TestDropoutWiring:
    def test_dropout_rate_propagates_into_forward_during_training(self, monkeypatch):
        # random.random() is patched to always return 0.0, which is < any
        # positive dropout_rate, so every hidden neuron must be dropped on
        # whichever forward pass this forces -- this only happens at all if
        # train() actually threads dropout_rate through to forward().
        monkeypatch.setattr('deepnode.neuron.random.random', lambda: 0.0)

        net = NeuralNetwork(hidden_layer_sizes=[4], input_size=2, output_size=2, random_seed=1)
        data = [([0.1, 0.2], [1.0, 0.0]), ([0.3, 0.4], [0.0, 1.0])]

        net.train(data, epochs=1, optimizer='sgd', batch_size=1, dropout_rate=0.5, print_rate=0)

        assert all(n.last_dropout_mask == 0.0 for n in net.hidden_layers[0])

    def test_dropout_rate_zero_never_drops_during_training(self, monkeypatch):
        monkeypatch.setattr('deepnode.neuron.random.random', lambda: 0.0)

        net = NeuralNetwork(hidden_layer_sizes=[4], input_size=2, output_size=2, random_seed=1)
        data = [([0.1, 0.2], [1.0, 0.0])]

        net.train(data, epochs=1, optimizer='sgd', batch_size=1, dropout_rate=0.0, print_rate=0)

        # dropout_rate=0.0 means "random.random() < 0.0" is never true,
        # regardless of what random.random() is patched to return.
        assert all(n.last_dropout_mask == 1.0 for n in net.hidden_layers[0])


# ---------------------------------------------------------------------------
# train() shuffles its input list in place — documenting a real side effect
# ---------------------------------------------------------------------------

class TestShufflesInputInPlace:
    def test_data_list_order_is_mutated_by_training(self):
        net = NeuralNetwork(hidden_layer_sizes=[3], input_size=2, output_size=2, random_seed=1)
        data = _toy_data(30)  # large enough that an identity-order shuffle is implausible
        original_order = list(data)

        net.train(data, epochs=1, optimizer='adam', batch_size=4, print_rate=0)

        # `data` is the same list object passed in, and train() calls
        # random.shuffle(data) on it directly -- callers who need to keep
        # their original ordering should pass a copy.
        assert data != original_order
        assert sorted(id(x) for x in data) == sorted(id(x) for x in original_order)  # same elements


# ---------------------------------------------------------------------------
# cross-entropy + sigmoid with non-one-hot (genuinely multi-label) targets
# ---------------------------------------------------------------------------

class TestNonOneHotCrossEntropy:
    def test_multi_label_targets_do_not_crash(self):
        # cost_function='cross-entropy' + output_activation='sigmoid' is
        # accepted by the constructor for independent binary/multi-label
        # targets, but train()'s built-in accuracy/loss tracking assumes a
        # single-correct-class (argmax) labeling scheme. This test only
        # confirms training completes without error on genuinely
        # multi-label data -- it does not assert the printed accuracy
        # metric is meaningful for this case, since it isn't.
        net = NeuralNetwork(hidden_layer_sizes=[4], input_size=2, output_size=3,
                             output_activation='sigmoid', cost_function='cross-entropy',
                             random_seed=1)
        data = [([0.1, 0.2], [1.0, 1.0, 0.0]), ([0.3, 0.4], [0.0, 1.0, 1.0])]
        net.train(data, epochs=2, optimizer='adam', batch_size=1, print_rate=0)


# ---------------------------------------------------------------------------
# Convergence smoke tests — one per optimizer, plus one for mse
# ---------------------------------------------------------------------------

XOR_DATA = [
    ([0.0, 0.0], [1.0, 0.0]),
    ([0.0, 1.0], [0.0, 1.0]),
    ([1.0, 0.0], [0.0, 1.0]),
    ([1.0, 1.0], [1.0, 0.0]),
]


def _categorical_cross_entropy_loss(net, data):
    total = 0.0
    for inputs, expected in data:
        output = net.forward(inputs)
        total += -sum(e * math.log(max(o, 1e-12)) for o, e in zip(output, expected))
    return total


class TestConvergence:
    """
    XOR is a good smoke-test problem here: it's not linearly separable, so
    a network that isn't actually learning (broken gradients, a broken
    optimizer) will not meaningfully reduce its loss on it. These
    hyperparameters were tuned empirically and verified stable across
    several random seeds before being fixed here.
    """

    @pytest.mark.parametrize("optimizer, lr, epochs, momentum", [
        pytest.param('sgd', 0.5, 2000, 0.9, id='sgd'),
        pytest.param('adam', 0.05, 500, 0.9, id='adam'),
        pytest.param('adagrad', 0.5, 1000, 0.9, id='adagrad'),
        pytest.param('rmsprop', 0.05, 500, 0.9, id='rmsprop'),
    ])
    def test_xor_loss_drops_sharply_for_every_optimizer(self, optimizer, lr, epochs, momentum):
        net = NeuralNetwork(hidden_layer_sizes=[8], input_size=2, output_size=2,
                             hidden_activation='tanh', output_activation='softmax',
                             cost_function='cross-entropy', random_seed=42)
        data = list(XOR_DATA)

        initial_loss = _categorical_cross_entropy_loss(net, data)
        net.train(data, epochs=epochs, optimizer=optimizer, initial_learning_rate=lr,
                  momentum=momentum, batch_size=4, print_rate=0)
        final_loss = _categorical_cross_entropy_loss(net, data)

        # Empirically this converges to within ~0.1% of zero loss; a
        # generous 10% threshold leaves wide margin against flakiness
        # while still failing hard if gradients or the optimizer break.
        assert final_loss < initial_loss * 0.1

    def test_mse_loss_drops_sharply_on_toy_regression(self):
        net = NeuralNetwork(hidden_layer_sizes=[8], input_size=2, output_size=1,
                             hidden_activation='tanh', output_activation='sigmoid',
                             cost_function='mse', random_seed=1)
        data = [([0.0, 0.0], [0.1]), ([1.0, 0.0], [0.9]),
                ([0.0, 1.0], [0.9]), ([1.0, 1.0], [0.1])]

        def mse_loss():
            return sum(
                0.5 * sum((o - e) ** 2 for o, e in zip(net.forward(inputs), expected))
                for inputs, expected in data
            )

        initial_loss = mse_loss()
        net.train(list(data), epochs=500, optimizer='adam', initial_learning_rate=0.05,
                  batch_size=4, print_rate=0)
        final_loss = mse_loss()

        assert final_loss < initial_loss * 0.1
