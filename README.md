# Neural Network

A small, dependency-free feedforward neural network library written in pure
Python — a from-first-principles implementation of neurons, activation
functions, backpropagation, and gradient-based optimizers, with no NumPy or
other numerical libraries required.

## Features

- **Configurable architecture** — any number of hidden layers, each with its
  own size, via `NeuralNetwork(hidden_layer_sizes=[...], ...)`
- **Activation functions** — sigmoid, ReLU, leaky ReLU, tanh, linear, and
  softmax (softmax is applied at the output layer only, since it depends on
  all output neurons jointly rather than one neuron at a time)
- **Cost functions** — mean squared error (MSE) and cross-entropy.
  Cross-entropy requires a `sigmoid` or `softmax` output activation, since
  that's the only pairing for which its gradient shortcut is mathematically
  valid — the library raises a clear error if you try any other combination
- **Optimizers** — SGD (with momentum), Adam (with correctly time-stepped
  bias correction), AdaGrad, and RMSprop, each available as both a
  per-sample update and a mini-batch-averaged update
- **Dropout** — inverted dropout during training, with gradients correctly
  zeroed for dropped neurons during backpropagation
- **Mini-batch training** — `train()` handles shuffling, batching,
  learning-rate decay, gradient clipping, and progress logging, including
  flushing a trailing partial batch at the end of each epoch instead of
  discarding it
- **Save / load** — `save()` and `NeuralNetwork.load()` persist and restore
  the full architecture (layer sizes, activation functions, cost function)
  along with every neuron's weights, biases, and optimizer state, not just
  the raw numbers

## Installation

This project isn't published as a package yet. For now, clone the
repository and import directly from the repo root:

```bash
git clone https://github.com/<your-username>/Neural-Network.git
cd Neural-Network
```

```python
from network import NeuralNetwork
```

## Quick start

```python
from network import NeuralNetwork

# 2 inputs -> one hidden layer of 4 neurons -> 2 outputs
network = NeuralNetwork(
    hidden_layer_sizes=[4],
    input_size=2,
    output_size=2,
    hidden_activation='relu',
    output_activation='softmax',
    cost_function='cross-entropy',
    random_seed=42
)

# data is a list of (inputs, expected_outputs) tuples
data = [
    ([0.1, 0.9], [1.0, 0.0]),
    ([0.8, 0.2], [0.0, 1.0]),
    # ...
]

network.train(
    data,
    epochs=50,
    optimizer='adam',
    initial_learning_rate=0.01,
    batch_size=8,
    print_rate=0
)

prediction = network.forward([0.3, 0.7])
print(prediction)

network.save("model.json")
loaded_network = NeuralNetwork.load("model.json")
```

## Project structure

- `neuron.py` — the `Neuron` class: weights, bias, activation function, and
  the per-neuron forward pass
- `network.py` — the `NeuralNetwork` class: layer construction, forward
  propagation, backpropagation, optimizers, and training
- `digits/`, `iris/`, `testing/` — example training scripts against MNIST
  digits, the Iris dataset, and synthetic data

## Notes

- This is a pure-Python implementation with no vectorization, so it's meant
  for learning and small experiments rather than large-scale training —
  expect it to be considerably slower than NumPy- or tensor-library-based
  implementations.
- Model files saved by earlier versions of `save()` (before architecture
  metadata was added) are not compatible with the current `load()` and will
  need to be regenerated.

## License

MIT — see [LICENSE](LICENSE).
