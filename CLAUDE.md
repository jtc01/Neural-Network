# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A feedforward neural network implemented from scratch in Python, with no ML framework dependencies. The core library (`neuron.py`, `network.py`) is used by three consumers: MNIST digit recognition scripts, a simple 2-input test harness, and a Flask REST API.

## Running things

All entry-point scripts import `network` and `neuron` with bare imports (`from network import NeuralNetwork`), which only resolves when the repo root is on `sys.path`. Run them as modules from the repo root, not as bare file paths — `python digits/main.py` fails with `ModuleNotFoundError: No module named 'network'` because Python puts the script's own directory (`digits/`), not the repo root, on `sys.path`.

```bash
# MNIST training (current approach — uses train() method with Adam + mini-batches)
python -m digits.training

# MNIST training (older approach — manual per-sample loop)
python -m digits.main

# Simple 2-input classifier test
python -m testing.main

# Flask REST API (runs on http://127.0.0.1:5000) — this one has a hardcoded
# sys.path.append and can be run directly
python app/app.py
```

Dependencies: `datasets` (HuggingFace), `flask`. Install via `pip install datasets flask` inside the `venv`.

## Architecture

### Core classes

**`Neuron` (`neuron.py`)** — a single neuron. Stores optimizer state (velocities, squared gradient accumulators for Adam, bias equivalents) and intermediate forward-pass values (`last_inputs`, `last_weighted_sum`, `last_output`) needed for backprop. `node_value` holds ∂Cost/∂weighted_sum after backprop runs.

**`NeuralNetwork` (`network.py`)** — a variable-depth feedforward network. Key constructor params:
- `hidden_layers`: list of layer sizes, e.g. `[512, 256, 128]`
- `hidden_activation` / `output_activation`: `'sigmoid'`, `'relu'`, `'leaky_relu'`, `'tanh'`, `'linear'`, `'softmax'`
- `cost_function`: `'mse'` or `'cross-entropy'`

Weights are He-initialized (`gauss(0, sqrt(2/fan_in))`). Softmax is not an activation function on individual neurons — the network stores `using_softmax=True` and applies it post-hoc on the final linear outputs.

### Two backprop APIs

There are two generations of the backprop interface. The older one updates weights immediately per sample:
```python
network.forward(inputs)
network.backpropagate_output_layer(expected_outputs, learning_rate)
network.backpropagate_hidden_layers(learning_rate)
```

The newer mini-batch API separates gradient accumulation from weight updates:
```python
# Per sample in a batch:
network.forward(inputs)
network.compute_output_node_values(expected_outputs)
network.compute_hidden_node_values()
network.accumulate_gradients()
# Once per batch:
network.apply_gradient_accumulations_adam(learning_rate, batch_size, ...)  # or _sgd / _adagrad
network.reset_accumulated_gradients()
```

The `train()` method wraps the mini-batch API and is the preferred entry point for new training code. It takes `optimizer='sgd'|'adam'|'adagrad'` (default `'adam'`) and dispatches to the matching `apply_gradient_accumulations_*` method each batch.

### Saving / loading

`network.save(filename)` and `network.load(filename)` serialize weights, biases, velocities, and squared gradient accumulators to JSON. The `load()` method validates layer/neuron/weight counts before applying — it prints an error and returns early on mismatch rather than raising. Epoch snapshots are saved as `network_state_epoch_N.json` in the repo root.

### Flask API (`app/app.py`)

Endpoints: `POST /create-network`, `POST /forward`, `POST /backpropagate`, `GET /list-networks`. Networks are stored in memory in a module-level dict — they are lost on server restart. The hardcoded `sys.path.append` in `app.py` assumes the repo is at `/Users/joshuacao/Documents/GitHub/Neural-Network`.
