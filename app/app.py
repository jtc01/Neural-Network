import sys
sys.path.append('/Users/joshuacao/Documents/GitHub/Neural-Network')

from flask import Flask, request, jsonify
from network import NeuralNetwork

app = Flask(__name__)

# Dictionary to store networks by name
# Structure: { "network_name": NeuralNetwork }
networks = {}

#CREATE NETWORK ENDPOINT
@app.route('/create-network', methods=['POST'])
def create_network():
    """
    Creates a new NeuralNetwork and stores it in the networks dictionary.

    Expected JSON body:
    {
        "name":               (str)         Unique name for this network
        "input_size":         (int)         Number of input nodes
        "hidden_layers":      (list[int])   List of neuron counts per hidden layer, e.g. [4, 3]
        "output_size":        (int)         Number of output nodes
        "hidden_activation":  (str)         Activation for hidden layers: "sigmoid" | "relu" |
                                              "leaky_relu" | "tanh" | "linear" | "exponential"
        "output_activation":  (str)         Activation for output layer: same options + "softmax"
        "cost_function":      (str)         "mse" | "cross-entropy"
        "random_seed":        (int | null)  Seed for reproducible weight init (omit for random)
    }

    Returns 201 on success, 400 on validation error, 409 if name already exists.

    Test Command:
    curl -X POST http://127.0.0.1:5000/create-network \
    -H "Content-Type: application/json" \
    -d '{
        "name": "test_net",
        "input_size": 2,
        "hidden_layers": [4, 3],
        "output_size": 1
    }'
    """
    data = request.get_json(force=True)

    # ── Validate required fields ──────────────────────────────────────────────
    required = ['name', 'hidden_layers']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({
            'error': f"Missing required fields: {', '.join(missing)}"
        }), 400

    name = data['name']

    # Reject duplicate names
    if name in networks:
        return jsonify({
            'error': f"A network named '{name}' already exists. "
                     f"Choose a different name or delete the existing one first."
        }), 409

    hidden_layers = data['hidden_layers']
    if not isinstance(hidden_layers, list) or not all(isinstance(n, int) and n > 0 for n in hidden_layers):
        return jsonify({
            'error': "'hidden_layers' must be a non-empty list of positive integers, e.g. [4, 3]"
        }), 400

    # ── Pull optional params, applying the same defaults as NeuralNetwork.__init__ ──
    input_size         = data.get('input_size',        2)
    output_size        = data.get('output_size',       2)
    hidden_activation  = data.get('hidden_activation', 'sigmoid')
    output_activation  = data.get('output_activation', 'sigmoid')
    cost_function      = data.get('cost_function',     'mse')
    random_seed        = data.get('random_seed',       None)

    # ── Light validation on enum-style fields ─────────────────────────────────
    valid_activations = {'sigmoid', 'relu', 'leaky_relu', 'tanh', 'linear', 'exponential', 'softmax'}
    for field, value in [('hidden_activation', hidden_activation),
                         ('output_activation', output_activation)]:
        if value not in valid_activations:
            return jsonify({
                'error': f"'{field}' must be one of {sorted(valid_activations)}, got '{value}'"
            }), 400

    valid_costs = {'mse', 'cross-entropy'}
    if cost_function not in valid_costs:
        return jsonify({
            'error': f"'cost_function' must be one of {sorted(valid_costs)}, got '{cost_function}'"
        }), 400

    # ── Build the network ─────────────────────────────────────────────────────
    network = NeuralNetwork(
        hidden_layers=hidden_layers,
        input_size=input_size,
        output_size=output_size,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        random_seed=random_seed,
        cost_function=cost_function
    )

    # ── Store it ──────────────────────────────────────────────────────────────
    networks[name] = network

    # ── Build a human-readable architecture string for the response ───────────
    arch = [str(input_size)] + [str(s) for s in hidden_layers] + [str(output_size)]
    architecture_str = ' -> '.join(arch)

    return jsonify({
        'message':      f"Network '{name}' created successfully.",
        'name':         name,
        'architecture': architecture_str,
        'config': {
            'input_size':        input_size,
            'hidden_layers':     hidden_layers,
            'output_size':       output_size,
            'hidden_activation': hidden_activation,
            'output_activation': output_activation,
            'cost_function':     cost_function,
            'random_seed':       random_seed
        }
    }), 201

#FORWARD PASS ENDPOINT
@app.route('/forward', methods=['POST'])
def forward():
    """
    Runs a forward pass through a stored network.
 
    Expected JSON body:
    {
        "name":   (str)         Name of the network to use
        "inputs": (list[float]) Input values, must match the network's input_size
    }
 
    Returns 200 with the network's output on success.
    Returns 400 if inputs are missing, the wrong length, or contain non-numeric values.
    Returns 404 if the network name is not found.

    Test Command:
    curl -X POST http://127.0.0.1:5000/forward \
    -H "Content-Type: application/json" \
    -d '{
        "name": "test_net",
        "inputs": [0.5, 0.8]
    }'
    """
    data = request.get_json(force=True)
 
    # ── Validate required fields ──────────────────────────────────────────────
    missing = [f for f in ['name', 'inputs'] if f not in data]
    if missing:
        return jsonify({
            'error': f"Missing required fields: {', '.join(missing)}"
        }), 400
 
    name   = data['name']
    inputs = data['inputs']
 
    # ── Look up the network ───────────────────────────────────────────────────
    if name not in networks:
        return jsonify({
            'error': f"No network named '{name}' found. "
                     f"Available networks: {list(networks.keys())}"
        }), 404
 
    network = networks[name]
 
    # ── Validate inputs ───────────────────────────────────────────────────────
    if not isinstance(inputs, list):
        return jsonify({'error': "'inputs' must be a list of numbers."}), 400
 
    if len(inputs) != network.input_size:
        return jsonify({
            'error': f"Network '{name}' expects {network.input_size} input(s), "
                     f"but {len(inputs)} were provided."
        }), 400
 
    if not all(isinstance(v, (int, float)) for v in inputs):
        return jsonify({'error': "All values in 'inputs' must be numbers."}), 400
 
    # ── Run the forward pass ──────────────────────────────────────────────────
    outputs = network.forward(inputs)
 
    return jsonify({
        'network': name,
        'inputs':  inputs,
        'outputs': outputs
    }), 200


#BACKPROPAGATION ENDPOINT
@app.route('/backpropagate', methods=['POST'])
def backpropagate():
    """
    Runs a backpropagation pass through a stored network using its last forward pass.
    forward() must have been called on the network before this endpoint is used.
 
    Expected JSON body:
    {
        "name":             (str)         Name of the network to train
        "expected_outputs": (list[float]) Expected output values, must match the network's output_size
        "learning_rate":    (float)       Optional. How much to adjust weights and biases (default: 0.05)
    }
 
    Returns 200 on success.
    Returns 400 if expected_outputs are the wrong length, non-numeric, or no forward pass has been run yet.
    Returns 404 if the network name is not found.

    Test Command:
    curl -X POST http://127.0.0.1:5000/backpropagate \
    -H "Content-Type: application/json" \
    -d '{
        "name": "test_net",
        "expected_outputs": [1.0],
        "learning_rate": 0.05
    }'
    """
    data = request.get_json(force=True)
 
    # ── Validate required fields ──────────────────────────────────────────────
    missing = [f for f in ['name', 'expected_outputs'] if f not in data]
    if missing:
        return jsonify({
            'error': f"Missing required fields: {', '.join(missing)}"
        }), 400
 
    name             = data['name']
    expected_outputs = data['expected_outputs']
    learning_rate    = data.get('learning_rate', 0.05)
 
    # ── Look up the network ───────────────────────────────────────────────────
    if name not in networks:
        return jsonify({
            'error': f"No network named '{name}' found. "
                     f"Available networks: {list(networks.keys())}"
        }), 404
 
    network = networks[name]
 
    # ── Ensure a forward pass has been run first ──────────────────────────────
    if network.last_outputs is None:
        return jsonify({
            'error': f"Network '{name}' has no previous output. "
                      "Call /forward at least once before backpropagating."
        }), 400
 
    # ── Validate expected_outputs ─────────────────────────────────────────────
    if not isinstance(expected_outputs, list):
        return jsonify({'error': "'expected_outputs' must be a list of numbers."}), 400
 
    if len(expected_outputs) != network.output_size:
        return jsonify({
            'error': f"Network '{name}' has {network.output_size} output(s), "
                     f"but {len(expected_outputs)} expected output(s) were provided."
        }), 400
 
    if not all(isinstance(v, (int, float)) for v in expected_outputs):
        return jsonify({'error': "All values in 'expected_outputs' must be numbers."}), 400
 
    # ── Validate learning_rate ────────────────────────────────────────────────
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        return jsonify({'error': "'learning_rate' must be a positive number."}), 400
 
    # ── Run backpropagation ───────────────────────────────────────────────────
    network.backpropagate_output_layer(expected_outputs, learning_rate)
    network.backpropagate_hidden_layers(learning_rate)
 
    return jsonify({
        'network':          name,
        'last_inputs':      network.last_inputs,
        'predicted_outputs': network.last_outputs,
        'expected_outputs': expected_outputs,
        'learning_rate':    learning_rate
    }), 200

@app.route('/list-networks', methods=['GET'])
def list_networks():
    """

    Test Command: http://127.0.0.1:5000/list-networks

    Returns a summary of every network currently stored.
 
    Response structure:
    {
        "count": 2,
        "networks": [
            {
                "name":               "xor_net",
                "architecture":       "2 -> 4 -> 3 -> 1",
                "input_size":         2,
                "hidden_layers":      [4, 3],
                "num_hidden_layers":  2,
                "output_size":        1,
                "hidden_activation":  "relu",
                "output_activation":  "sigmoid",
                "cost_function":      "mse"
            },
            ...
        ]
    }
    """
    network_list = []
 
    for name, network in networks.items():
        # Build the architecture string from the stored network attributes
        arch = (
            [str(network.input_size)]
            + [str(s) for s in network.hidden_layers_sizes]
            + [str(network.output_size)]
        )
 
        # NeuralNetwork replaces 'softmax' with 'linear' internally and sets
        # a flag, so restore the display name for the caller
        output_activation = (
            'softmax' if network.using_softmax else network.output_activation
        )
 
        network_list.append({
            'name':              name,
            'architecture':      ' -> '.join(arch),
            'input_size':        network.input_size,
            'hidden_layers':     network.hidden_layers_sizes,
            'num_hidden_layers': network.num_hidden_layers,
            'output_size':       network.output_size,
            'hidden_activation': network.hidden_activation,
            'output_activation': output_activation,
            'cost_function':     network.cost_function
        })
 
    return jsonify({
        'count':    len(network_list),
        'networks': network_list
    }), 200


if __name__ == '__main__':
    app.run(debug=True)