import math
import random
from neuron import Neuron



class NeuralNetwork:
    """
    A flexible feedforward neural network with variable architecture.
    
    Architecture: input_size -> hidden_layer_sizes -> output_size
    Example: input_size=2, hidden_layer_sizes=[3,4,2], output_size=2 creates:
    2 inputs -> 3 hidden neurons -> 4 hidden neurons -> 2 hidden neurons -> 2 outputs
    
    This network performs forward propagation through the layers:
    1. Input layer receives the input data
    2. Multiple hidden layers process data sequentially
    3. Output layer produces final predictions
    """
    
    def __init__(self, hidden_layer_sizes, input_size=2, output_size=2, hidden_activation='sigmoid', output_activation='sigmoid', random_seed=None, cost_function='mse'):
        """
        Initialize the neural network with random weights and biases.
        
        Args:
            hidden_activation (str): Activation function for hidden layer neurons
            output_activation (str): Activation function for output layer neurons
            random_seed (int): Seed for reproducible random weight initialization
        """
        # Set random seed for reproducible results
        if random_seed is not None:
            random.seed(random_seed)
        
        # Store variable architecture parameters
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes.copy()  # Store the architecture
        self.output_size = output_size
        self.num_hidden_layers = len(hidden_layer_sizes)


        valid_cross_entropy_activations = ('sigmoid', 'softmax')
        if cost_function == 'cross-entropy' and output_activation not in valid_cross_entropy_activations:
            raise ValueError(
                f"cost_function='cross-entropy' requires output_activation to be "
                f"'sigmoid' or 'softmax' (got '{output_activation}'). The gradient "
                f"shortcut used for cross-entropy is only mathematically valid for "
                f"these two pairings — using it with another activation would "
                f"silently produce incorrect gradients. Use cost_function='mse' "
                f"if you want a different output activation."
            )
        self.cost_function = cost_function
        
        # Store activation function types
        self.hidden_activation = hidden_activation
        if output_activation == 'softmax':
            self.using_softmax = True
            self.output_activation = 'linear'  # Use exponential for softmax calculation
        else:
            self.using_softmax = False
            self.output_activation = output_activation
        
         # Initialize variable number of hidden layers
        self.hidden_layers = []  # This will be a list of lists (layers of neurons)
        
        # Create each hidden layer
        for layer_idx, layer_size in enumerate(hidden_layer_sizes):
            layer_neurons = []
            
            # Determine input size for this layer
            if layer_idx == 0:
                # First hidden layer takes inputs from input layer
                input_connections = self.input_size
            else:
                # Subsequent hidden layers take inputs from previous hidden layer
                input_connections = hidden_layer_sizes[layer_idx - 1]
            
            # Create neurons for this layer
            for neuron_idx in range(layer_size):
                # Generate random weights for connections to this neuron
                weights = [random.gauss(0, math.sqrt(2/input_connections)) for _ in range(input_connections)]
                weight_velocities = [0.0] * len(weights)
                bias = 0.0  # Initialize bias to 0 for better training stability
                # Create the neuron and add it to the layer
                neuron = Neuron(weights=weights, bias=bias, activation=hidden_activation)
                layer_neurons.append(neuron)
                #print(f"Hidden Layer {layer_idx+1}, Neuron {neuron_idx+1}: weights={[round(w, 3) for w in weights]}, bias={round(bias, 3)}")
        
            self.hidden_layers.append(layer_neurons)
        
        self.output_layer = []
        last_hidden_size = hidden_layer_sizes[-1] if hidden_layer_sizes else self.input_size

    
        for i in range(self.output_size):
            weights = [random.gauss(0, math.sqrt(2/last_hidden_size)) for _ in range(last_hidden_size)]
            bias = 0.0  # Initialize bias to 0 for better training stability
            # Create the output neuron
            neuron = Neuron(weights=weights, bias=bias, activation=self.output_activation)
            self.output_layer.append(neuron)
            
            # Store intermediate values for debugging and visualization
        self.last_inputs = None
        self.last_hidden_outputs = []
        self.last_outputs = None

        self.cost = 0.0
    
    def forward(self, inputs, dropout_rate=0.0):
        """
        Perform forward propagation through the entire network.
        
        Process flow:
        1. Input layer receives the input data
        2. Each hidden neuron processes the inputs
        3. Hidden layer outputs become inputs to output layer
        4. Each output neuron processes hidden layer outputs
        5. Return final network outputs
        
        Args:
            inputs (list): List of input values (should be length 2)
            
        Returns:
            list: List of output values (length 2)
        """


        # Validate input size

        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        
        # Store inputs for debugging
        self.last_inputs = inputs.copy()
        
        # Step 1: Forward pass through all hidden layers sequentially
        current_inputs = inputs
        self.last_hidden_outputs = []  # Reset hidden outputs storage
        
        for layer_idx, layer in enumerate(self.hidden_layers):
            layer_outputs = []
            
            # Process current inputs through each neuron in this layer
            for neuron in layer:
                output = neuron.forward(current_inputs, dropout_rate=dropout_rate)
                layer_outputs.append(output)
            
            # Store this layer's outputs for debugging
            self.last_hidden_outputs.append(layer_outputs.copy())
        
            # The outputs of this layer become inputs to the next layer
            current_inputs = layer_outputs
        
        # Step 2: Forward pass through output layer
        # Each output neuron takes the hidden layer outputs as inputs
        final_outputs = []
        for neuron in self.output_layer:
            output = neuron.forward(current_inputs)
            final_outputs.append(output)

        if self.using_softmax:
            # Apply softmax to the final outputs, assuming linear activation was used for output neurons
            max_output = max(final_outputs)  # For numerical stability
            exp_outputs = [math.exp(o - max_output) for o in final_outputs]  # Subtract max for stability
            sum_exp_outputs = sum(exp_outputs)
            final_outputs = [exp_o / sum_exp_outputs for exp_o in exp_outputs]
        
        # Store final outputs for debugging
        self.last_outputs = final_outputs.copy()
        return final_outputs
    
    def get_network_info(self):
        """
        Get detailed information about the network's current state.
        
        Returns:
            dict: Dictionary containing network information
        """
        # CHANGE: Create architecture string for variable layers
        arch_parts = [str(self.input_size)]
        arch_parts.extend([str(size) for size in self.hidden_layer_sizes])
        arch_parts.append(str(self.output_size))
        architecture_string = "-".join(arch_parts)
        
        info = {
            'architecture': architecture_string,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'num_hidden_layers': self.num_hidden_layers,
            'hidden_activation': self.hidden_activation,
            'output_activation': self.output_activation,
            'last_inputs': self.last_inputs,
            'last_hidden_outputs': self.last_hidden_outputs,
            'last_outputs': self.last_outputs,
            'cost': self.cost,
            'hidden_layers_weights': [],  # CHANGE: Now a list of lists
            'hidden_layers_biases': [],   # CHANGE: Now a list of lists
            'output_layer_weights': [],
            'output_layer_biases': []
        }
        
        # CHANGE: Collect parameters from all hidden layers
        for layer_idx, layer in enumerate(self.hidden_layers):
            layer_weights = []
            layer_biases = []
            for neuron in layer:
                layer_weights.append(neuron.weights)
                layer_biases.append(neuron.bias)
            info['hidden_layers_weights'].append(layer_weights)
            info['hidden_layers_biases'].append(layer_biases)
        
        # Collect output layer parameters (unchanged)
        for neuron in self.output_layer:
            info['output_layer_weights'].append(neuron.weights)
            info['output_layer_biases'].append(neuron.bias)
        
        return info
    
    def predict(self, inputs_list):
        """
        Make predictions on multiple input samples.
        
        Args:
            inputs_list (list): List of input samples, each sample is a list of values
            
        Returns:
            list: List of output predictions, each prediction is a list of values
        """
        predictions = []
        for inputs in inputs_list:
            output = self.forward(inputs)
            predictions.append(output)
        return predictions
    
    def print_network_structure(self):
        """
        Print a visual representation of the variable network structure.
        """
        print("\n=== Network Architecture ===")
        print(f"Input Layer ({self.input_size} nodes)")
        print("     |")
        
        # CHANGE: Print all hidden layers
        for layer_idx, layer in enumerate(self.hidden_layers):
            layer_size = len(layer)
            print(f"Hidden Layer {layer_idx + 1} ({layer_size} neurons)")
            for neuron_idx, neuron in enumerate(layer):
                weights = [round(w, 3) for w in neuron.weights]
                bias = round(neuron.bias, 3)
                print(f"  Neuron {neuron_idx + 1}: W={weights}, b={bias}")
            print("     |")
        
        print(f"Output Layer ({self.output_size} neurons)")
        for i, neuron in enumerate(self.output_layer):
            weights = [round(w, 3) for w in neuron.weights]
            bias = round(neuron.bias, 3)
            print(f"  Neuron {i+1}: W={weights}, b={bias}")        

    def compute_output_node_values(self, expected_outputs):
        """
        Calculate node values for the output layer neurons based on the expected outputs.

        For the output layer, the node value is:
        node_value = ∂Cost/∂(weighted_sum)
                    = ∂Cost/∂activation × ∂activation/∂(weighted_sum)
        
        Where:
        - ∂Cost/∂activation = (predicted_output - expected_output)  [from MSE cost function]
        - ∂activation/∂(weighted_sum) = activation_derivative(weighted_sum)

        expected_outputs (list): The expected output values for this data point
                                    Example: [1, 0] or [0, 1]
        
        Returns:
            None (stores node values in each output neuron)
        """
        # Guard: the cross-entropy shortcut (predicted - expected) is only the
        # correct gradient when paired with 'softmax' or 'sigmoid' output
        # activations. Using it with any other activation silently produces
        # incorrect gradients, so refuse rather than compute them.
        if self.cost_function == 'cross-entropy' and not (self.using_softmax or self.output_activation == 'sigmoid'):
            raise ValueError(
                f"cost_function='cross-entropy' requires output_activation to be "
                f"'sigmoid' or 'softmax' (network is configured with "
                f"'{self.output_activation}'). The gradient shortcut used for "
                f"cross-entropy is only mathematically valid for these two "
                f"pairings — using it with another activation would silently "
                f"produce incorrect gradients."
            )

        if len(expected_outputs) != len(self.output_layer):
            raise ValueError(f"Expected {len(self.output_layer)} output values, got {len(expected_outputs)}")
        
        # Calculate node value for each output neuron
        for neuron_idx, neuron in enumerate(self.output_layer):
            # Get the predicted output (activation) for this neuron. When softmax
            # is in use, neuron.last_output is the raw pre-softmax linear score,
            # so the actual predicted probability must come from the network's
            # post-softmax outputs instead.
            predicted_output = self.last_outputs[neuron_idx] if self.using_softmax else neuron.last_output

            # Get the expected output for this neuron
            expected_output = expected_outputs[neuron_idx]

            if self.cost_function == 'cross-entropy':
                # Calculate ∂Cost/∂activation using the cross-entropy cost function
                neuron.node_value = predicted_output - expected_output
            elif self.cost_function == 'mse':
                # Calculate ∂Cost/∂activation
                # For MSE cost = (1/2) * Σ(predicted - expected)²
                # The derivative is: (predicted - expected)
                cost_derivative = predicted_output - expected_output

                # Calculate ∂activation/∂(weighted_sum)
                # This is the derivative of the activation function
                activation_derivative = self.activation_derivative(
                    neuron.last_weighted_sum,
                    neuron.activation
                )

                # Calculate node value using chain rule
                # node_value = ∂Cost/∂activation × ∂activation/∂(weighted_sum)
                neuron.node_value = cost_derivative * activation_derivative
            else:
                print(f"Warning: Unknown cost function '{self.cost_function}'. Defaulting to MSE.")
                #Same as above
                cost_derivative = predicted_output - expected_output

                activation_derivative = self.activation_derivative(
                    neuron.last_weighted_sum,
                    neuron.activation
                )

                neuron.node_value = cost_derivative * activation_derivative


    def backpropagate_output_layer(self, expected_outputs, learning_rate=0.05, weight_clip_value=5.0, bias_clip_value=10.0, momentum=0.9, update_weights=True):
        """
        Calculate node values for the output layer neurons and update their weights and biases.
        
        For the output layer, the node value is:
        node_value = ∂Cost/∂(weighted_sum)
                    = ∂Cost/∂activation × ∂activation/∂(weighted_sum)
        
        Where:
        - ∂Cost/∂activation = (predicted_output - expected_output)  [from MSE cost function]
        - ∂activation/∂(weighted_sum) = activation_derivative(weighted_sum)
        
        After calculating node values, weights are updated using:
        new_weight = old_weight - learning_rate × node_value × input_to_weight
        
        This method assumes forward() has already been called, so each neuron
        has stored values in last_weighted_sum, last_output, and last_inputs.
        
        Args:
            expected_outputs (list): The expected output values for this data point
                                    Example: [1, 0] or [0, 1]
            learning_rate (float): How much to adjust weights and biases
        
        Returns:
            None (stores node values and updates weights/biases directly in each output neuron)
        """
        # Guard: the cross-entropy shortcut (predicted - expected) is only the
        # correct gradient when paired with 'softmax' or 'sigmoid' output
        # activations. Using it with any other activation silently produces
        # incorrect gradients, so refuse rather than compute them.
        if self.cost_function == 'cross-entropy' and not (self.using_softmax or self.output_activation == 'sigmoid'):
            raise ValueError(
                f"cost_function='cross-entropy' requires output_activation to be "
                f"'sigmoid' or 'softmax' (network is configured with "
                f"'{self.output_activation}'). The gradient shortcut used for "
                f"cross-entropy is only mathematically valid for these two "
                f"pairings — using it with another activation would silently "
                f"produce incorrect gradients."
            )

        # Validate that we have expected outputs for each output neuron
        if len(expected_outputs) != len(self.output_layer):
            raise ValueError(f"Expected {len(self.output_layer)} output values, got {len(expected_outputs)}")
        
        # Calculate node value for each output neuron
        for neuron_idx, neuron in enumerate(self.output_layer):
            # Get the predicted output (activation) for this neuron. When softmax
            # is in use, neuron.last_output is the raw pre-softmax linear score,
            # so the actual predicted probability must come from the network's
            # post-softmax outputs instead.
            predicted_output = self.last_outputs[neuron_idx] if self.using_softmax else neuron.last_output

            # Get the expected output for this neuron
            expected_output = expected_outputs[neuron_idx]

            if self.cost_function == 'cross-entropy':
                # Calculate ∂Cost/∂activation using the cross-entropy cost function
                neuron.node_value = predicted_output - expected_output
            elif self.cost_function == 'mse':
                # Calculate ∂Cost/∂activation
                # For MSE cost = (1/2) * Σ(predicted - expected)²
                # The derivative is: (predicted - expected)
                cost_derivative = predicted_output - expected_output

                # Calculate ∂activation/∂(weighted_sum)
                # This is the derivative of the activation function
                activation_derivative = self.activation_derivative(
                    neuron.last_weighted_sum,
                    neuron.activation
                )

                # Calculate node value using chain rule
                # node_value = ∂Cost/∂activation × ∂activation/∂(weighted_sum)
                neuron.node_value = cost_derivative * activation_derivative
            else:
                print(f"Warning: Unknown cost function '{self.cost_function}'. Defaulting to MSE.")
                #Same as above
                cost_derivative = predicted_output - expected_output

                activation_derivative = self.activation_derivative(
                    neuron.last_weighted_sum,
                    neuron.activation
                )

                neuron.node_value = cost_derivative * activation_derivative

            # ============================================
            # UPDATE WEIGHTS AND BIAS FOR THIS OUTPUT NEURON
            # ============================================
            
            # Update each weight
            # Gradient for weight_i = node_value × input_i
            # The input that flows through weight_i is stored in last_inputs[i]
            for weight_idx in range(len(neuron.weights)):
                # Get the input value that this weight receives
                # For output layer, this is the activation from the last hidden layer
                input_value = neuron.last_inputs[weight_idx]
                
                # Calculate gradient: ∂Cost/∂weight = node_value × input_value
                neuron.weight_gradient_accumulations[weight_idx] += neuron.node_value * input_value

                if update_weights:

                    weight_gradient = max(min(neuron.weight_gradient_accumulations[weight_idx], weight_clip_value), -weight_clip_value)  # Clip gradients to prevent extreme updates
                    
                    # Update velocity using momentum and gradient descent
                    neuron.weight_velocities[weight_idx] = momentum * neuron.weight_velocities[weight_idx] - learning_rate * weight_gradient

                    # Update weight using velocity
                    neuron.weights[weight_idx] += neuron.weight_velocities[weight_idx]
            
            # Update bias
            # Gradient for bias = node_value (since bias input is always 1)
            neuron.bias_gradient_accumulation += neuron.node_value

            if update_weights:
                bias_gradient = max(min(neuron.bias_gradient_accumulation, bias_clip_value), -bias_clip_value)  # Clip gradients to prevent extreme updates

                # Update bias velocity using momentum and gradient descent
                neuron.bias_velocity = momentum * neuron.bias_velocity - learning_rate * bias_gradient
                
                # Update bias using gradient descent
                neuron.bias += neuron.bias_velocity

    def activation_derivative(self, weighted_sum, activation_function):
        """
        Calculate the derivative of an activation function at a given weighted sum.
        
        Args:
            weighted_sum (float): The weighted sum (z) at which to evaluate the derivative
            activation_function (str): The type of activation function
            
        Returns:
            float: The derivative value
        """
        if activation_function == 'sigmoid':
            # For sigmoid: σ'(z) = σ(z) × (1 - σ(z))
            # We can calculate this using the activation value
            activation = 1.0 / (1.0 + math.exp(-weighted_sum)) if -500 < weighted_sum < 500 else (1.0 if weighted_sum > 0 else 0.0)
            return activation * (1.0 - activation)
        
        elif activation_function == 'relu':
            # For ReLU: f'(z) = 1 if z > 0, else 0
            return 1.0 if weighted_sum > 0 else 0.0
        
        elif activation_function == 'leaky_relu':
            # For Leaky ReLU: f'(z) = 1 if z > 0, else 0.01
            return 1.0 if weighted_sum > 0 else 0.01
        
        elif activation_function == 'tanh':
            # For tanh: f'(z) = 1 - tanh²(z)
            tanh_value = math.tanh(weighted_sum)
            return 1.0 - tanh_value ** 2
        
        elif activation_function == 'linear':
            # For linear: f'(z) = 1
            return 1.0
        
        else:
            # Default to sigmoid derivative
            print(f"Warning: Unknown activation function '{activation_function}'. Using sigmoid derivative.")
            activation = 1.0 / (1.0 + math.exp(-weighted_sum)) if -500 < weighted_sum < 500 else (1.0 if weighted_sum > 0 else 0.0)
            return activation * (1.0 - activation)
        
    def compute_hidden_node_values(self):
        """
        Calculate node values for all hidden layer neurons using backpropagation.
        This method works backwards through the hidden layers, starting from the layer closest to the output and moving towards the input.
        
        For each hidden neuron, the node value is:
        node_value = activation_derivative(weighted_sum) × Σ(weight_to_next × node_value_next)

        Prerequisites:
        - forward() must have been called (so neurons have last_weighted_sum and last_inputs values)
        - backpropagate_output_layer() must have been called first

        Args:
            None (uses stored values in neurons)

        Returns:
            None (stores node values in each hidden neuron)
        """

        # Process hidden layers in reverse order (from output back to input)
        # Start from the last hidden layer and move backwards
        for layer_idx in range(len(self.hidden_layers) - 1, -1, -1):
            current_layer = self.hidden_layers[layer_idx]
            
            # Determine which layer comes after this one
            if layer_idx == len(self.hidden_layers) - 1:
                # This is the last hidden layer, so next layer is output layer
                next_layer = self.output_layer
            else:
                # Next layer is the following hidden layer
                next_layer = self.hidden_layers[layer_idx + 1]
            
            # Calculate node value for each neuron in this layer
            for neuron_idx, neuron in enumerate(current_layer):
                # Initialize node value to 0
                node_value = 0.0
                
                # Sum contributions from ALL neurons in the next layer
                for next_neuron_idx, next_neuron in enumerate(next_layer):
                    # Get the weight connecting this neuron to the next neuron
                    # The weight is stored in the next neuron's weights array
                    # at the index corresponding to this neuron's position
                    weight_to_next = next_neuron.weights[neuron_idx]
                    
                    # Get the node value of the next neuron (already calculated)
                    next_node_value = next_neuron.node_value
                    
                    # Add this contribution to the sum
                    node_value += weight_to_next * next_node_value
                
                # Multiply by the activation derivative for this neuron
                activation_deriv = self.activation_derivative(
                    neuron.last_weighted_sum,
                    neuron.activation
                )
                
                # Multiply by this neuron's dropout mask from the forward pass.
                # If the neuron was dropped (mask == 0.0), it contributed no
                # output, so it must also contribute no gradient. If it was
                # kept (mask == 1.0 / (1 - dropout_rate)), the same inverted-
                # dropout scaling used on the forward pass is applied here too,
                # keeping forward and backward passes consistent.
                node_value *= activation_deriv * neuron.last_dropout_mask
                
                # Store the calculated node value in the neuron
                neuron.node_value = node_value

    def backpropagate_hidden_layers(self, learning_rate=0.05, weight_clip_value=5.0, bias_clip_value=10.0, momentum=0.9, update_weights=True):
        """
        Calculate node values for all hidden layer neurons using backpropagation,
        then update weights and biases.
        
        This method works backwards through the hidden layers, starting from the layer
        closest to the output and moving towards the input.
        
        For each hidden neuron, the node value is:
        node_value = activation_derivative(weighted_sum) × Σ(weight_to_next × node_value_next)
        
        After calculating node values, weights are updated using:
        new_weight = old_weight - learning_rate × node_value × input_to_weight
        
        Where input_to_weight is the activation value that flows through that weight
        (stored in neuron.last_inputs).
        
        Prerequisites:
        - forward() must have been called (so neurons have last_weighted_sum and last_inputs values)
        - backpropagate_output_layer() must have been called first
        
        Args:
            learning_rate (float): How much to adjust weights and biases
        
        Returns:
            None (stores node values and updates weights/biases directly in each hidden neuron)
        """
       
        # Process hidden layers in reverse order (from output back to input)
        # Start from the last hidden layer and move backwards
        for layer_idx in range(len(self.hidden_layers) - 1, -1, -1):
            current_layer = self.hidden_layers[layer_idx]
            
            # Determine which layer comes after this one
            if layer_idx == len(self.hidden_layers) - 1:
                # This is the last hidden layer, so next layer is output layer
                next_layer = self.output_layer
            else:
                # Next layer is the following hidden layer
                next_layer = self.hidden_layers[layer_idx + 1]
            
            # Calculate node value for each neuron in this layer
            for neuron_idx, neuron in enumerate(current_layer):
                # Initialize node value to 0
                node_value = 0.0
                
                # Sum contributions from ALL neurons in the next layer
                for next_neuron_idx, next_neuron in enumerate(next_layer):
                    # Get the weight connecting this neuron to the next neuron
                    # The weight is stored in the next neuron's weights array
                    # at the index corresponding to this neuron's position
                    weight_to_next = next_neuron.weights[neuron_idx]
                    
                    # Get the node value of the next neuron (already calculated)
                    next_node_value = next_neuron.node_value
                    
                    # Add this contribution to the sum
                    node_value += weight_to_next * next_node_value
                
                # Multiply by the activation derivative for this neuron
                activation_deriv = self.activation_derivative(
                    neuron.last_weighted_sum,
                    neuron.activation
                )
                
                # Multiply by this neuron's dropout mask from the forward pass.
                # If the neuron was dropped (mask == 0.0), it contributed no
                # output, so it must also contribute no gradient. If it was
                # kept (mask == 1.0 / (1 - dropout_rate)), the same inverted-
                # dropout scaling used on the forward pass is applied here too,
                # keeping forward and backward passes consistent.
                node_value *= activation_deriv * neuron.last_dropout_mask
                
                # Store the calculated node value in the neuron
                neuron.node_value = node_value

                # ============================================
                # UPDATE WEIGHTS AND BIAS FOR THIS NEURON
                # ============================================
                
                # Update each weight
                # Gradient for weight_i = node_value × input_i
                # The input that flows through weight_i is stored in last_inputs[i]
                for weight_idx in range(len(neuron.weights)):
                    # Get the input value that this weight receives
                    input_value = neuron.last_inputs[weight_idx]
                    
                    # Calculate gradient: ∂Cost/∂weight = node_value × input_value
                    neuron.weight_gradient_accumulations[weight_idx] += neuron.node_value * input_value
                    
                    if update_weights:
                        weight_gradient = max(min(neuron.weight_gradient_accumulations[weight_idx], weight_clip_value), -weight_clip_value)  # Clip gradients to prevent extreme updates

                        # Update velocity using momentum and gradient descent
                        neuron.weight_velocities[weight_idx] = momentum * neuron.weight_velocities[weight_idx] - learning_rate * weight_gradient

                        # Update weight using velocity
                        neuron.weights[weight_idx] += neuron.weight_velocities[weight_idx]
                
                # Update bias
                # Gradient for bias = node_value × 1 (since bias input is always 1)
                # So bias_gradient = node_value
                neuron.bias_gradient_accumulation += neuron.node_value
                if update_weights:
                    bias_gradient = max(min(neuron.bias_gradient_accumulation, bias_clip_value), -bias_clip_value)  # Clip gradients to prevent extreme updates

                    # Update bias velocity using momentum and gradient descent
                    neuron.bias_velocity = momentum * neuron.bias_velocity - learning_rate * bias_gradient
                    
                    # Update bias using gradient descent
                    neuron.bias += neuron.bias_velocity

    def accumulate_gradients(self):
        """
        Accumulate gradients for all neurons in the network after backpropagation.
        
        This method should be called after compute_output_node_values() and compute_hidden_node_values() to accumulate the gradients for all neurons based on their node values and inputs.
        Args:
            None (uses stored values in neurons)
        Returns:
            None (stores accumulated gradients in each neuron)
        """
        for neuron in self.output_layer:
            for weight_idx in range(len(neuron.weights)):
                input_value = neuron.last_inputs[weight_idx]
                neuron.weight_gradient_accumulations[weight_idx] += neuron.node_value * input_value
            neuron.bias_gradient_accumulation += neuron.node_value
        for layer in self.hidden_layers:
            for neuron in layer:
                for weight_idx in range(len(neuron.weights)):
                    input_value = neuron.last_inputs[weight_idx]
                    neuron.weight_gradient_accumulations[weight_idx] += neuron.node_value * input_value
                neuron.bias_gradient_accumulation += neuron.node_value
    
    def update_weights_sgd(self, learning_rate=0.05, weight_clip_value=5.0, bias_clip_value=10.0, momentum=0.9):
        """
        Updates the weights and biases of all neurons in the network using
        Stochastic Gradient Descent (SGD). This function directly applies the
        gradient stored in each neuron to its weights and biases, scaled by the
        learning rate. This is the simplest weight update method and is best
        suited for single-sample training without mini-batches.

        This function should be called after compute_output_gradients() and
        compute_hidden_gradients() have been called to ensure that each neuron
        has a valid node value before updating.

        Args:
            learning_rate (float): The scaling factor for the gradient update.
                                Smaller values (e.g. 0.001) make the network
                                learn more slowly but more reliably. Larger
                                values (e.g. 0.01) learn faster but risk
                                overshooting.
            clip (float or None):  If provided, gradients are clamped to the
                                range [-clip, clip] before being applied.
                                Helps prevent exploding gradients. If None,
                                no clipping is applied.

        Returns:
            None: Weights and biases are updated directly inside each neuron.
        """
        
        for neuron in self.output_layer:
            # Update each weight in output layer
            for weight_idx in range(len(neuron.weights)):
                input_value = neuron.last_inputs[weight_idx]
                weight_gradient = neuron.node_value * input_value
                weight_gradient = max(min(weight_gradient, weight_clip_value), -weight_clip_value) if weight_clip_value is not None else weight_gradient  # Clip gradients if clip value is provided
                neuron.weight_velocities[weight_idx] = momentum * neuron.weight_velocities[weight_idx] - learning_rate * weight_gradient
                neuron.weights[weight_idx] += neuron.weight_velocities[weight_idx]

            # Update bias in output layer
            bias_gradient = neuron.node_value
            bias_gradient = max(min(bias_gradient, bias_clip_value), -bias_clip_value) if bias_clip_value is not None else bias_gradient  # Clip gradients if clip value is provided
            neuron.bias_velocity = momentum * neuron.bias_velocity - learning_rate * bias_gradient
            neuron.bias += neuron.bias_velocity

        for layer in self.hidden_layers:
            for neuron in layer:
                # Update each weight in hidden layers
                for weight_idx in range(len(neuron.weights)):
                    input_value = neuron.last_inputs[weight_idx]
                    weight_gradient = neuron.node_value * input_value
                    weight_gradient = max(min(weight_gradient, weight_clip_value), -weight_clip_value) if weight_clip_value is not None else weight_gradient  # Clip gradients if clip value is provided
                    neuron.weight_velocities[weight_idx] = momentum * neuron.weight_velocities[weight_idx] - learning_rate * weight_gradient
                    neuron.weights[weight_idx] += neuron.weight_velocities[weight_idx]

                # Update bias in hidden layers
                bias_gradient = neuron.node_value
                bias_gradient = max(min(bias_gradient, bias_clip_value), -bias_clip_value) if bias_clip_value is not None else bias_gradient  # Clip gradients if clip value is provided
                neuron.bias_velocity = momentum * neuron.bias_velocity - learning_rate * bias_gradient
                neuron.bias += neuron.bias_velocity
    
    def apply_gradient_accumulations_sgd(self, learning_rate=0.05, batch_size=1, weight_clip_value=5.0, bias_clip_value=10.0, momentum=0.9):
        """
        Updates the weights and biases of all neurons in the network using the
        average of the gradients accumulated over a mini-batch, via Stochastic
        Gradient Descent (SGD). Rather than updating weights after every single
        sample, mini-batch training accumulates gradients across multiple samples
        and averages them before applying the update. This produces more stable
        gradient estimates and generally leads to better training.

        This function should be called after accumulate_gradients() has been
        called once for each sample in the mini-batch. After calling this
        function, reset_accumulated_gradients() should be called before
        processing the next mini-batch.

        Args:
            learning_rate (float): The scaling factor for the gradient update.
                                Smaller values (e.g. 0.001) make the network
                                learn more slowly but more reliably. Larger
                                values (e.g. 0.01) learn faster but risk
                                overshooting.
            batch_size (int):      The number of samples in the mini-batch. Used
                                to average the accumulated gradients before
                                applying the update.
            clip (float or None):  If provided, gradients are clamped to the
                                range [-clip, clip] before being applied.
                                Helps prevent exploding gradients. If None,
                                no clipping is applied.

        Returns:
            None: Weights and biases are updated directly inside each neuron.
                Accumulated gradients are not reset by this function and must
                be reset manually by calling reset_accumulated_gradients().
        """
        for neuron in self.output_layer:
            # Update each weight in output layer
            for weight_idx in range(len(neuron.weights)):
                avg_weight_gradient = neuron.weight_gradient_accumulations[weight_idx] / batch_size
                avg_weight_gradient = max(min(avg_weight_gradient, weight_clip_value), -weight_clip_value) if weight_clip_value is not None else avg_weight_gradient  # Clip gradients if clip value is provided
                neuron.weight_velocities[weight_idx] = momentum * neuron.weight_velocities[weight_idx] - learning_rate * avg_weight_gradient
                neuron.weights[weight_idx] += neuron.weight_velocities[weight_idx]

            # Update bias in output layer
            avg_bias_gradient = neuron.bias_gradient_accumulation / batch_size
            avg_bias_gradient = max(min(avg_bias_gradient, bias_clip_value), -bias_clip_value) if bias_clip_value is not None else avg_bias_gradient  # Clip gradients if clip value is provided
            neuron.bias_velocity = momentum * neuron.bias_velocity - learning_rate * avg_bias_gradient
            neuron.bias += neuron.bias_velocity

        for layer in self.hidden_layers:
            for neuron in layer:
                # Update each weight in hidden layers
                for weight_idx in range(len(neuron.weights)):
                    avg_weight_gradient = neuron.weight_gradient_accumulations[weight_idx] / batch_size
                    avg_weight_gradient = max(min(avg_weight_gradient, weight_clip_value), -weight_clip_value) if weight_clip_value is not None else avg_weight_gradient  # Clip gradients if clip value is provided
                    neuron.weight_velocities[weight_idx] = momentum * neuron.weight_velocities[weight_idx] - learning_rate * avg_weight_gradient
                    neuron.weights[weight_idx] += neuron.weight_velocities[weight_idx]

                # Update bias in hidden layers
                avg_bias_gradient = neuron.bias_gradient_accumulation / batch_size
                avg_bias_gradient = max(min(avg_bias_gradient, bias_clip_value), -bias_clip_value) if bias_clip_value is not None else avg_bias_gradient  # Clip gradients if clip value is provided
                neuron.bias_velocity = momentum * neuron.bias_velocity - learning_rate * avg_bias_gradient
                neuron.bias += neuron.bias_velocity

    def update_weights_adam(self, learning_rate=0.001, weight_clip_value=5.0, bias_clip_value=10.0, momentum=0.9, squared_gradient_term=0.999):
        """
        Updates the weights and biases of all neurons in the network using the
        Adam optimization algorithm. Adam combines momentum and adaptive learning
        rates to provide efficient and effective training.

        This function should be called after compute_output_gradients() and
        compute_hidden_gradients() have been called to ensure that each neuron
        has a valid node value before updating.

        Args:
            learning_rate (float): The base learning rate for Adam updates.
            weight_clip_value (float or None): If provided, weight gradients are
                clamped to the range [-weight_clip_value, weight_clip_value]
                before being applied. Helps prevent exploding gradients. If None,
                no clipping is applied."""
        for neuron in self.output_layer:
            for weight_idx in range(len(neuron.weights)):
                input_value = neuron.last_inputs[weight_idx]
                weight_gradient = neuron.node_value * input_value
                weight_gradient = max(min(weight_gradient, weight_clip_value), -weight_clip_value) if weight_clip_value is not None else weight_gradient  # Clip gradients if clip value is provided

                # Update first moment (velocity)
                neuron.weight_velocities[weight_idx] = momentum * neuron.weight_velocities[weight_idx] + (1 - momentum) * weight_gradient

                # Update second moment (squared gradient)
                neuron.weight_squared_gradient_accumulations[weight_idx] = squared_gradient_term * neuron.weight_squared_gradient_accumulations[weight_idx] + (1 - squared_gradient_term) * (weight_gradient ** 2)
                # Compute bias-corrected moments
                velocity_corrected = neuron.weight_velocities[weight_idx] / (1 - momentum)
                squared_gradient_corrected = neuron.weight_squared_gradient_accumulations[weight_idx] / (1 - squared_gradient_term)
                # Update weight using Adam update rule
                neuron.weights[weight_idx] -= learning_rate * velocity_corrected / (math.sqrt(squared_gradient_corrected) + 1e-8)
            # Update bias in output layer
            bias_gradient = neuron.node_value
            bias_gradient = max(min(bias_gradient, bias_clip_value), -bias_clip_value) if bias_clip_value is not None else bias_gradient  # Clip gradients if clip value is provided
            neuron.bias_velocity = momentum * neuron.bias_velocity + (1 - momentum) * bias_gradient
            neuron.bias_squared_gradient_accumulation = squared_gradient_term * neuron.bias_squared_gradient_accumulation + (1 - squared_gradient_term) * (bias_gradient ** 2)
            bias_velocity_corrected = neuron.bias_velocity / (1 - momentum)
            bias_squared_gradient_corrected = neuron.bias_squared_gradient_accumulation / (1 - squared_gradient_term)
            neuron.bias -= learning_rate * bias_velocity_corrected / (math.sqrt(bias_squared_gradient_corrected) + 1e-8)

        for layer in self.hidden_layers:
            for neuron in layer: 
                for weight_idx in range(len(neuron.weights)):
                    input_value = neuron.last_inputs[weight_idx]
                    weight_gradient = neuron.node_value * input_value
                    weight_gradient = max(min(weight_gradient, weight_clip_value), -weight_clip_value) if weight_clip_value is not None else weight_gradient  # Clip gradients if clip value is provided

                    # Update first moment (velocity)
                    neuron.weight_velocities[weight_idx] = momentum * neuron.weight_velocities[weight_idx] + (1 - momentum) * weight_gradient

                    # Update second moment (squared gradient)
                    neuron.weight_squared_gradient_accumulations[weight_idx] = squared_gradient_term * neuron.weight_squared_gradient_accumulations[weight_idx] + (1 - squared_gradient_term) * (weight_gradient ** 2)
                    # Compute bias-corrected moments
                    velocity_corrected = neuron.weight_velocities[weight_idx] / (1 - momentum)
                    squared_gradient_corrected = neuron.weight_squared_gradient_accumulations[weight_idx] / (1 - squared_gradient_term)

                    # Update weight using Adam update rule
                    neuron.weights[weight_idx] -= learning_rate * velocity_corrected / (math.sqrt(squared_gradient_corrected) + 1e-8)

                # Update bias in hidden layers
                bias_gradient = neuron.node_value
                bias_gradient = max(min(bias_gradient, bias_clip_value), -bias_clip_value) if bias_clip_value is not None else bias_gradient  # Clip gradients if clip value is provided
                neuron.bias_velocity = momentum * neuron.bias_velocity + (1 - momentum) * bias_gradient
                neuron.bias_squared_gradient_accumulation = squared_gradient_term * neuron.bias_squared_gradient_accumulation + (1 - squared_gradient_term) * (bias_gradient ** 2)
                bias_velocity_corrected = neuron.bias_velocity / (1 - momentum)
                bias_squared_gradient_corrected = neuron.bias_squared_gradient_accumulation / (1 - squared_gradient_term)
                neuron.bias -= learning_rate * bias_velocity_corrected / (math.sqrt(bias_squared_gradient_corrected) + 1e-8)

    def update_weights_adagrad(self, learning_rate=0.01, weight_clip_value=5.0, bias_clip_value=10.0):
        """
        Updates the weights and biases of all neurons in the network using the
        AdaGrad optimization algorithm. AdaGrad adapts the learning rate per
        parameter by accumulating a running sum of squared gradients. Parameters
        that receive frequent large gradients get a smaller effective learning
        rate; rarely updated parameters get a larger one.

        This function should be called after compute_output_gradients() and
        compute_hidden_gradients() have been called to ensure that each neuron
        has a valid node value before updating.

        Args:
            learning_rate (float): The base learning rate. Default: 0.01.
            weight_clip_value (float or None): If provided, weight gradients are
                clamped to [-weight_clip_value, weight_clip_value]. If None, no
                clipping is applied.
            bias_clip_value (float or None): If provided, bias gradients are
                clamped to [-bias_clip_value, bias_clip_value]. If None, no
                clipping is applied.

        Returns:
            None: Weights and biases are updated directly inside each neuron.
        """
        for neuron in self.output_layer:
            for weight_idx in range(len(neuron.weights)):
                input_value = neuron.last_inputs[weight_idx]
                weight_gradient = neuron.node_value * input_value
                weight_gradient = max(min(weight_gradient, weight_clip_value), -weight_clip_value) if weight_clip_value is not None else weight_gradient

                neuron.weight_squared_gradient_accumulations[weight_idx] += weight_gradient ** 2
                neuron.weights[weight_idx] -= learning_rate / (math.sqrt(neuron.weight_squared_gradient_accumulations[weight_idx]) + 1e-8) * weight_gradient

            bias_gradient = neuron.node_value
            bias_gradient = max(min(bias_gradient, bias_clip_value), -bias_clip_value) if bias_clip_value is not None else bias_gradient
            neuron.bias_squared_gradient_accumulation += bias_gradient ** 2
            neuron.bias -= learning_rate / (math.sqrt(neuron.bias_squared_gradient_accumulation) + 1e-8) * bias_gradient

        for layer in self.hidden_layers:
            for neuron in layer:
                for weight_idx in range(len(neuron.weights)):
                    input_value = neuron.last_inputs[weight_idx]
                    weight_gradient = neuron.node_value * input_value
                    weight_gradient = max(min(weight_gradient, weight_clip_value), -weight_clip_value) if weight_clip_value is not None else weight_gradient

                    neuron.weight_squared_gradient_accumulations[weight_idx] += weight_gradient ** 2
                    neuron.weights[weight_idx] -= learning_rate / (math.sqrt(neuron.weight_squared_gradient_accumulations[weight_idx]) + 1e-8) * weight_gradient

                bias_gradient = neuron.node_value
                bias_gradient = max(min(bias_gradient, bias_clip_value), -bias_clip_value) if bias_clip_value is not None else bias_gradient
                neuron.bias_squared_gradient_accumulation += bias_gradient ** 2
                neuron.bias -= learning_rate / (math.sqrt(neuron.bias_squared_gradient_accumulation) + 1e-8) * bias_gradient

    def apply_gradient_accumulations_adam(self, learning_rate=0.001, batch_size=1, weight_clip_value=5.0, bias_clip_value=10.0, momentum=0.9, squared_gradient_term=0.999):
        """
        Updates the weights and biases of all neurons in the network using the
        average of the gradients accumulated over a mini-batch, via the Adam
        optimization algorithm. Rather than updating weights after every single
        sample, mini-batch training accumulates gradients across multiple samples
        and averages them before applying the update. This produces more stable
        gradient estimates and generally leads to better training.

        This function should be called after accumulate_gradients() has been
        called once for each sample in the mini-batch. After calling this
        function, reset_accumulated_gradients() should be called before
        processing the next mini-batch.

        Args:
            learning_rate (float): The base learning rate for Adam updates.
            batch_size (int): The number of samples in the mini-batch. Used to
                average the accumulated gradients before applying the update.
            weight_clip_value (float or None): If provided, weight gradients are
                clamped to the range [-weight_clip_value, weight_clip_value]
                before being applied. Helps prevent exploding gradients. If None,
                no clipping is applied.
            bias_clip_value (float or None): If provided, bias gradients are
                clamped to the range [-bias_clip_value, bias_clip_value] before
                being applied. Helps prevent exploding gradients. If None, no
                clipping is applied.
        """

        for neuron in self.output_layer:
            for weight_idx in range(len(neuron.weights)):
                avg_weight_gradient = neuron.weight_gradient_accumulations[weight_idx] / batch_size
                avg_weight_gradient = max(min(avg_weight_gradient, weight_clip_value), -weight_clip_value) if weight_clip_value is not None else avg_weight_gradient  # Clip gradients if clip value is provided

                # Update first moment (velocity)
                neuron.weight_velocities[weight_idx] = momentum * neuron.weight_velocities[weight_idx] + (1 - momentum) * avg_weight_gradient

                # Update second moment (squared gradient)
                neuron.weight_squared_gradient_accumulations[weight_idx] = squared_gradient_term * neuron.weight_squared_gradient_accumulations[weight_idx] + (1 - squared_gradient_term) * (avg_weight_gradient ** 2)

                # Compute bias-corrected moments
                velocity_corrected = neuron.weight_velocities[weight_idx] / (1 - momentum)
                squared_gradient_corrected = neuron.weight_squared_gradient_accumulations[weight_idx] / (1 - squared_gradient_term)

                # Update weight using Adam update rule
                neuron.weights[weight_idx] -= learning_rate * velocity_corrected / (math.sqrt(squared_gradient_corrected) + 1e-8)

            # Update bias in output layer
            avg_bias_gradient = neuron.bias_gradient_accumulation / batch_size
            avg_bias_gradient = max(min(avg_bias_gradient, bias_clip_value), -bias_clip_value) if bias_clip_value is not None else avg_bias_gradient  # Clip gradients if clip value is provided
            neuron.bias_velocity = momentum * neuron.bias_velocity + (1 - momentum) * avg_bias_gradient
            neuron.bias_squared_gradient_accumulation = squared_gradient_term * neuron.bias_squared_gradient_accumulation + (1 - squared_gradient_term) * (avg_bias_gradient ** 2)
            bias_velocity_corrected = neuron.bias_velocity / (1 - momentum)
            bias_squared_gradient_corrected = neuron.bias_squared_gradient_accumulation / (1 - squared_gradient_term)
            neuron.bias -= learning_rate * bias_velocity_corrected / (math.sqrt(bias_squared_gradient_corrected) + 1e-8)

        for layer in self.hidden_layers:
            for neuron in layer:
                for weight_idx in range(len(neuron.weights)):
                    avg_weight_gradient = neuron.weight_gradient_accumulations[weight_idx] / batch_size
                    avg_weight_gradient = max(min(avg_weight_gradient, weight_clip_value), -weight_clip_value) if weight_clip_value is not None else avg_weight_gradient  # Clip gradients if clip value is provided

                    # Update first moment (velocity)
                    neuron.weight_velocities[weight_idx] = momentum * neuron.weight_velocities[weight_idx] + (1 - momentum) * avg_weight_gradient

                    # Update second moment (squared gradient)
                    neuron.weight_squared_gradient_accumulations[weight_idx] = squared_gradient_term * neuron.weight_squared_gradient_accumulations[weight_idx] + (1 - squared_gradient_term) * (avg_weight_gradient ** 2)

                    # Compute bias-corrected moments
                    velocity_corrected = neuron.weight_velocities[weight_idx] / (1 - momentum)
                    squared_gradient_corrected = neuron.weight_squared_gradient_accumulations[weight_idx] / (1 - squared_gradient_term)

                    # Update weight using Adam update rule
                    neuron.weights[weight_idx] -= learning_rate * velocity_corrected / (math.sqrt(squared_gradient_corrected) + 1e-8)

                # Update bias in hidden layers
                avg_bias_gradient = neuron.bias_gradient_accumulation / batch_size
                avg_bias_gradient = max(min(avg_bias_gradient, bias_clip_value), -bias_clip_value) if bias_clip_value is not None else avg_bias_gradient  # Clip gradients if clip value is provided
                neuron.bias_velocity = momentum * neuron.bias_velocity + (1 - momentum) * avg_bias_gradient
                neuron.bias_squared_gradient_accumulation = squared_gradient_term * neuron.bias_squared_gradient_accumulation + (1 - squared_gradient_term) * (avg_bias_gradient ** 2)
                bias_velocity_corrected = neuron.bias_velocity / (1 - momentum)
                bias_squared_gradient_corrected = neuron.bias_squared_gradient_accumulation / (1 - squared_gradient_term)
                neuron.bias -= learning_rate * bias_velocity_corrected / (math.sqrt(bias_squared_gradient_corrected) + 1e-8)

    def apply_gradient_accumulations_adagrad(self, learning_rate=0.01, batch_size=1, weight_clip_value=5.0, bias_clip_value=10.0):
        """
        Updates the weights and biases of all neurons in the network using the
        average of the gradients accumulated over a mini-batch, via the AdaGrad
        optimization algorithm. Rather than updating weights after every single
        sample, mini-batch training accumulates gradients across multiple samples
        and averages them before applying the update.

        This function should be called after accumulate_gradients() has been
        called once for each sample in the mini-batch. After calling this
        function, reset_accumulated_gradients() should be called before
        processing the next mini-batch.

        Args:
            learning_rate (float): The base learning rate. Default: 0.01.
            batch_size (int): The number of samples in the mini-batch. Used to
                average the accumulated gradients before applying the update.
            weight_clip_value (float or None): If provided, weight gradients are
                clamped to [-weight_clip_value, weight_clip_value]. If None, no
                clipping is applied.
            bias_clip_value (float or None): If provided, bias gradients are
                clamped to [-bias_clip_value, bias_clip_value]. If None, no
                clipping is applied.

        Returns:
            None: Weights and biases are updated directly inside each neuron.
                Accumulated gradients are not reset by this function and must
                be reset manually by calling reset_accumulated_gradients().
        """
        for neuron in self.output_layer:
            for weight_idx in range(len(neuron.weights)):
                avg_weight_gradient = neuron.weight_gradient_accumulations[weight_idx] / batch_size
                avg_weight_gradient = max(min(avg_weight_gradient, weight_clip_value), -weight_clip_value) if weight_clip_value is not None else avg_weight_gradient

                neuron.weight_squared_gradient_accumulations[weight_idx] += avg_weight_gradient ** 2
                neuron.weights[weight_idx] -= learning_rate / (math.sqrt(neuron.weight_squared_gradient_accumulations[weight_idx]) + 1e-8) * avg_weight_gradient

            avg_bias_gradient = neuron.bias_gradient_accumulation / batch_size
            avg_bias_gradient = max(min(avg_bias_gradient, bias_clip_value), -bias_clip_value) if bias_clip_value is not None else avg_bias_gradient
            neuron.bias_squared_gradient_accumulation += avg_bias_gradient ** 2
            neuron.bias -= learning_rate / (math.sqrt(neuron.bias_squared_gradient_accumulation) + 1e-8) * avg_bias_gradient

        for layer in self.hidden_layers:
            for neuron in layer:
                for weight_idx in range(len(neuron.weights)):
                    avg_weight_gradient = neuron.weight_gradient_accumulations[weight_idx] / batch_size
                    avg_weight_gradient = max(min(avg_weight_gradient, weight_clip_value), -weight_clip_value) if weight_clip_value is not None else avg_weight_gradient

                    neuron.weight_squared_gradient_accumulations[weight_idx] += avg_weight_gradient ** 2
                    neuron.weights[weight_idx] -= learning_rate / (math.sqrt(neuron.weight_squared_gradient_accumulations[weight_idx]) + 1e-8) * avg_weight_gradient

                avg_bias_gradient = neuron.bias_gradient_accumulation / batch_size
                avg_bias_gradient = max(min(avg_bias_gradient, bias_clip_value), -bias_clip_value) if bias_clip_value is not None else avg_bias_gradient
                neuron.bias_squared_gradient_accumulation += avg_bias_gradient ** 2
                neuron.bias -= learning_rate / (math.sqrt(neuron.bias_squared_gradient_accumulation) + 1e-8) * avg_bias_gradient

    def apply_gradient_accumulations_RMSprop(self, learning_rate=0.001, batch_size=1, weight_clip_value=5.0, bias_clip_value=10.0, squared_gradient_term=0.999):
        """
        Updates the weights and biases of all neurons in the network using the
        average of the gradients accumulated over a mini-batch, via the RMSprop
        optimization algorithm. RMSprop divides the learning rate by a running
        average of the magnitude of recent gradients for each parameter, which
        helps stabilize updates when gradients vary widely in scale.

        This function should be called after accumulate_gradients() has been
        called once for each sample in the mini-batch. After calling this
        function, reset_accumulated_gradients() should be called before
        processing the next mini-batch.

        Args:
            learning_rate (float): The base learning rate. Default: 0.001.
            batch_size (int): The number of samples in the mini-batch. Used to
                average the accumulated gradients before applying the update.
            weight_clip_value (float or None): If provided, weight gradients are
                clamped to [-weight_clip_value, weight_clip_value]. If None, no
                clipping is applied.
            bias_clip_value (float or None): If provided, bias gradients are
                clamped to [-bias_clip_value, bias_clip_value]. If None, no
                clipping is applied.
            squared_gradient_term (float): Controls the decay rate of the running
                average of squared gradients. Default: 0.999.

        Returns:
            None: Weights and biases are updated directly inside each neuron.
                Accumulated gradients are not reset by this function and must
                be reset manually by calling reset_accumulated_gradients().
        """
        for neuron in self.output_layer:
            for weight_idx in range(len(neuron.weights)):
                avg_weight_gradient = neuron.weight_gradient_accumulations[weight_idx] / batch_size
                avg_weight_gradient = max(min(avg_weight_gradient, weight_clip_value), -weight_clip_value) if weight_clip_value is not None else avg_weight_gradient

                neuron.weight_squared_gradient_accumulations[weight_idx] = squared_gradient_term * neuron.weight_squared_gradient_accumulations[weight_idx] + (1 - squared_gradient_term) * (avg_weight_gradient ** 2)
                neuron.weights[weight_idx] -= learning_rate * avg_weight_gradient / (math.sqrt(neuron.weight_squared_gradient_accumulations[weight_idx]) + 1e-8)

            avg_bias_gradient = neuron.bias_gradient_accumulation / batch_size
            avg_bias_gradient = max(min(avg_bias_gradient, bias_clip_value), -bias_clip_value) if bias_clip_value is not None else avg_bias_gradient
            neuron.bias_squared_gradient_accumulation = squared_gradient_term * neuron.bias_squared_gradient_accumulation + (1 - squared_gradient_term) * (avg_bias_gradient ** 2)
            neuron.bias -= learning_rate * avg_bias_gradient / (math.sqrt(neuron.bias_squared_gradient_accumulation) + 1e-8)

        for layer in self.hidden_layers:
            for neuron in layer:
                for weight_idx in range(len(neuron.weights)):
                    avg_weight_gradient = neuron.weight_gradient_accumulations[weight_idx] / batch_size
                    avg_weight_gradient = max(min(avg_weight_gradient, weight_clip_value), -weight_clip_value) if weight_clip_value is not None else avg_weight_gradient

                    neuron.weight_squared_gradient_accumulations[weight_idx] = squared_gradient_term * neuron.weight_squared_gradient_accumulations[weight_idx] + (1 - squared_gradient_term) * (avg_weight_gradient ** 2)
                    neuron.weights[weight_idx] -= learning_rate * avg_weight_gradient / (math.sqrt(neuron.weight_squared_gradient_accumulations[weight_idx]) + 1e-8)

                avg_bias_gradient = neuron.bias_gradient_accumulation / batch_size
                avg_bias_gradient = max(min(avg_bias_gradient, bias_clip_value), -bias_clip_value) if bias_clip_value is not None else avg_bias_gradient
                neuron.bias_squared_gradient_accumulation = squared_gradient_term * neuron.bias_squared_gradient_accumulation + (1 - squared_gradient_term) * (avg_bias_gradient ** 2)
                neuron.bias -= learning_rate * avg_bias_gradient / (math.sqrt(neuron.bias_squared_gradient_accumulation) + 1e-8)

    def reset_accumulated_gradients(self):
        """
        Resets the accumulated gradients for all neurons in the network to zero.
        This should be called after applying the accumulated gradients to update
        weights and biases, and before starting to accumulate gradients for the
        next mini-batch.

        Args:
            None (operates directly on neurons in the network)

        Returns:
            None (resets gradient accumulations in place)
        """
        for neuron in self.output_layer:
            neuron.weight_gradient_accumulations = [0.0 for _ in neuron.weights]
            neuron.bias_gradient_accumulation = 0.0
        for layer in self.hidden_layers:
            for neuron in layer:
                neuron.weight_gradient_accumulations = [0.0 for _ in neuron.weights]
                neuron.bias_gradient_accumulation = 0.0

    def train(self, data, epochs=1, optimizer='adam', initial_learning_rate=0.001,
          learning_rate_decay=1.0, batch_size=32, dropout_rate=0.0,
          weight_clip_value=5.0, bias_clip_value=10.0, momentum=0.9,
          squared_gradient_term=0.999, print_rate=1000):
        """
        Train the network on the given data using the specified optimizer.

        This is the main training entry point. It handles shuffling, batching,
        gradient accumulation, weight updates, and progress printing for each
        epoch. Both SGD and Adam use the same training loop structure, differing
        only in how weights are updated.

        Args:
            data (list):                  List of (inputs, expected_outputs) tuples.
            epochs (int):                 Number of full passes through the dataset.
                                        Default: 1.
            optimizer (str):              Which optimizer to use. Options are 'sgd',
                                        'adam', 'adagrad', and 'rmsprop'. Default: 'adam'.
            initial_learning_rate (float):Starting learning rate. Decayed each epoch
                                        by learning_rate_decay. Default: 0.001.
            learning_rate_decay (float):  Multiplicative decay factor applied to the
                                        learning rate after each epoch. 1.0 means
                                        no decay, 0.95 means 5% decay per epoch.
                                        Default: 1.0.
            batch_size (int):             Number of samples to accumulate gradients
                                        over before applying a weight update.
                                        Default: 32.
            dropout_rate (float):         Fraction of hidden neurons to randomly
                                        disable during each forward pass. 0.0
                                        disables dropout. Default: 0.0.
            weight_clip_value (float):    Gradients for weights are clamped to
                                        [-weight_clip_value, weight_clip_value]
                                        before being applied. Default: 5.0.
            bias_clip_value (float):      Gradients for biases are clamped to
                                        [-bias_clip_value, bias_clip_value]
                                        before being applied. Default: 10.0.
            momentum (float):             Momentum term for both SGD and Adam.
                                        For SGD this controls the velocity decay.
                                        For Adam this is the β1 parameter.
                                        Default: 0.9.
            squared_gradient_term (float):Adam and RMSprop only. Controls the decay
                                        rate of the second moment / squared gradient
                                        estimate (Adam's β2 parameter). Ignored when
                                        optimizer is 'sgd' or 'adagrad'. Default: 0.999.
            print_rate (int):             How often to print training progress, in
                                        number of samples. Default: 1000.

        Returns:
            None: Weights and biases are updated in place.

        Raises:
            ValueError: If optimizer is not 'sgd' or 'adam'.
        """
        if optimizer not in ('sgd', 'adam', 'adagrad', 'rmsprop'):
            raise ValueError(f"Unknown optimizer '{optimizer}'. Choose 'sgd', 'adam', 'adagrad', or 'rmsprop'.")

        for epoch in range(epochs):
            lr = initial_learning_rate * (learning_rate_decay ** epoch)
            local_accuracy_sum = 0.0
            local_loss_sum = 0.0

            random.shuffle(data)
            self.reset_accumulated_gradients()

            for idx, (inputs, expected_outputs) in enumerate(data):
                # Forward pass
                self.forward(inputs, dropout_rate=dropout_rate)

                # Compute gradients
                self.compute_output_node_values(expected_outputs)
                self.compute_hidden_node_values()
                self.accumulate_gradients()

                # Apply weight update at end of each batch
                if (idx + 1) % batch_size == 0:
                    if optimizer == 'adam':
                        self.apply_gradient_accumulations_adam(
                            learning_rate=lr,
                            batch_size=batch_size,
                            weight_clip_value=weight_clip_value,
                            bias_clip_value=bias_clip_value,
                            momentum=momentum,
                            squared_gradient_term=squared_gradient_term
                        )
                    elif optimizer == 'sgd':
                        self.apply_gradient_accumulations_sgd(
                            learning_rate=lr,
                            batch_size=batch_size,
                            weight_clip_value=weight_clip_value,
                            bias_clip_value=bias_clip_value,
                            momentum=momentum
                        )
                    elif optimizer == 'adagrad':
                        self.apply_gradient_accumulations_adagrad(
                            learning_rate=lr,
                            batch_size=batch_size,
                            weight_clip_value=weight_clip_value,
                            bias_clip_value=bias_clip_value
                        )
                    elif optimizer == 'rmsprop':
                        self.apply_gradient_accumulations_RMSprop(
                            learning_rate=lr,
                            batch_size=batch_size,
                            weight_clip_value=weight_clip_value,
                            bias_clip_value=bias_clip_value,
                            squared_gradient_term=squared_gradient_term
                        )
                    self.reset_accumulated_gradients()

                # Track accuracy and loss
                predicted_label = self.last_outputs.index(max(self.last_outputs))
                expected_label = expected_outputs.index(max(expected_outputs))
                local_accuracy_sum += 1.0 if predicted_label == expected_label else 0.0

                if self.cost_function == 'cross-entropy':
                    p = self.last_outputs[expected_label]
                    local_loss_sum += -math.log(p) if p > 0 else float('inf')
                else:
                    local_loss_sum += 0.5 * sum((r - e) ** 2 for r, e in zip(self.last_outputs, expected_outputs))

                # Print progress
                if print_rate != 0:
                    if (idx + 1) % print_rate == 0:
                        print(f"Epoch {epoch+1}, Sample {idx+1}: {predicted_label} | {expected_label}")
                        print(f"Local Accuracy (last {print_rate}): {local_accuracy_sum * 100 / print_rate:.1f}%")
                        print(f"Local Loss (last {print_rate}): {local_loss_sum / print_rate:.4f}")
                        print(f"Learning Rate: {lr:.6f}")
                        print("-" * 30)
                        local_accuracy_sum = 0.0
                        local_loss_sum = 0.0



    def save(self, filename):
        """
        Save the network's weights and biases to a JSON file.

        Args:
            filename (str): Path to the JSON file to save to
        """
        import json

        data = {
            "hidden_layers": [],
            "output_layer": []
        }

        # Save each hidden layer's weights and biases
        for layer in self.hidden_layers:
            layer_data = []
            for neuron in layer:
                layer_data.append({
                "weights": neuron.weights,
                "bias": neuron.bias,
                "weight_velocities": neuron.weight_velocities,
                "weight_squared_gradient_accumulations": neuron.weight_squared_gradient_accumulations,
                "bias_velocity": neuron.bias_velocity,
                "bias_squared_gradient_accumulation": neuron.bias_squared_gradient_accumulation
            })
            data["hidden_layers"].append(layer_data)

        # Save output layer's weights and biases
        for neuron in self.output_layer:
            data["output_layer"].append({
                "weights": neuron.weights,
                "bias": neuron.bias,
                "weight_velocities": neuron.weight_velocities,
                "weight_squared_gradient_accumulations": neuron.weight_squared_gradient_accumulations,
                "bias_velocity": neuron.bias_velocity,
                "bias_squared_gradient_accumulation": neuron.bias_squared_gradient_accumulation
            })

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Network saved to {filename}")
    
    def load(self, filename):
        """
        Load weights and biases from a JSON file into the existing network.

        Args:
            filename (str): Path to the JSON file to load from
        """
        import json

        with open(filename, "r") as f:
            data = json.load(f)

        if len(data["hidden_layers"]) != len(self.hidden_layers):
            print(f"Failed to load network: expected {len(self.hidden_layers)} hidden layers, but file has {len(data['hidden_layers'])}")
            return
        
        for layer_idx, layer_data in enumerate(data["hidden_layers"]):
            if len(layer_data) != len(self.hidden_layers[layer_idx]):
                print(f"Failed to load network: expected {len(self.hidden_layers[layer_idx])} neurons in hidden layer {layer_idx+1}, but file has {len(layer_data)}")
                return
            for neuron_idx, neuron_data in enumerate(layer_data):
                if len(neuron_data["weights"]) != len(self.hidden_layers[layer_idx][neuron_idx].weights):
                    print(f"Failed to load network: expected {len(self.hidden_layers[layer_idx][neuron_idx].weights)} weights for neuron {neuron_idx+1} in hidden layer {layer_idx+1}, but file has {len(neuron_data['weights'])}")# Thats one long line of code
                    return
                if len(neuron_data["weight_velocities"]) != len(self.hidden_layers[layer_idx][neuron_idx].weights):
                    print(f"Failed to load network: expected {len(self.hidden_layers[layer_idx][neuron_idx].weights)} weight_velocities for neuron {neuron_idx+1} in hidden layer {layer_idx+1}, but file has {len(neuron_data['weight_velocities'])}")
                    return
                if len(neuron_data["weight_squared_gradient_accumulations"]) != len(self.hidden_layers[layer_idx][neuron_idx].weights):
                    print(f"Failed to load network: expected {len(self.hidden_layers[layer_idx][neuron_idx].weights)} squared gradient accumulations for neuron {neuron_idx+1} in hidden layer {layer_idx+1}, but file has {len(neuron_data['weight_squared_gradient_accumulations'])}")
                    return
                
        
        if len(data["output_layer"]) != len(self.output_layer):
            print(f"Failed to load network: expected {len(self.output_layer)} neurons in output layer, but file has {len(data['output_layer'])}")
            return
        
        for neuron_idx, neuron_data in enumerate(data["output_layer"]):
            if len(neuron_data["weights"]) != len(self.output_layer[neuron_idx].weights):
                print(f"Failed to load network: expected {len(self.output_layer[neuron_idx].weights)} weights for output neuron {neuron_idx+1}, but file has {len(neuron_data['weights'])}")
                return
            if len(neuron_data["weight_velocities"]) != len(self.output_layer[neuron_idx].weights):
                print(f"Failed to load network: expected {len(self.output_layer[neuron_idx].weights)} weight_velocities for output neuron {neuron_idx+1}, but file has {len(neuron_data['weight_velocities'])}")
                return
            if len(neuron_data["weight_squared_gradient_accumulations"]) != len(self.output_layer[neuron_idx].weights):
                print(f"Failed to load network: expected {len(self.output_layer[neuron_idx].weights)} squared gradient accumulations for output neuron {neuron_idx+1}, but file has {len(neuron_data['weight_squared_gradient_accumulations'])}")
                return
            

        # Load hidden layer weights and biases
        for layer_idx, layer_data in enumerate(data["hidden_layers"]):
            for neuron_idx, neuron_data in enumerate(layer_data):
                self.hidden_layers[layer_idx][neuron_idx].weights = neuron_data["weights"]
                self.hidden_layers[layer_idx][neuron_idx].bias = neuron_data["bias"]
                self.hidden_layers[layer_idx][neuron_idx].weight_velocities = neuron_data["weight_velocities"]
                self.hidden_layers[layer_idx][neuron_idx].weight_squared_gradient_accumulations = neuron_data["weight_squared_gradient_accumulations"]
                self.hidden_layers[layer_idx][neuron_idx].bias_velocity = neuron_data["bias_velocity"]
                self.hidden_layers[layer_idx][neuron_idx].bias_squared_gradient_accumulation = neuron_data["bias_squared_gradient_accumulation"]

        # Load output layer weights and biases
        for neuron_idx, neuron_data in enumerate(data["output_layer"]):
            self.output_layer[neuron_idx].weights = neuron_data["weights"]
            self.output_layer[neuron_idx].bias = neuron_data["bias"]
            self.output_layer[neuron_idx].weight_velocities = neuron_data["weight_velocities"]
            self.output_layer[neuron_idx].weight_squared_gradient_accumulations = neuron_data["weight_squared_gradient_accumulations"]
            self.output_layer[neuron_idx].bias_velocity = neuron_data["bias_velocity"]
            self.output_layer[neuron_idx].bias_squared_gradient_accumulation = neuron_data["bias_squared_gradient_accumulation"]



        print(f"Network loaded from {filename}")
        



"""
Journal
9/26 Making a map wasn't actually that useful for the neuron
9/27 About to run the code for the first time
9/29 I don't wanna start learning
10/5 I just added a ton of stuff without testing yolo
"""