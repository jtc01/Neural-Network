import math
import random
from neuron import Neuron
from testing import data



class NeuralNetwork:
    """
    A flexible feedforward neural network with variable architecture.
    
    Architecture: input_size -> hidden_layers -> output_size
    Example: input_size=2, hidden_layers=[3,4,2], output_size=2 creates:
    2 inputs -> 3 hidden neurons -> 4 hidden neurons -> 2 hidden neurons -> 2 outputs
    
    This network performs forward propagation through the layers:
    1. Input layer receives the input data
    2. Multiple hidden layers process data sequentially
    3. Output layer produces final predictions
    """
    
    def __init__(self, hidden_layers, input_size=2, output_size=2, hidden_activation='sigmoid', output_activation='sigmoid', random_seed=None, cost_function='mse'):
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
        self.hidden_layers_sizes = hidden_layers.copy()  # Store the architecture
        self.output_size = output_size
        self.num_hidden_layers = len(hidden_layers)
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
        for layer_idx, layer_size in enumerate(hidden_layers):
            layer_neurons = []
            
            # Determine input size for this layer
            if layer_idx == 0:
                # First hidden layer takes inputs from input layer
                input_connections = self.input_size
            else:
                # Subsequent hidden layers take inputs from previous hidden layer
                input_connections = hidden_layers[layer_idx - 1]
            
            # Create neurons for this layer
            for neuron_idx in range(layer_size):
                # Generate random weights for connections to this neuron
                weights = [random.gauss(0, math.sqrt(2/input_connections)) for _ in range(input_connections)]
                velocities = [0.0] * len(weights)
                bias = 0.0  # Initialize bias to 0 for better training stability
                # Create the neuron and add it to the layer
                neuron = Neuron(weights=weights, bias=bias, activation_function=hidden_activation)
                layer_neurons.append(neuron)
                #print(f"Hidden Layer {layer_idx+1}, Neuron {neuron_idx+1}: weights={[round(w, 3) for w in weights]}, bias={round(bias, 3)}")
        
            self.hidden_layers.append(layer_neurons)
        
        self.output_layer = []
        last_hidden_size = hidden_layers[-1] if hidden_layers else self.input_size

    
        for i in range(self.output_size):
            weights = [random.gauss(0, math.sqrt(2/last_hidden_size)) for _ in range(last_hidden_size)]
            bias = 0.0  # Initialize bias to 0 for better training stability
            # Create the output neuron
            neuron = Neuron(weights=weights, bias=bias, activation_function=self.output_activation)
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
        arch_parts.extend([str(size) for size in self.hidden_layers_sizes])
        arch_parts.append(str(self.output_size))
        architecture_string = "-".join(arch_parts)
        
        info = {
            'architecture': architecture_string,
            'hidden_layers_sizes': self.hidden_layers_sizes,
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
        if len(expected_outputs) != len(self.output_layer):
            raise ValueError(f"Expected {len(self.output_layer)} output values, got {len(expected_outputs)}")
        
        # Calculate node value for each output neuron
        for neuron_idx, neuron in enumerate(self.output_layer):
            # Get the predicted output (activation) for this neuron
            predicted_output = neuron.last_output
            
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
                    neuron.activation_function
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
                    neuron.activation_function
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
        # Validate that we have expected outputs for each output neuron
        if len(expected_outputs) != len(self.output_layer):
            raise ValueError(f"Expected {len(self.output_layer)} output values, got {len(expected_outputs)}")
        
        # Calculate node value for each output neuron
        for neuron_idx, neuron in enumerate(self.output_layer):
            # Get the predicted output (activation) for this neuron
            predicted_output = neuron.last_output
            
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
                    neuron.activation_function
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
                    neuron.activation_function
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
                    neuron.velocities[weight_idx] = momentum * neuron.velocities[weight_idx] - learning_rate * weight_gradient

                    # Update weight using velocity
                    neuron.weights[weight_idx] += neuron.velocities[weight_idx]
            
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
                    neuron.activation_function
                )
                
                node_value *= activation_deriv
                
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
                    neuron.activation_function
                )
                
                node_value *= activation_deriv
                
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
                        neuron.velocities[weight_idx] = momentum * neuron.velocities[weight_idx] - learning_rate * weight_gradient

                        # Update weight using velocity
                        neuron.weights[weight_idx] += neuron.velocities[weight_idx]
                
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
                neuron.velocities[weight_idx] = momentum * neuron.velocities[weight_idx] - learning_rate * weight_gradient
                neuron.weights[weight_idx] += neuron.velocities[weight_idx]

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
                    neuron.velocities[weight_idx] = momentum * neuron.velocities[weight_idx] - learning_rate * weight_gradient
                    neuron.weights[weight_idx] += neuron.velocities[weight_idx]

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
                neuron.velocities[weight_idx] = momentum * neuron.velocities[weight_idx] - learning_rate * avg_weight_gradient
                neuron.weights[weight_idx] += neuron.velocities[weight_idx]

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
                    neuron.velocities[weight_idx] = momentum * neuron.velocities[weight_idx] - learning_rate * avg_weight_gradient
                    neuron.weights[weight_idx] += neuron.velocities[weight_idx]

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
                neuron.velocities[weight_idx] = momentum * neuron.velocities[weight_idx] + (1 - momentum) * weight_gradient

                # Update second moment (squared gradient)
                neuron.squared_gradient_accumulations[weight_idx] = squared_gradient_term * neuron.squared_gradient_accumulations[weight_idx] + (1 - squared_gradient_term) * (weight_gradient ** 2)

                # Compute bias-corrected moments
                velocity_corrected = neuron.velocities[weight_idx] / (1 - momentum)
                squared_gradient_corrected = neuron.squared_gradient_accumulations[weight_idx] / (1 - squared_gradient_term)

                # Update weight using Adam update rule
                neuron.weights[weight_idx] -= learning_rate * velocity_corrected / (math.sqrt(squared_gradient_corrected) + 1e-8)
            # Update bias in output layer
            bias_gradient = neuron.node_value
            bias_gradient = max(min(bias_gradient, bias_clip_value), -bias_clip_value) if bias_clip_value is not None else bias_gradient  # Clip gradients if clip value is provided
            neuron.bias_velocity = momentum * neuron.bias_velocity + (1 - momentum) * bias_gradient
            neuron.bias_squared_gradient_accumulation = squared_gradient_term * neuron.bias_squared_gradient_accumulation + (1 - squared_gradient_term) * (bias_gradient ** 2)
            bias_velocity_corrected = neuron.bias_velocity / (1 - momentum)
            bias_squared_gradient_corrected = neuron.bias_squared_gradient_accumulation / (1 - squared_gradient_term)
            neuron.bias -= learning_rate * bias_gradient / (math.sqrt(bias_squared_gradient_corrected) + 1e-8)

        for layer in self.hidden_layers:
            for neuron in layer:
                for weight_idx in range(len(neuron.weights)):
                    input_value = neuron.last_inputs[weight_idx]
                    weight_gradient = neuron.node_value * input_value
                    weight_gradient = max(min(weight_gradient, weight_clip_value), -weight_clip_value) if weight_clip_value is not None else weight_gradient  # Clip gradients if clip value is provided

                    # Update first moment (velocity)
                    neuron.velocities[weight_idx] = momentum * neuron.velocities[weight_idx] + (1 - momentum) * weight_gradient

                    # Update second moment (squared gradient)
                    neuron.squared_gradient_accumulations[weight_idx] = squared_gradient_term * neuron.squared_gradient_accumulations[weight_idx] + (1 - squared_gradient_term) * (weight_gradient ** 2)

                    # Compute bias-corrected moments
                    velocity_corrected = neuron.velocities[weight_idx] / (1 - momentum)
                    squared_gradient_corrected = neuron.squared_gradient_accumulations[weight_idx] / (1 - squared_gradient_term)

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

    def train(self, data, epochs=1, initial_learning_rate=0.005, learning_rate_decay=1.0, print_rate=1000, weight_clip_value=5.0, bias_clip_value=10.0, momentum=0.9, batch_size=1, dropout_rate=0.0):
        """
        Trains the network for a specified number of epochs on the given training data.

        Args:
            data (List): List of training samples, where each sample is a tuple (inputs, expected_outputs)
            Each 'inputs' is a 1D list of input values in some normalized order, and each 'expected_outputs' is a list of expected output values.

            epochs (int): Number of times to iterate through the entire training dataset
            initial_learning_rate (float): The starting learning rate for weight updates
            learning_rate_decay (float): Factor by which to decay the learning rate after each epoch (e.g., 0.95 for 5% decay per epoch)
            print_rate (int): How often to print the results of a single training sample (accuracy, loss, distribution, etc.)
            weight_clip_value (float): Maximum absolute value for weight gradients to prevent extreme updates
            bias_clip_value (float): Maximum absolute value for bias gradients to prevent extreme updates
            momentum (float): Momentum factor for weight and bias updates (0.0 means no momentum, 0.9 is common)

        Returns:
            None (the network's weights and biases are updated in place)
        """

        for epoch in range(epochs):
            
            total_accuracy = 0.0
            total_loss = 0.0

            local_accuracy_sum = 0.0
            local_loss_sum = 0.0

            lr = initial_learning_rate * (learning_rate_decay ** epoch)  # Decay learning rate each epoch
            
            random.seed(epoch)
            random.shuffle(data)  # Shuffle data each epoch for better training

            for idx, (inputs, expected_outputs) in enumerate(data):
                self.forward(inputs)  # Forward pass to compute outputs and store intermediate values
                if idx + 1 % batch_size == 0:
                    self.backpropagate_output_layer(expected_outputs, learning_rate=lr, weight_clip_value=weight_clip_value, bias_clip_value=bias_clip_value, momentum=momentum, update_weights=True)
                    self.backpropagate_hidden_layers(learning_rate=lr, weight_clip_value=weight_clip_value, bias_clip_value=bias_clip_value, momentum=momentum, update_weights=True)
                else:
                    self.backpropagate_output_layer(expected_outputs, learning_rate=lr, weight_clip_value=weight_clip_value, bias_clip_value=bias_clip_value, momentum=momentum, update_weights=False)
                    self.backpropagate_hidden_layers(learning_rate=lr, weight_clip_value=weight_clip_value, bias_clip_value=bias_clip_value, momentum=momentum, update_weights=False)

                # Update accuracy sum for this sample
                local_accuracy_sum += 1.0 if self.last_outputs.index(max(self.last_outputs)) == expected_outputs.index(max(expected_outputs)) else 0.0

                # Update loss for calculation and printing
                if self.cost_function == 'cross-entropy':
                    local_loss_sum += -math.log(self.last_outputs[expected_outputs.index(max(expected_outputs))]) if self.last_outputs[expected_outputs.index(max(expected_outputs))] > 0 else float('inf')
                elif self.cost_function == 'mse':
                    local_loss_sum += 0.5 * sum((r - e) ** 2 for r, e in zip(self.last_outputs, expected_outputs))
                else:
                    local_loss_sum += 0.5 * sum((r - e) ** 2 for r, e in zip(self.last_outputs, expected_outputs))  # Default to MSE

                if (idx + 1) % print_rate == 0:
                    results = self.last_outputs
                    predicted_label = results.index(max(results))
                    expected_label = expected_outputs.index(max(expected_outputs))
                    accuracy = 1.0 if predicted_label == expected_label else 0.0
                    print(f"Epoach {epoch+1}, Sample {idx+1}: Network Result: {predicted_label} | Sample Label: {expected_label}")
                    if self.cost_function == 'cross-entropy':
                        loss = -math.log(results[expected_label]) if results[expected_label] > 0 else float('inf')
                    elif self.cost_function == 'mse':
                        loss = 0.5 * sum((r - e) ** 2 for r, e in zip(results, expected_outputs))
                    else:
                        loss = 0.5 * sum((r - e) ** 2 for r, e in zip(results, expected_outputs))  # Default to MSE
                    print(f"Loss: {loss:.4f}")
                    print(f"Output Distribution: [{', '.join(f'{r:.4f}' for r in results)}]")
                    print(f"Label Distribution:  [{', '.join(f'{e:.4f}' for e in expected_outputs)}]")

                    print(f"Local Accuracy (last {print_rate} samples): {local_accuracy_sum * 100 / print_rate:.4f}%")
                    print(f"Local Loss (last {print_rate} samples): {local_loss_sum / print_rate:.4f}")
                    print("-" * 30)

                    local_accuracy_sum = 0.0
                    local_loss_sum = 0.0
    
    def train_adam(self, data, epochs=1, initial_learning_rate=0.005, print_rate=1000, weight_clip_value=5.0, bias_clip_value=10.0, momentum=0.9, squared_gradient_term=0.999, batch_size=1, dropout_rate=0.0):
        """
        Trains the network for a specified number of epochs on the given training data.

        Args:
            data (List): List of training samples, where each sample is a tuple (inputs, expected_outputs)
            Each 'inputs' is a 1D list of input values in some normalized order, and each 'expected_outputs' is a list of expected output values.

            epochs (int): Number of times to iterate through the entire training dataset
            initial_learning_rate (float): The starting learning rate for weight updates
            learning_rate_decay (float): Factor by which to decay the learning rate after each epoch (e.g., 0.95 for 5% decay per epoch)
            print_rate (int): How often to print the results of a single training sample (accuracy, loss, distribution, etc.)
            weight_clip_value (float): Maximum absolute value for weight gradients to prevent extreme updates
            bias_clip_value (float): Maximum absolute value for bias gradients to prevent extreme updates
            momentum (float): Momentum factor for weight and bias updates (0.0 means no momentum, 0.9 is common)

        Returns:
            None (the network's weights and biases are updated in place)
        """

        for epoch in range(epochs):
            
            total_accuracy = 0.0
            total_loss = 0.0

            local_accuracy_sum = 0.0
            local_loss_sum = 0.0

            lr = initial_learning_rate  # Adam uses adaptive learning rates, so we don't decay it here

            random.seed(epoch)
            random.shuffle(data)  # Shuffle data each epoch for better training

            for idx, (inputs, expected_outputs) in enumerate(data):
                self.forward(inputs)  # Forward pass to compute outputs and store intermediate values
                if idx + 1 % batch_size == 0:
                    self.backpropagate_output_layer(expected_outputs, learning_rate=lr, weight_clip_value=weight_clip_value, bias_clip_value=bias_clip_value, momentum=momentum, update_weights=True)
                    self.backpropagate_hidden_layers(learning_rate=lr, weight_clip_value=weight_clip_value, bias_clip_value=bias_clip_value, momentum=momentum, update_weights=True)
                else:
                    self.backpropagate_output_layer(expected_outputs, learning_rate=lr, weight_clip_value=weight_clip_value, bias_clip_value=bias_clip_value, momentum=momentum, update_weights=False)
                    self.backpropagate_hidden_layers(learning_rate=lr, weight_clip_value=weight_clip_value, bias_clip_value=bias_clip_value, momentum=momentum, update_weights=False)

                # Update accuracy sum for this sample
                local_accuracy_sum += 1.0 if self.last_outputs.index(max(self.last_outputs)) == expected_outputs.index(max(expected_outputs)) else 0.0

                # Update loss for calculation and printing
                if self.cost_function == 'cross-entropy':
                    local_loss_sum += -math.log(self.last_outputs[expected_outputs.index(max(expected_outputs))]) if self.last_outputs[expected_outputs.index(max(expected_outputs))] > 0 else float('inf')
                elif self.cost_function == 'mse':
                    local_loss_sum += 0.5 * sum((r - e) ** 2 for r, e in zip(self.last_outputs, expected_outputs))
                else:
                    local_loss_sum += 0.5 * sum((r - e) ** 2 for r, e in zip(self.last_outputs, expected_outputs))  # Default to MSE

                if (idx + 1) % print_rate == 0:
                    results = self.last_outputs
                    predicted_label = results.index(max(results))
                    expected_label = expected_outputs.index(max(expected_outputs))
                    accuracy = 1.0 if predicted_label == expected_label else 0.0
                    print(f"Epoach {epoch+1}, Sample {idx+1}: Network Result: {predicted_label} | Sample Label: {expected_label}")
                    if self.cost_function == 'cross-entropy':
                        loss = -math.log(results[expected_label]) if results[expected_label] > 0 else float('inf')
                    elif self.cost_function == 'mse':
                        loss = 0.5 * sum((r - e) ** 2 for r, e in zip(results, expected_outputs))
                    else:
                        loss = 0.5 * sum((r - e) ** 2 for r, e in zip(results, expected_outputs))  # Default to MSE
                    print(f"Loss: {loss:.4f}")
                    print(f"Output Distribution: [{', '.join(f'{r:.4f}' for r in results)}]")
                    print(f"Label Distribution:  [{', '.join(f'{e:.4f}' for e in expected_outputs)}]")

                    print(f"Local Accuracy (last {print_rate} samples): {local_accuracy_sum * 100 / print_rate:.4f}%")
                    print(f"Local Loss (last {print_rate} samples): {local_loss_sum / print_rate:.4f}")
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
                    "bias": neuron.bias
                })
            data["hidden_layers"].append(layer_data)

        # Save output layer's weights and biases
        for neuron in self.output_layer:
            data["output_layer"].append({
                "weights": neuron.weights,
                "bias": neuron.bias
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
        
        if len(data["output_layer"]) != len(self.output_layer):
            print(f"Failed to load network: expected {len(self.output_layer)} neurons in output layer, but file has {len(data['output_layer'])}")
            return
        
        for neuron_idx, neuron_data in enumerate(data["output_layer"]):
            if len(neuron_data["weights"]) != len(self.output_layer[neuron_idx].weights):
                print(f"Failed to load network: expected {len(self.output_layer[neuron_idx].weights)} weights for output neuron {neuron_idx+1}, but file has {len(neuron_data['weights'])}")
                return

        # Load hidden layer weights and biases
        for layer_idx, layer_data in enumerate(data["hidden_layers"]):
            for neuron_idx, neuron_data in enumerate(layer_data):
                self.hidden_layers[layer_idx][neuron_idx].weights = neuron_data["weights"]
                self.hidden_layers[layer_idx][neuron_idx].bias = neuron_data["bias"]

        # Load output layer weights and biases
        for neuron_idx, neuron_data in enumerate(data["output_layer"]):
            self.output_layer[neuron_idx].weights = neuron_data["weights"]
            self.output_layer[neuron_idx].bias = neuron_data["bias"]

        print(f"Network loaded from {filename}")
        



"""
Journal
9/26 Making a map wasn't actually that useful for the neuron
9/27 About to run the code for the first time
9/29 I don't wanna start learning
10/5 I just added a ton of stuff without testing yolo
"""