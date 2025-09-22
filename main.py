import math

class Neuron:
    """
    A single neuron implementation for a neural network.
    
    This class represents a basic artificial neuron that can:
    - Accept multiple inputs
    - Apply weights to each input
    - Add a bias term
    - Apply an activation function
    - Return a single output
    """
    
    def __init__(self, weights=None, bias=0.0, activation_function='sigmoid'):
        """
        Initialize the neuron with weights, bias, and activation function.
        
        Args:
            weights (list): List of weights for each input connection
            bias (float): Bias value to be added before activation
            activation_function (str): Type of activation function to use
        """
        self.weights = weights if weights is not None else []
        self.bias = bias
        self.activation_function = activation_function
        
        # Store the last computed values for debugging purposes
        self.last_inputs = None
        self.last_weighted_sum = None
        self.last_output = None
    
    def sigmoid(self, x):
        """
        Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
        
        This function maps any real number to a value between 0 and 1.
        Commonly used for binary classification problems.
        
        Args:
            x (float): Input value to the activation function
            
        Returns:
            float: Output value between 0 and 1
        """
        # Prevent overflow by clipping extreme values
        if x > 500:
            return 1.0
        elif x < -500:
            return 0.0
        else:
            return 1.0 / (1.0 + math.exp(-x))
    
    def relu(self, x):
        """
        ReLU (Rectified Linear Unit) activation function: f(x) = max(0, x)
        
        This function returns the input if positive, otherwise returns 0.
        Commonly used in hidden layers of deep networks.
        
        Args:
            x (float): Input value to the activation function
            
        Returns:
            float: Output value (0 if input is negative, input value if positive)
        """
        return max(0.0, x)
    
    def tanh(self, x):
        """
        Hyperbolic tangent activation function: f(x) = tanh(x)
        
        This function maps any real number to a value between -1 and 1.
        Similar to sigmoid but centered around 0.
        
        Args:
            x (float): Input value to the activation function
            
        Returns:
            float: Output value between -1 and 1
        """
        return math.tanh(x)
    
    def linear(self, x):
        """
        Linear activation function: f(x) = x
        
        This function returns the input unchanged.
        Used in regression problems or output layers.
        
        Args:
            x (float): Input value to the activation function
            
        Returns:
            float: Input value unchanged
        """
        return x
    
    def apply_activation(self, x):
        """
        Apply the specified activation function to the input.
        
        Args:
            x (float): Input value to apply activation to
            
        Returns:
            float: Result after applying the activation function
        """
        if self.activation_function == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_function == 'relu':
            return self.relu(x)
        elif self.activation_function == 'tanh':
            return self.tanh(x)
        elif self.activation_function == 'linear':
            return self.linear(x)
        else:
            # Default to sigmoid if unknown activation function
            print(f"Warning: Unknown activation function '{self.activation_function}'. Using sigmoid.")
            return self.sigmoid(x)
    
    def forward(self, inputs):
        """
        Perform the forward pass through the neuron.
        
        This method implements the core neuron computation:
        1. Multiply each input by its corresponding weight
        2. Sum all weighted inputs
        3. Add the bias term
        4. Apply the activation function
        5. Return the final output
        
        Args:
            inputs (list): List of input values
            
        Returns:
            float: The neuron's output after processing
            
        Raises:
            ValueError: If the number of inputs doesn't match the number of weights
        """
        # Validate input dimensions
        if len(inputs) != len(self.weights):
            raise ValueError(f"Number of inputs ({len(inputs)}) must match number of weights ({len(self.weights)})")
        
        # Store inputs for debugging
        self.last_inputs = inputs.copy()
        
        # Step 1: Calculate the weighted sum of inputs
        # This is the dot product of inputs and weights: Σ(input_i * weight_i)
        weighted_sum = 0.0
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i]
        
        # Step 2: Add the bias term
        # The bias allows the neuron to shift the activation function
        weighted_sum += self.bias
        
        # Store the weighted sum for debugging
        self.last_weighted_sum = weighted_sum
        
        # Step 3: Apply the activation function
        # This introduces non-linearity to the neuron's output
        output = self.apply_activation(weighted_sum)
        
        # Store the output for debugging
        self.last_output = output
        
        return output
    
    def set_weights(self, weights):
        """
        Set new weights for the neuron.
        
        Args:
            weights (list): New list of weights
        """
        self.weights = weights.copy()
    
    def set_bias(self, bias):
        """
        Set a new bias value for the neuron.
        
        Args:
            bias (float): New bias value
        """
        self.bias = bias
    
    def get_info(self):
        """
        Get information about the neuron's current state.
        
        Returns:
            dict: Dictionary containing neuron information
        """
        return {
            'weights': self.weights,
            'bias': self.bias,
            'activation_function': self.activation_function,
            'last_inputs': self.last_inputs,
            'last_weighted_sum': self.last_weighted_sum,
            'last_output': self.last_output
        }


class NeuralNetwork:
    """
    A simple feedforward neural network with one hidden layer.
    
    Architecture: 2 inputs -> 2 hidden neurons -> 2 outputs
    
    This network performs forward propagation through the layers:
    1. Input layer receives the input data
    2. Hidden layer processes inputs through neurons with weights and biases
    3. Output layer produces final predictions
    """
    
    def __init__(self, hidden_activation='sigmoid', output_activation='sigmoid', random_seed=None):
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
        
        # Network architecture parameters
        self.input_size = 2
        self.hidden_size = 2
        self.output_size = 2
        
        # Store activation function types
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        
        # Initialize the hidden layer (2 neurons, each taking 2 inputs)
        self.hidden_layer = []
        for i in range(self.hidden_size):
            # Generate random weights for connections from input layer to this hidden neuron
            # Weights are initialized between -1 and 1
            weights = [random.uniform(-1, 1) for _ in range(self.input_size)]
            # Generate random bias between -1 and 1
            bias = random.uniform(-1, 1)
            # Create the hidden neuron
            neuron = Neuron(weights=weights, bias=bias, activation_function=hidden_activation)
            self.hidden_layer.append(neuron)
            print(f"Hidden neuron {i+1}: weights={[round(w, 3) for w in weights]}, bias={round(bias, 3)}")
        
        # Initialize the output layer (2 neurons, each taking 2 inputs from hidden layer)
        self.output_layer = []
        for i in range(self.output_size):
            # Generate random weights for connections from hidden layer to this output neuron
            weights = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
            # Generate random bias between -1 and 1
            bias = random.uniform(-1, 1)
            # Create the output neuron
            neuron = Neuron(weights=weights, bias=bias, activation_function=output_activation)
            self.output_layer.append(neuron)
            print(f"Output neuron {i+1}: weights={[round(w, 3) for w in weights]}, bias={round(bias, 3)}")
        
        # Store intermediate values for debugging and visualization
        self.last_inputs = None
        self.last_hidden_outputs = None
        self.last_outputs = None
    
    def forward(self, inputs):
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
        
        # Step 1: Forward pass through hidden layer
        # Each hidden neuron processes the same input data
        hidden_outputs = []
        for i, neuron in enumerate(self.hidden_layer):
            output = neuron.forward(inputs)
            hidden_outputs.append(output)
        
        # Store hidden layer outputs for debugging
        self.last_hidden_outputs = hidden_outputs.copy()
        
        # Step 2: Forward pass through output layer
        # Each output neuron takes the hidden layer outputs as inputs
        final_outputs = []
        for i, neuron in enumerate(self.output_layer):
            output = neuron.forward(hidden_outputs)
            final_outputs.append(output)
        
        # Store final outputs for debugging
        self.last_outputs = final_outputs.copy()
        
        return final_outputs
    
    def get_network_info(self):
        """
        Get detailed information about the network's current state.
        
        Returns:
            dict: Dictionary containing network information
        """
        info = {
            'architecture': f"{self.input_size}-{self.hidden_size}-{self.output_size}",
            'hidden_activation': self.hidden_activation,
            'output_activation': self.output_activation,
            'last_inputs': self.last_inputs,
            'last_hidden_outputs': self.last_hidden_outputs,
            'last_outputs': self.last_outputs,
            'hidden_layer_weights': [],
            'hidden_layer_biases': [],
            'output_layer_weights': [],
            'output_layer_biases': []
        }
        
        # Collect hidden layer parameters
        for i, neuron in enumerate(self.hidden_layer):
            info['hidden_layer_weights'].append(neuron.weights)
            info['hidden_layer_biases'].append(neuron.bias)
        
        # Collect output layer parameters
        for i, neuron in enumerate(self.output_layer):
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
        Print a visual representation of the network structure.
        """
        print("\n=== Network Architecture ===")
        print("Input Layer (2 nodes)")
        print("     |")
        print("Hidden Layer (2 neurons)")
        for i in range(self.hidden_size):
            weights = [round(w, 3) for w in self.hidden_layer[i].weights]
            bias = round(self.hidden_layer[i].bias, 3)
            print(f"  Neuron {i+1}: W={weights}, b={bias}")
        print("     |")
        print("Output Layer (2 neurons)")
        for i in range(self.output_size):
            weights = [round(w, 3) for w in self.output_layer[i].weights]
            bias = round(self.output_layer[i].bias, 3)
            print(f"  Neuron {i+1}: W={weights}, b={bias}")