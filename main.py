import math
import random

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
    A flexible feedforward neural network with variable architecture.
    
    Architecture: input_size -> hidden_layers -> output_size
    Example: input_size=2, hidden_layers=[3,4,2], output_size=2 creates:
    2 inputs -> 3 hidden neurons -> 4 hidden neurons -> 2 hidden neurons -> 2 outputs
    
    This network performs forward propagation through the layers:
    1. Input layer receives the input data
    2. Multiple hidden layers process data sequentially
    3. Output layer produces final predictions
    """
    
    def __init__(self, hidden_layers, input_size=2, output_size=2, hidden_activation='sigmoid', output_activation='sigmoid', random_seed=None):
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
        
        # Store activation function types
        self.hidden_activation = hidden_activation
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
                weights = [random.uniform(-1, 1) for _ in range(input_connections)]
                bias = random.uniform(-1, 1)
                neuron = Neuron(weights=weights, bias=bias, activation_function=hidden_activation)
                layer_neurons.append(neuron)
                print(f"Hidden Layer {layer_idx+1}, Neuron {neuron_idx+1}: weights={[round(w, 3) for w in weights]}, bias={round(bias, 3)}")
        
            self.hidden_layers.append(layer_neurons)
        
        self.output_layer = []
        last_hidden_size = hidden_layers[-1] if hidden_layers else self.input_size
    
        for i in range(self.output_size):
            weights = [random.uniform(-1, 1) for _ in range(last_hidden_size)]
            bias = random.uniform(-1, 1)
            # Create the output neuron
            neuron = Neuron(weights=weights, bias=bias, activation_function=output_activation)
            self.output_layer.append(neuron)
            print(f"Output neuron {i+1}: weights={[round(w, 3) for w in weights]}, bias={round(bias, 3)}")
            
            # Store intermediate values for debugging and visualization
        self.last_inputs = None
        self.last_hidden_outputs = []
        self.last_outputs = None

        self.cost = 0.0
    
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
        
        # Step 1: Forward pass through all hidden layers sequentially
        current_inputs = inputs
        self.last_hidden_outputs = []  # Reset hidden outputs storage
        
        for layer_idx, layer in enumerate(self.hidden_layers):
            layer_outputs = []
            
            # Process current inputs through each neuron in this layer
            for neuron in layer:
                output = neuron.forward(current_inputs)
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

    def calculate_cost(self, inputs_array, expected_outputs_array):
        """
        Calculate the cost (loss) for a batch of data using Mean Squared Error.
        
        Cost = (1/2) * Σ(predicted_output - expected_output)²
        
        This measures how wrong the network's predictions are compared to the expected outputs.
        Lower cost means better predictions.
        
        Args:
            inputs_array (list): 2D list where each element is a list of input values
                                Example: [[x1, y1], [x2, y2], [x3, y3]]
            expected_outputs_array (list): 2D list where each element is a list of expected output values
                                        Example: [[t1, f1], [t2, f2], [t3, f3]]
            
        Returns:
            float: The calculated cost value
            
        Raises:
            ValueError: If inputs_array and expected_outputs_array have different lengths
        """

        if len(inputs_array) != len(expected_outputs_array):
            raise ValueError(f"Number of input samples ({len(inputs_array)}) must match number of output samples ({len(expected_outputs_array)})")
    
        total_cost = 0.0
        
        # Calculate cost for each data sample
        for inputs, expected_outputs in zip(inputs_array, expected_outputs_array):
            # Get the network's prediction for this input
            predicted_outputs = self.forward(inputs)
            
            # Validate output dimensions
            if len(predicted_outputs) != len(expected_outputs):
                raise ValueError(f"Predicted outputs length ({len(predicted_outputs)}) doesn't match expected outputs length ({len(expected_outputs)})")
            
            # Calculate squared error for each output neuron
            for predicted, expected in zip(predicted_outputs, expected_outputs):
                error = predicted - expected
                total_cost += error ** 2
        
        # Divide by 2 (as per MSE formula)
        total_cost = total_cost / 2.0
        
        # Store the cost in the instance variable
        self.cost = total_cost
        
        return total_cost
    def calculate_single_cost(self, inputs, actual_outputs):
        """
        Calculate the cost for a single data point.
        
        Args:
            data_point (DataPoint): Single DataPoint object
            
        Returns:
            float: The cost for this single data point
        """

        # Get prediction
        predicted_outputs = self.forward(inputs)
        
        if len(predicted_outputs) != len(actual_outputs):
            raise ValueError(f"Predicted outputs length ({len(predicted_outputs)}) doesn't match expected outputs length ({len(expected_outputs)})")

        # Calculate cost
        cost = 0.0
        for predicted, actual in zip(predicted_outputs, actual_outputs):
            error = predicted - actual
            cost += error ** 2
        
        cost = cost / 2.0
        
        return cost
    
    def learn(self, inputs_array, expected_outputs_array, learning_rate=0.01, h=0.0001):
        """
        Train the network using numerical gradient approximation (finite differences).
        
        This is a primitive learning method that:
        1. Tests each weight/bias by making a small change (h)
        2. Measures how the cost changes
        3. Adjusts weights/biases in the direction that reduces cost
        
        This method is slow but educational - it shows how gradient descent works.
        Real neural networks use backpropagation which is much faster.
        
        Args:
            inputs_array (list): 2D list of input samples [[x1, y1], [x2, y2], ...]
            expected_outputs_array (list): 2D list of expected outputs [[t1, f1], [t2, f2], ...]
            learning_rate (float): How much to adjust weights/biases (default: 0.01)
            h (float): Small value for testing weight changes (default: 0.0001)
            
        Returns:
            float: The cost after this learning iteration
        """
        
        # Calculate the initial cost (before any changes)
        initial_cost = self.calculate_cost(inputs_array, expected_outputs_array)
        
        # Create multi-dimensional arrays to store cost gradients
        # Structure mirrors the network architecture
        hidden_weight_gradients = []  # Gradients for hidden layer weights
        hidden_bias_gradients = []    # Gradients for hidden layer biases
        output_weight_gradients = []  # Gradients for output layer weights
        output_bias_gradients = []    # Gradients for output layer biases
        
        print(f"Initial cost: {initial_cost:.6f}")
        print("Calculating gradients...")
        
        # ============================================
        # STEP 1: Calculate gradients for HIDDEN LAYERS
        # ============================================
        for layer_idx, layer in enumerate(self.hidden_layers):
            layer_weight_gradients = []
            layer_bias_gradients = []
            
            for neuron_idx, neuron in enumerate(layer):
                neuron_weight_gradients = []
                
                # Test each weight in this neuron
                for weight_idx in range(len(neuron.weights)):
                    # Save original weight
                    original_weight = neuron.weights[weight_idx]
                    
                    # Increase weight by h
                    neuron.weights[weight_idx] = original_weight + h
                    
                    # Calculate new cost with modified weight
                    new_cost = self.calculate_cost(inputs_array, expected_outputs_array)
                    
                    # Calculate gradient: (change in cost) / (change in weight)
                    gradient = (new_cost - initial_cost) / h
                    neuron_weight_gradients.append(gradient)
                    
                    # Restore original weight
                    neuron.weights[weight_idx] = original_weight
                
                layer_weight_gradients.append(neuron_weight_gradients)
                
                # Test the bias for this neuron
                original_bias = neuron.bias
                
                # Increase bias by h
                neuron.bias = original_bias + h
                
                # Calculate new cost with modified bias
                new_cost = self.calculate_cost(inputs_array, expected_outputs_array)
                
                # Calculate gradient
                bias_gradient = (new_cost - initial_cost) / h
                layer_bias_gradients.append(bias_gradient)
                
                # Restore original bias
                neuron.bias = original_bias
            
            hidden_weight_gradients.append(layer_weight_gradients)
            hidden_bias_gradients.append(layer_bias_gradients)
        
        # ============================================
        # STEP 2: Calculate gradients for OUTPUT LAYER
        # ============================================
        for neuron_idx, neuron in enumerate(self.output_layer):
            neuron_weight_gradients = []
            
            # Test each weight in this output neuron
            for weight_idx in range(len(neuron.weights)):
                # Save original weight
                original_weight = neuron.weights[weight_idx]
                
                # Increase weight by h
                neuron.weights[weight_idx] = original_weight + h
                
                # Calculate new cost with modified weight
                new_cost = self.calculate_cost(inputs_array, expected_outputs_array)
                
                # Calculate gradient
                gradient = (new_cost - initial_cost) / h
                neuron_weight_gradients.append(gradient)
                
                # Restore original weight
                neuron.weights[weight_idx] = original_weight
            
            output_weight_gradients.append(neuron_weight_gradients)
            
            # Test the bias for this output neuron
            original_bias = neuron.bias
            
            # Increase bias by h
            neuron.bias = original_bias + h
            
            # Calculate new cost with modified bias
            new_cost = self.calculate_cost(inputs_array, expected_outputs_array)
            
            # Calculate gradient
            bias_gradient = (new_cost - initial_cost) / h
            output_bias_gradients.append(bias_gradient)
            
            # Restore original bias
            neuron.bias = original_bias
        
        print("Gradients calculated. Updating weights and biases...")
        
        # ============================================
        # STEP 3: Update all HIDDEN LAYER weights and biases
        # ============================================
        for layer_idx, layer in enumerate(self.hidden_layers):
            for neuron_idx, neuron in enumerate(layer):
                # Update weights
                for weight_idx in range(len(neuron.weights)):
                    gradient = hidden_weight_gradients[layer_idx][neuron_idx][weight_idx]
                    # Move in opposite direction of gradient (gradient descent)
                    neuron.weights[weight_idx] -= learning_rate * gradient
                
                # Update bias
                bias_gradient = hidden_bias_gradients[layer_idx][neuron_idx]
                neuron.bias -= learning_rate * bias_gradient
        
        # ============================================
        # STEP 4: Update all OUTPUT LAYER weights and biases
        # ============================================
        for neuron_idx, neuron in enumerate(self.output_layer):
            # Update weights
            for weight_idx in range(len(neuron.weights)):
                gradient = output_weight_gradients[neuron_idx][weight_idx]
                neuron.weights[weight_idx] -= learning_rate * gradient
            
            # Update bias
            bias_gradient = output_bias_gradients[neuron_idx]
            neuron.bias -= learning_rate * bias_gradient
        
        # Calculate final cost after updates
        final_cost = self.calculate_cost(inputs_array, expected_outputs_array)
        self.cost = final_cost
        
        print(f"Final cost: {final_cost:.6f}")
        print(f"Cost improvement: {initial_cost - final_cost:.6f}\n")
        
        return final_cost

def generate_data_two_inputs(numSamples):
    data=[]
    for i in range(numSamples):
        x = random.uniform(-5,5)
        y = random.uniform(-5,5)
        dataPoint = DataPoint(x,y)
        data.append(dataPoint)
    return data

class DataPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.set_expected_outputs(x,y)

    def give_inputs(self):
        inputs=[self.x,self.y]
        return inputs
    
    def set_expected_outputs(self, x, y):#dynamic function that can change depending on what we want
        
        if y > (x/2)**2 - 2:
            self.t=1
            self.f=0
        else:
            self.t=0
            self.f=1

        """
        if y>x*-1:
            self.t=1
            self.f=0
        else:
            self.t=0
            self.f=1
        """
        
        """
        if y<-x**4-x**3+3*x**2+2*x+4 and y>x**4/5-x**3+x/2:
            self.t=1
            self.f=0
        else:
            self.t=0
            self.f=1
        """

def main():

    network = NeuralNetwork(
        hidden_layers=[4],
        input_size=2,
        output_size=2,
        hidden_activation='sigmoid',
        output_activation='sigmoid',
        random_seed=42
    )

    while True:
        inputs=[]
        data_points=[]
        expected_outputs=[]

        data_points=generate_data_two_inputs(50).copy()

        for point in data_points:
            input=[point.x, point.y]
            inputs.append(input)
            expected_output=[point.t, point.f]
            expected_outputs.append(expected_output)
        cost = network.learn(inputs, expected_outputs)
        if cost<=1:
            break
    network.print_network_structure()
    a=0
    for i in range(100):
        point = DataPoint(random.uniform(-5,5), random.uniform(-5,5))
        inputs = [point.x, point.y]
        expected = [point.t, point.f]
        output = network.forward(inputs)
        if ((output[0]>output[1] and expected[0]>expected[1]) or (output[0]<output[1] and expected[0]<expected[1])):
            a+=1
    print(a)

main()

"""
Journal
9/26 Making a map wasn't actually that useful for the neuron
9/27 About to run the code for the first time
9/29 I don't wanna start learning
"""