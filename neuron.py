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
        self.velocities = [0.0] * len(self.weights)  # For momentum-based updates
        self.bias = bias
        self.bias_velocity = 0.0  # For momentum-based updates
        self.activation_function = activation_function
        
        # Store the last computed values for debugging purposes
        self.last_inputs = None
        self.last_weighted_sum = None
        self.last_output = None

        # Node value = ∂Cost/∂(weighted_sum) for this neuron
        self.node_value = 0.0

    
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
    
    def leaky_relu(self, x):
        """
        Leaky ReLU activation function: f(x) = max(0.01*x, x)
        
        This function allows a small gradient when the input is negative,
        which can help prevent "dying ReLU" problems in deep networks.
        
        Args:
            x (float): Input value to the activation function
            
        Returns:
            float: Output value (0.01*x if input is negative, input value if positive)
        """
        return x if x > 0 else 0.01 * x
    
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
    
    def exponential(self, x):
        """
        Exponential activation function: f(x) = e^x
        
        This function maps any real number to a positive value.
        Can be used in certain types of networks or output layers.
        
        Args:
            x (float): Input value to the activation function
        """
        return math.exp(x)
    
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
        elif self.activation_function == 'leaky_relu':
            return self.leaky_relu(x)
        elif self.activation_function == 'tanh':
            return self.tanh(x)
        elif self.activation_function == 'linear':
            return self.linear(x)
        elif self.activation_function == 'exponential':
            return self.exponential(x)
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
            'last_output': self.last_output,
            'node_value': self.node_value
        }