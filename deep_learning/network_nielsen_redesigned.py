# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 00:30:20 2019

@author: User
"""

"""
network.py
~~~~~~~~~~
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  
Gradients are calculated using backpropagation.  
"""

"""
python object
https://www.miximum.fr/blog/introduction-au-deep-learning-2/
https://pythonmachinelearning.pro/complete-guide-to-deep-neural-networks-part-2/
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py
"""

"""
python object

SGD
	FOR ALL  MINI_BATCHES :
		FOR ALL  TRAINING EXAMPLES :
			single_step_gradient_descent
				single_training_example_gradient
					backprop
						feedforward
							sigmoid
						backward_update
							error_LAST_layer
								sigmoid_derivative
							update_nabla_w
							FOR ALL  LAYER :
								error_PREVIOUS_layer
									sigmoid_derivative
								update_nabla_w
							END FOR
					update_nabla			
			update_network
		ENF FOR
	ENF FOR
	evaluate
		feedforward

"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):
   
    def __init__(self,layers):     
        self.layers=layers
        self.nbr_layers= len(layers)
        self.biases = self.init_biases(layers)
        self.weights = self.init_weights(layers)
        
    """
    init network's BIASES and WEIGHTS
    """
    def init_biases(self,layers):      
        ### no bias on the first layer 
        ### --> start to init at the SECOND layer (1:)
        lst_layer_EXCEPT_FIRST_one=layers[1:]
        biases = [np.random.randn(follow, 1) for follow in lst_layer_EXCEPT_FIRST_one]
        return biases
        
    def init_weights(self,layers):        
        """ since the LAST LAYER (:-1) DOESN'T HAVE WEIGHT """
        """ The last layer with weights is the LAST BUT ONE LAYER """
        lst_layer_EXCEPT_LAST_one=layers[:-1]
        """ The first layer with weights: is the SECOND LAYER (1:)""" 
        """ since the FIRST LAYER (0:) DOESN'T HAVE WEIGHT """
        lst_layer_EXCEPT_FIRST_one=layers[1:]
  
        weights = [np.random.randn(follow, previous) for previous,follow in 
                   zip(lst_layer_EXCEPT_LAST_one,lst_layer_EXCEPT_FIRST_one)
                ]
        return weights

    """
    init GRADIENT (nabla WEIGHTS and nabla BIASES)
	before to start gradient descent  
    """
    def init_nabla(self,nodes):
        nabla = [np.zeros(node.shape) for node in nodes]
        return nabla 

    """
    create MINI_BATCHES for Stochastic Gradient Descent
    """
    def create_mini_batches(self,training_data,mini_batch_size):
        sample_size = len(training_data)
        random.shuffle(training_data)
        mini_batches = [
                training_data[partition_size:partition_size+mini_batch_size]
                for partition_size in range(0, sample_size, mini_batch_size)
                ]
        return mini_batches       
    """
    Stochastic Gradient Descent
	use gradient descent to FIND THE WEIGHTS AND BIASES  which MINIMIZE THE COST FUNCTION
	estimate the gradient by computing for a SMALL SAMPLE OF RANDOMLY chosen training inputs
	``mini_batch`` --> list of tuples ``(training input,desired output)``
	``eta`` 		--> the learning rate, the step step length dscent :
    """
    def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):
        if test_data: test_data_size = len(test_data)      
        #### loop on epoch: average on on all training into the sample
        for epoch in range(epochs):
            """ mini_batches: small sample of RANDOMLY chosen training inputs """
            """ (Hence the term 'STOCHASTIC' term) """
            mini_batches =self.create_mini_batches(training_data,mini_batch_size)
            """ loop on mini_batch: Update weights and biases """
            for mini_batch in mini_batches:
                self.single_step_gradient_descent (mini_batch,eta)
				
            if test_data:
                print ('Epoch {0}: {1} / {2}'.format(epoch,epoch+1,epoch+2),test_data_size)
            else:
                print ('Epoch {0} complete'.format(epoch))

    """
    update single_step_gradient_descent
    """
    """Update the network's (weights and biases) 
	by applying gradient descent using backpropagation to a single mini batch.
     ``mini_batch`` --> list of tuples ``(training input,desired output)``
	 ``eta`` 		--> the learning rate, the step step length dscent :
	 """
    def single_step_gradient_descent(self,mini_batch,eta):
        """ init GRADIENT """
        nabla_w =self.init_nabla(self.weights)
        nabla_b =self.init_nabla(self.biases)  
        neurons_gradient= (nabla_w,nabla_b)     
        """ for EVERY TRAINING EXAMPLE of the mini_batch: """
        for training_input,desired_output in mini_batch:            
            """--> find to HOW UPDATE network (minimize the cost function)"""
            """-->compute GRADIENT: """
            example_minibatch= (training_input,desired_output)
            new_nabla_b, new_nabla_w = self.single_training_example_gradient(example_minibatch,neurons_gradient)
            """UPDATE NETWORK: update weights and biases"""
            example_mini_batch_size=len(mini_batch)
            gradient_hyperparameters= (example_mini_batch_size,eta)
            example_gradient= (new_nabla_b,new_nabla_w)
            self.update_network(gradient_hyperparameters,example_gradient)

    """applying gradient descent using backpropagation"""
    def single_training_example_gradient(self,example_minibatch,neurons_gradient):
        """setting PARAMETERS """
        training_input = example_minibatch[0]
        desired_output = example_minibatch[1]
        nabla_w = neurons_gradient[0]
        nabla_b = neurons_gradient[1]
        """ computation GRADIENT (through BACKPROPAGATION) """
        new_nabla_b, new_nabla_w = self.backprop(training_input,desired_output)
        nabla_b = self.update_nabla(nabla_b, new_nabla_b)
        nabla_w = self.update_nabla(nabla_w, new_nabla_w)
        return (new_nabla_b,new_nabla_w)

    def update_network(self,gradient_hyperparameters,neurons_gradient):
        """ setting PARAMETERS"""
        mini_batch_size = gradient_hyperparameters[0]
        eta = gradient_hyperparameters[1]
        nabla_w = neurons_gradient[0]
        nabla_b = neurons_gradient[1]
        """ update_network : (weights,biases) = AVERAGE GRADIENT"""
        """ update in the OPPOSITE gradient's direction """
        weights = [w-(eta/mini_batch_size)*nw for w, nw in zip(self.weights, nabla_w)]
        biases = [b-(eta/mini_batch_size)*nb for b, nb in zip(self.biases, nabla_b)]        
        return (weights,biases)
        
    def update_nabla(self,nabla_input,delta_nabla_input):
        """ setting parameters"""
        nabla=list(nabla_input)
        delta_nabla=list(delta_nabla_input)
        """ computation"""
        nabla= [nabla+delta_nabla for nabla, delta_nabla in zip(nabla, delta_nabla)]
        return (nabla)

    """
    backprop
    """
    """Return a tuple ``(nabla_b, nabla_w)`` representing the
    gradient for the cost function C_x.  ``nabla_b`` and
    ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    to ``self.biases`` and ``self.weights`"""
    def backprop(self,training_input,desired_output):        
        """feedforward : store all the ACTIVATIONS and Z VECTORS, layer by layer"""
        input_effect_network =self.feedforward(training_input)
        """backward pass : update GRADIENT"""
        nabla_b, nabla_w = self.backward_update(desired_output,input_effect_network)
        return (nabla_b,nabla_w)                
        
    def feedforward(self,training_input):        
        activation = training_input
        activations = [training_input]
        zs = []   
        for b, w in zip(self.biases, self.weights):
            """ list to store all the Z VECTORS, layer by layer"""
            z = np.dot(w, activation)+b
            zs.append(z)            
            """ list to store all the ACTIVATIONS, layer by layer"""
            activation = sigmoid(z)
            activations.append(activation)
            """ list feedforward's result (be use in backpropagation later ) """
            input_effect_network= (activations,zs)
        return input_effect_network
           
    def backward_update(self,desired_output,input_effect_network):
        """INIT GRADIENT (nabla)"""
        nabla_b =self.init_nabla(self.biases)
        nabla_w =self.init_nabla(self.weights)
        """LAST layer OUTPUT local ERROR gradient (BACKPROPAGATION EQUATION I)"""
        delta_error=self.error_LAST_layer(desired_output,input_effect_network)               
        """update GRADIENT WEIGHT last layer (BACKPROPAGATION EQUATION IV)"""
        last_layer=1
        activations = input_effect_network[0]
        in_out_output_layer=(last_layer,activations,delta_error)
        nabla_w[-1] = self.update_nabla_w(nabla_w,in_out_output_layer)
        """update GRADIENT BIAS last layer (BACKPROPAGATION EQUATION III)"""
        nabla_b[-1] = delta_error
        #init zs for the next layer
        zs = input_effect_network[1]
        current_layer=(zs,delta_error)
        
        """  BACKPROPAGATE the ERROR to the Hidden layerS """
		# layer = 1 means the last layer of neurons
		# layer = 2 is 	  the second-last layer and so on.
        for layer in range(2, self.nbr_layers):             
            """PREVIOUS layer OUTPUT local error gradient (BACKPROPAGATION EQUATION II)"""
            delta_error = self.error_PREVIOUS_layer(layer,current_layer)
            """update GRADIENT WEIGHT previous layer (BACKPROPAGATION EQUATION IV)"""
            in_out_hidden_layer=(layer,activations,delta_error)
            nabla_w[-layer] = self.update_nabla_w(nabla_w,in_out_hidden_layer)			
            #update GRADIENT BIAS (BACKPROPAGATION EQUATION III)
            nabla_b[-layer] = delta_error          
        return (nabla_b, nabla_w)

    """BACKPROPAGATION EQUATION I :  ERROR in the output layer"""
    def error_LAST_layer(self,desired_output,input_effect_network):
        """ init activations and Z vectors """
        activations = input_effect_network[0]
        zs = input_effect_network[1]
        """ LAST layer (OUTPUT network) local error gradient """
        delta_error= self.cost_derivative(activations[-1], desired_output) * sigmoid_derivative(zs[-1])            
        return (delta_error)
    
    """BACKPROPAGATION EQUATION II : """
    """ERROR on the PREVIOUS LAYER based on current layer's error"""
    def error_PREVIOUS_layer(self,num_layer,current_layer):
        """ init activations and Z vectors """       
        zs = current_layer[0]
        delta_error = current_layer[1]
        """ calcul local ERROR """
        """ signe ``-`` (negative) :compute from LAST layer the FIRST """
        z = zs[-num_layer]
        sp = sigmoid_derivative(z)
        """TRANSPOSE = moving the error backward"""
        """turning a row vector into an ordinary (column) vector."""
        delta_error = np.dot(self.weights[-num_layer+1].transpose(), delta_error) * sp             
        return (delta_error)

    """BACKPROPAGATION EQUATION IV : cost change depend weight"""
    def update_nabla_w(self,nabla_w,in_out_layer):
        """ setting PARAMETERS"""
        nabla_w_layer = list(nabla_w)
        layer = in_out_layer[0]
        activations = in_out_layer[1]
        delta_error = in_out_layer[2]
        """ compute GRADIENT WEIGHT"""
        nabla_w_layer = np.dot(delta_error, activations[-layer-1].transpose())
        return nabla_w_layer
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
                             
    def cost_derivative(self, output_activations, desired_output):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-desired_output)  
      
#### Miscellaneous functions      
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z)) 

def sigmoid_derivative(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
