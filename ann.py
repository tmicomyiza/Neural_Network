import numpy as np 
from termcolor import colored


class ANN:


    def __init__(self, num_inputs=4 , num_outputs=1, hidden=[3, 5]):
        self.num_inputs = num_inputs   # dimension of input
        self.num_outputs = num_outputs # dimension of output
        self.hidden = hidden # number of hidden layers


        layers = [self.num_inputs] + self.hidden + [self.num_outputs]


        # initiate random weights
        self.weights = []

        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1])

            self.weights.append(w) # list of weight matrices, len =  n_layers - 1


        # initiate activations
        self.activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])

            self.activations.append(a)

        
        # initiate derivatives
        self.derivatives = []
        for i in range(len(layers) - 1):
            a = np.zeros((layers[i], layers[i + 1]))

            self.derivatives.append(a)

        


    # receives inputs
    # returns activations
    def forward_propagate(self, inputs):

        # the first layer, is the input itself
        activations = inputs
        self.activations[0] = inputs


        for i, w in enumerate(self.weights):
            # multiplication of input layer and the next layer
            net_inputs = np.dot(activations, w)

            # calculate activations
            activations = self.sigmoid(net_inputs)

            # save teh activations
            self.activations[i + 1] = activations 
        
        # returns the output layer. which is the last layer
        return activations


    def back_propagate(self, error, verbose=False):

        # start from the output layer to input layer
        for i in reversed(range(len(self.derivatives))):
            activation = self.activations[i + 1]
            delta = error * self.sigmoid_derivative(activation)
            # find transpose of the delta matrix
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            curr_activation = self.activations[i]
            # find transpose of the activation matrix
            curr_activation_trans = curr_activation.reshape(curr_activation.shape[0], - 1)

            self.derivatives[i] = np.dot(curr_activation_trans, delta_reshaped)

            # incrementing the error as we loop through layers
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for w{}: {}".format(i, self.derivatives[i]))

        return error


    def gradient_descent(self, learning_rate):
        for i in range (len (self.weights)):
            weights = self.weights[i]
            # print("Original w{} {}".format(i,weights))

            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate
            # print("updated w{} {}".format(i, weights))
            self.weights[i] = weights


    def train(self, inputs, targets, epochs, learning_rate):

        for i in range (epochs):
            sum_error = 0
            # iterate through inputs and targests
            for (input, target) in zip(inputs, targets):

                 # perform forward propagation
                output = self.forward_propagate(input)

                # calculate error 
                error = target - output

                # back propagate
                self.back_propagate(error)

                # learning rate
                self.gradient_descent(learning_rate=1)

                sum_error += self.mse(target, output)
            
            #report error 
            # print("Error: {} at epoch {}".format(sum_error / len(inputs), i))

    def mse(self, target, output):
        return np.average((target - output)**2)

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)


    #activation function
    def sigmoid(self, p):
        return 1 / (1 + np.exp(-p))




def prety_print(output, inputs):

    for i in range (len(output)):
        prediciton = output[i]
        if prediciton > 0.5:
            print(colored("{} is Iris-Versicolor".format(inputs[i]), "green"))
        else:
            print(colored("{} is Iris-setosa".format(inputs[i]),"green"))

        print(colored ("==================", "blue"))

if __name__ == "__main__":
    # create ANN
    ann = ANN(4, 1, [5])

    # some inputs

    inputs = np.array ([[5.1,3.5,1.4,0.2],
                [4.9,3.0,1.4,0.2],
                [5.7,4.4,1.5,0.4],
                [5.1,3.5,1.4,0.3],
                [5.2,4.1,1.5,0.1],
                [5.1,3.3,1.7,0.5],
                [7.0,3.2,4.7,1.4],
                [6.4,3.2,4.5,1.5],
                [5.5,2.3,4.0,1.3],
                [4.9,2.4,3.3,1.0],
                [6.1,2.9,4.7,1.4],
                [5.9,3.2,4.8,1.8]])

    targets = np.array([[0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1]])
    
    # train
    ann.train(inputs, targets, 1000, 0.1)


    #testing
    test_input = np.array([[5.0,3.3,1.4,0.2],
                            [5.9,3.0,4.2,1.5],
                            [7.7,2.8,6.7,2.0]
                            ])
    test_target = np.array([[0],[1],[1]])

    output = ann.forward_propagate(test_input)

    prety_print(output, test_input)
    # print("\n \n \n")
    # print("testing results: ")
    # print("input is {}".format(test_input))
    # print("it returned {}".format(output))
    # print("it should be {}".format(test_target))