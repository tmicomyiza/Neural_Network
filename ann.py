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



    # method to implement back propagations
    def back_propagate(self, error):

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

        return error





    # calculates and updates the weights 
    def gradient_descent(self, learning_rate):
        for i in range (len (self.weights)):
            weights = self.weights[i]
            # print("Original w{} {}".format(i,weights))

            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate
            # print("updated w{} {}".format(i, weights))
            self.weights[i] = weights



    # method to train the algorithm
    def train(self, inputs, targets, epochs, learning_rate):

        previous_error = 0

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

                previous_error = sum_error

                sum_error += self.mse(target, output)


  
            
            #report error 
            print("Error: {} at epoch {}".format(sum_error / len(inputs), i))

            if previous_error > sum_error: 
                break 

    def mse(self, target, output):
        return np.average((target - output)**2)

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)


    #activation function
    def sigmoid(self, p):
        return 1 / (1 + np.exp(-p))


def print_helper(data):
    temp = []
    for item in data:
        if item == 0:
            temp.append("Iris-setosa")

        elif item == 1:
            temp.append("Iris-versicolor")

        else:
            temp.append("Iris-virginica")


    return temp


def prety_print(ann, output, inputs, targets):
    filename = open("results.txt", "w")

    accuracy = 0
    #format targets into strings
    targets = print_helper(targets)
    current = ""


    for i in range (len(targets)):
        prediciton = output[i]
        if prediciton > 0.7:
            filename.write("ANN predicts {} as Iris-versicolor, and it is {}\n".format(inputs[i], targets[i]))
            current = "Iris-versicolor"
        
        elif prediciton > 0.3:
            filename.write("ANN predicts {} as Iris-virginica, and it is {}\n".format(inputs[i], targets[i]))
            current = "Iris-virginica"

        else:
            filename.write("ANN predicts {} as Iris-setosa, and it is {}\n".format(inputs[i], targets[i]))
            current = "Iris-setosa"


        if targets[i] == current:
            accuracy += 1

    filename.close()

    print(colored("Accuracy is {}".format(accuracy/ len(targets)), "green"))

def training_data(ann):
    infile = open("ANN-Iris-data.txt", 'r')
    Iris_data = infile.readlines()


    Iris_Setosa  = []  
    Iris_versicolor = []  
    Iris_virginica = []

    # systematically categorize the input into three parts
    # Iris-Setosa, Iris-versicolor, and Iris-virginica
    for i,line in enumerate(Iris_data):
        if i < 50:
            Iris_Setosa.append(line.split(","))
        
        elif i < 100:
            Iris_versicolor.append(line.split(","))

        else:
            Iris_virginica.append(line.split(","))


    # randomly select testing data with is 20 % from each group
    testing_data = select_10(Iris_Setosa) + select_10(Iris_versicolor) + select_10 (Iris_virginica)

    # randomly select validation data with is 20 % from each group
    validation_data = select_10(Iris_Setosa) + select_10(Iris_versicolor) + select_10 (Iris_virginica)

    # rest is training data
    training_data = Iris_Setosa + Iris_versicolor + Iris_virginica


    infile.close()

    process_data_instatiate (ann, training_data, validation_data, testing_data)



def process_data_instatiate(ann, train_data, validation_data, test_data):

    train = transform(train_data)
    train_inputs = train[0]
    train_targets = train[1]

    test = transform(test_data)
    test_inputs = test[0]
    test_targets = test[1]

    valid = transform(validation_data)
    valid_inputs = valid[0]
    valid_targets = valid[1]

    # run actual program
    run(ann, train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets)




# returns a tuple of two lists
def transform (data):
    # Transform train_data into a format the algorithm understand
    train_inputs = []
    train_targets = []
    i = 0
    for item in data:
        outputs = str(item.pop())[:-1]
        inputs = list(map(float, item))
        train_inputs.append(inputs)

        if outputs == 'Iris-setosa':
            train_targets.append(0)
            i += 1

        elif outputs == 'Iris-versicolor':
            train_targets.append(1)
            i += 1

        elif outputs == 'Iris-virginica':
            train_targets.append(0.5)
            i += 1

   
    return [train_inputs, train_targets]





# randomly selects 10 elements from the given list
# and returns a list containing the selected elements
def select_10 (data):
    selected = []
    for i in range (10):
        n = len(data)
        index = np.random.randint(0, n)
        selected.append(data[index])

        data.pop(index)

    return selected





def run(ann, train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets):

    inputs = np.array(train_inputs)
    targets = np.array(train_targets)

    # train
    ann.train(inputs, targets, 1000, 0.1)

    validation_input = np.array(valid_inputs)
    validation_targets = np.array(valid_targets)

    #forward propagate
    ann.forward_propagate(validation_input)

    #prety_print(output, test_input)
    testing(ann, np.array(test_inputs), np.array(test_targets))




def testing(ann, test_inputs, test_targets):
    
    output = ann.forward_propagate(test_inputs)

    prety_print(ann, output, test_inputs, test_targets)

    status = input(colored("Do you want to provide manual tests? (y/n) ", "green"))


    # manual testing
    if status.lower() == 'y':
        inputs = input(colored("Enter: Sepal length,Sepal width, Petal length, Petal width separated by space: ", "green"))
        inputs = inputs.split()

        inputs = list(map(float, inputs))
        print(inputs)

        target = input(colored("What is the output? (Iris-setosa, Iris-versicolor, or Iris-virginica)","green"))

        if target == "Iris-virginica":
            testing(ann, np.array([inputs]), np.array([0.5]))

        elif target == "Iris-setosa":
            testing(ann, np.array([inputs]), np.array([0]))

        elif target == "Iris-versicolor":
            testing(ann, np.array([inputs]), np.array([1]))

        else:
            print(colored("Error: Invalid input {}".format(target), "red"))



if __name__ == "__main__":
    # create ANN
    ann = ANN(4, 1, [5])

    # this will read in data from database file and then run the algorithm
    training_data(ann)
