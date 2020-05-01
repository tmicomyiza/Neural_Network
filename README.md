# Artificial Neural Network

## AUTHOR:  Mico Micomyiza Theogene


## DESCRIPTION
Artificial Neural Network using Backpropagation algorithm
that categorizes data into three groups of plants.
Iris-setosa, Iris-versicolor, and Iris-virginica

# Structure

I created a class ANN which represents the artificial Neural Network
it has all methods to implement ANN.

Then, there are other functions outside the class.
Most of them are used to process the data from input file
and format them so that the algorithm understands them.

And, other functions such as pretty_print will reverse the information
from the format the algorithm wants to human readable format


## Training and Testing data
The database contains 150 inputs.

60% is for training
20% is for validation 
20% is for testing

The inputs are randomly allocatated to either group.

# The data base contains the following attributes:

* Sepal length in cm
* Sepal width in cm
* Petal length in cm
* Petal width in cm
* Class:
    * Iris Setosa
    * Iris Versicolour
    * Iris Virginica


## PYTHON FILES
ann.py


## REQUIREMENTS

Minimum Python requirements (earlier versions may work but have not been tested):

* Python 3 (3.7.6)
* numpy 
* termcolor

Other requirements

* Database file: ANN-Iris-data.txt the data provided. If you don't have it, i provided the file

* Note:
 make sure that the naming of the database file matches 'ANN-Iris-data.txt'.
 the program automatically loads information from this file. If it is named differently,
 you will get a runtime error

## HOW TO RUN IT

`python3 ann.py`

## Outputs

Once you run the algorithm, it will train itself using 60% of data in the database,
and validate during training using 20% of the data.

Then, it will automatically, run a test using the remaining 20%.

Once it finished running the tests, the results will be written on a file name
'results.txt'

'results.txt' has the following format
{input} {prediction} {actual classification}
*e.g ANN predicts [5.1 3.3 1.7 0.5] as Iris-setosa, and it is Iris-setosa

* the program will also print error as it the algorithm goes through trainning.
* when it finishes running, it will print the accuracy algorithm is classifying the
    the test given


# Reference
* https://towardsdatascience.com/everything-you-need-to-know-about-neural-networks-and-backpropagation-machine-learning-made-easy-e5285bc2be3a

* http://neuralnetworksanddeeplearning.com/chap2.html

* https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/






