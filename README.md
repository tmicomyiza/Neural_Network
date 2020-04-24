# Knapsack problem solver

## AUTHOR:  Mico Micomyiza Theogene


## DESCRIPTION
Knapsack problem solver using local search technique known as Genetic Algorithm.

We maximize value without exceeding the maximum weight of the backpack

    #Population
        it is a list of chromosomes

    # Chromosome:
        list of boxes


    # boxes: each box has 3 attributes
            1.  status: binary value which represents whether it is to be
                    packed or not

                    Note: 1 -> packed, 0 -> not packed

            2. weight

            3. value 

    



## PYTHON FILES
genetic_algorithm.py


## REQUIREMENTS

Minimum Python requirements (earlier versions may work but have not been tested):

* Python 3 (3.7.6)
* termcolor module for printing


## HOW TO RUN IT

`python3 genetic_algorithm.py`


after running the above command,
you will receive guidance on the use of the program on terminal.

    1. If you want, you can use terminal to insert inputs

    2. Otherwise, the program will randomly generate data to be used


* The program will print final results on the terminal. 

* It will also write information on a file named generations.txt which will
    contain all generations that we created in order to reach the solution





