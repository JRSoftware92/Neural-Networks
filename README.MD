# Artificial Neural Networks & Genetic Algorithms

## Sources 
http://www.ai-junkie.com/ga/intro/gat1.html
http://www.ai-junkie.com/ann/evolved/nnt1.html

Code adjusted from source code found from sites above. CURRENTLY UNTESTED.
Testing challenge to be devised.

## GeneticAlgorithm.java

Consists of two classes:
	- **Genome**: Basic data construct for genetic algorithms. Inner Class of GeneticAlgorithm. Consists of:
		- A *final double* fitness value.
		- A *Vector<Double>* containing weights for each input.
	- **GeneticAlgorithm**: Contains functionality for implementation of a genetic algorithm.

## NeuralNet.java

Consists of three classes:
	- **Neuron**: Basic data construct for simulating an artificial neuron. Inner class of NeuralNet. Consists of:
		- A *final int* size value, denoting the number of inputs in neuron.
		- A *Vector<Double>* containing weights for each input.
	- **NeuronLayer**: Basic data construct for simulating an artificial neuron layer. Inner class of NeuralNet. Consists of:
		- A *final int* size value, denoting the number of neurons found in the layer.
		- A *Vector<Neuron>*, containing all of the neurons found in the layer.
	- **NeuralNet**: Contains functionality for implementation of a neural network.
		