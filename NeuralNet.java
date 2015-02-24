package com.dev.model;

import java.util.Random;
import java.util.Vector;

/**
 * Class encapsulating data of an artificial neural network. 
 * Designed to be extended by subclass to add further functionality.
 * @author John Riley
 * @see http://www.ai-junkie.com/ann/evolved/nnt1.html
 */
public abstract class NeuralNet {
	
	//TODO
	protected double BIAS = 1d;
	protected double ACTIVATION_RESPONSE = 1d;
	
	private final int inputs, outputs, hidden, neurons;
	
	protected final Vector<NeuronLayer> layers;
	
	//-------------------------Initialization-------------------------------------------//

	/**
	 * Random Constructor. Initializes a randomly initialized neural network.
	 * @param numInputs
	 * @param numOutputs
	 * @param numLayers
	 * @param numNeuronsPerLayer
	 * @param rnd
	 */
	public NeuralNet(int numInputs, int numOutputs, int numLayers, int numNeuronsPerLayer, Random rnd){
		inputs = numInputs;
		outputs = numOutputs;
		hidden = numLayers;
		neurons = numNeuronsPerLayer;
		layers = new Vector<NeuronLayer>(hidden);
		
		initialize(rnd);
	}
	
	/**
	 * Initializes the neural network given random inputs.
	 * @param rnd
	 */
	private void initialize(Random rnd){
		for(int i = 0; i < hidden; i++){		//hidden layers
			layers.add(new NeuronLayer(neurons, inputs, rnd));
		}
		
		//output layer
		layers.add(new NeuronLayer(outputs, neurons, rnd));
	}
	
	//------------------------------Accessors----------------------------------//
	
	/**
	 * Returns the weights of the given layer/neuron/input.
	 * @param layer
	 * @param neuron
	 * @param input
	 * @return
	 */
	public double getWeight(int layer, int neuron, int input){
		return layers.get(layer).neurons.get(neuron).weights.get(input);
	}
	
	/**
	 * Returns a vector containing all weights in the neural network
	 * @return
	 */
	public Vector<Double> getWeights(){
		Vector<Double> weights = new Vector<Double>();
		for(NeuronLayer layer : layers){
			for(Neuron neuron : layer.neurons){
				for(Double weight : neuron.weights){
					weights.add(weight);
				}
			}
		}
		return weights;
	}
	
	/**
	 * Returns the total number of weights within the neural network.
	 * @return
	 */
	public int numberOfWeights(){
		int count = 0;
		for(NeuronLayer layer : layers){
			for(Neuron neuron : layer.neurons){
				count += neuron.size;
			}
		}
		
		return count;
	}
	
	/**
	 * Return the number of inputs used in this neural network.
	 * @return
	 */
	public int numberOfInputs(){ return this.inputs; }
	
	/**
	 * Return the number of outputs used in this neural network.
	 * @return
	 */
	public int numberOfOutputs(){ return this.outputs; }
	
	/**
	 * Returns the number of hidden layers found in this neural network.
	 * @return
	 */
	public int numberOfLayers(){ return this.hidden; }
	
	/**
	 * Returns the number of neurons found in each hidden layer of this neural network.
	 * @return
	 */
	public int neuronsPerLayer(){ return this.neurons; }
	
	//---------------------------Mutators--------------------------------------//
	
	/**
	 * Places weight at given layer/neuron/input, returns true if successfully placed.
	 * @param layer
	 * @param neuron
	 * @param input
	 * @param weight
	 * @return
	 */
	public boolean putWeight(int layer, int neuron, int input, double weight){
		boolean flag = false;
		try{
			layers.get(layer).neurons.get(neuron).setWeight(input, weight);
			flag = true;
		}
		catch(NullPointerException e){
			flag = false;
		}
		catch(ArrayIndexOutOfBoundsException e){
			flag = false;
		}
		return flag;
	}
	
	/**
	 * Replaces all input weights in the network with the given vector.
	 * @param weights
	 * @return
	 */
	public boolean putWeights(Vector<Double> weights){
		boolean flag = false;
		try{
			for(int i = 0; i < hidden + 1; i++){
				for(int j = 0; j < layers.get(i).size; j++){
					for(int k = 0; k < layers.get(i).neurons.get(j).size; k++){
						layers.get(i).neurons.get(j).setWeight(k, weights.get(k));
					}
				}
			}
			flag = true;
		}
		catch(NullPointerException e){
			flag = false;
		}
		catch(ArrayIndexOutOfBoundsException e){
			flag = false;
		}
		
		return flag;
	}
	
	//--------------------------Functionality-----------------------------------//
	
	/**
	 * Calculates output vector of a given input vector.
	 * @param inputs - Input vector
	 * @return
	 */
	public Vector<Double> update(Vector<Double> inputs){
		Vector<Double> outputs = new Vector<Double>();
		int weight = 0;
		
		if(inputs.size() != this.inputs){		//exits if size doesn't match
			return null;
		}
		
		//each layer
		for(int i = 0; i < hidden + 1; i++){
			if(i > 0){
				inputs = outputs;
			}
			
			outputs.clear();
			weight = 0;
			
			//for each neuron, sum weights, apply sigmoid, get output
			for(int j = 0; j < this.neurons; j++){
				double net = 0;
				
				//sum of products of weights and inputs
				for(int k = 0; k < this.inputs - 1; k++){
					net += layers.get(i).neurons.get(j).weights.get(k) * inputs.get(weight++);
				}
				
				//add bias
				net += layers.get(i).neurons.get(j).weights.get(this.inputs - 1) * BIAS;
				
				outputs.add(applySigmoid(net, ACTIVATION_RESPONSE));
				weight = 0;
			}
		}
		
		return outputs;
	}
	
	//---------------------------Neuron Layer Class---------------------------------//
	
	/**
	 * Class encapsulating data for an artificial neuron layer
	 * @author John Riley
	 *
	 */
	final static class NeuronLayer {

		final int size;
		final Vector<Neuron> neurons;
		
		NeuronLayer(int size, int neuronSize, Random rnd){
			this.size = size;
			neurons = new Vector<Neuron>(size);
			
			for(int i = 0; i < size; i++){
				neurons.add(new Neuron(neuronSize, rnd));
			}
		}
		
		NeuronLayer(Vector<Neuron> neurons){
			this.size = neurons.size();
			this.neurons = neurons;
		}
	}
	
//----------------------------------Neuron Class----------------------------------------//
	
	/**
	 * Class encapsulating data for an artificial neuron.
	 * @author John Riley
	 *
	 */
	final static class Neuron {
		
		final int size;
		protected Vector<Double> weights;
		
		Neuron(int size, Random rnd){
			this.size = size;
			weights = new Vector<Double>(this.size + 1);
			
			//ADD +1 TO SIZE FOR BIAS??? (bias is threshold added to weight with a -1 sign modifier
			for(int i = 0; i < this.size + 1; i++){
				weights.add(rnd.nextDouble());	//random weight
			}
		}
		
		Neuron(Vector<Double> weights){
			this.size = weights.size();
			this.weights = weights;
		}
		
		/**
		 * Sets the weight of the input at i to the given value.
		 * @param i - index of the input.
		 * @param val - Value of the input weight.
		 * @throws ArrayIndexOutOfBoundsException if i < 0 || i > this.size
		 */
		protected void setWeight(int i, double val) throws ArrayIndexOutOfBoundsException{
			weights.remove(i);
			weights.add(i, val);
		}
	}
	
	/**
	 * Applies the sigmoid function to the given activation and response inputs.
	 * @param activation
	 * @param response
	 * @return
	 */
	private static double applySigmoid(double activation, double response){
		return 1 / (1 + Math.pow(Math.E, ((-1 * activation)/response)));
	}
}