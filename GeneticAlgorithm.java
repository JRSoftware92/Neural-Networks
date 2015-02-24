package com.dev.model;

import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Vector;

/**
 * Contains genetic algorithm data.
 * @author John Riley
 * @see http://www.ai-junkie.com/ga/intro/gat1.html
 *
 */
public class GeneticAlgorithm {
	
	public static final double MAX_PERTURBATION = 1;
	public static final int MAX_ELITE_COPIES = 20;
	public static final int MAX_ELITE = 10;
	
	public final int populationSize;
	public final int genomeSize;	//num weights
	
	private double totalFitness;
	private double bestFitness;
	private double avgFitness;
	private double worstFitness;
	
	/*
	private int bestGenome;
	private int generationCounter;
	*/
	
	private final double mutationRate;		//Very low. .001
	private final double crossOverRate; 		//.7 is pretty good

	private Vector<Genome> population;
	
	//-----------------------Initialization-------------------------------------//
	
	public GeneticAlgorithm(int populationSize, int genomeSize, double mut, double cross){
		this.populationSize = populationSize;
		this.genomeSize = genomeSize;
		mutationRate = mut;
		crossOverRate = cross;
	}
	
	//---------------------------------Accessors-------------------------------//
	
	/**
	 * Returns the genome population of this genetic algorithm.
	 * @return
	 */
	public Vector<Genome> getGenomes(){ return population; }
	
	/**
	 * Returns the average fitness of a given genome if the genetic algorithm.
	 * @return
	 */
	public double avg(){ return avgFitness; }
	
	/**
	 * Returns the optimal fitness of all genomes in the genetic algorithm.
	 * @return
	 */
	public double best(){ return bestFitness; }
	
	//----------------------------------Functionality-----------------------------------------------------//
	
	/**
	 * Iterates over one generation of chromosomes, given an input population.
	 * @param pop
	 * @param rnd
	 */
	public void epoch(Vector<Genome> pop, Random rnd){
		Vector<Genome> newPop = new Vector<Genome>(pop.size());
		population = pop;
		resetValues();
		
		//for scaling and elitism
		//sort(population.firstElement(), population.lastElement());
		
		calculateValues();
		//Now to add a little elitism we shall add in some copies of the
		//fittest genomes. Make sure we add an EVEN number or the roulette
	    //wheel sampling will crash
		newPop = getNBest(MAX_ELITE, MAX_ELITE_COPIES, newPop);
		
		while(newPop.size() < populationSize){
			Genome a = getGenomeRoulette(rnd, totalFitness, population);
			Genome b = getGenomeRoulette(rnd, totalFitness, population);
			
			Vector<Double> kidA = new Vector<Double>(), kidB = new Vector<Double>();
			ArrayList<Vector<Double>> list = crossOver(a.weights, b.weights, kidA, kidB, rnd);
			kidA = list.get(0);
			kidB = list.get(1);
			
			kidA = mutate(kidA, rnd);
			kidB = mutate(kidB, rnd);
			
			//TODO Fitness may not work right...
			newPop.add(new Genome(a.fitness, kidA));
			newPop.add(new Genome(b.fitness, kidB));
		}
		
		population = newPop;
	}
	
	/**
	 * Resets the values of this algorithms fitness measurements.
	 */
	private void resetValues(){
		totalFitness = 0;
		avgFitness = 0;
		bestFitness = 0;
		worstFitness = Double.MAX_VALUE;
	}
	
	/**
	 * Calculates the values of this algorithms fitness measurements.
	 */
	private void calculateValues(){
		PriorityQueue<Genome> queue = new PriorityQueue<Genome>();
		Double sample;
		for(int i = 0; i < populationSize; i++){
			queue.add(population.get(i));
		}
		
		worstFitness = queue.poll().fitness;
		bestFitness = avgFitness = totalFitness = worstFitness;
		
		while(!queue.isEmpty()){
			sample = queue.remove().fitness;
			totalFitness += sample;
			if(queue.isEmpty()){
				bestFitness = sample;
				avgFitness = totalFitness / populationSize;
			}
		}
	}
	
	/**
	 * Returns a list of cross over vectors based on the given parents and children.
	 * @param a - First Parent
	 * @param b - Second Parent
	 * @param kidA - First Child
	 * @param kidB - Second Child
	 * @param rnd
	 * @return
	 */
	private ArrayList<Vector<Double>> crossOver(Vector<Double> a, Vector<Double> b, Vector<Double> kidA, Vector<Double> kidB, Random rnd){
		Vector<Double> newA = kidA, newB = kidB;
		
		if(rnd.nextFloat() > crossOverRate){
			newA = a;
			newB = b;
		}
		
		int point = rnd.nextInt(genomeSize - 1);
		for(int i = 0; i < point; i++){
			newA.add(a.get(i));
			newB.add(b.get(i));
		}
		for(int i = point; i < a.size(); i++){
			newA.add(b.get(i));
			newB.add(a.get(i));
		}
		ArrayList<Vector<Double>> vectors = new ArrayList<Vector<Double>>();
		vectors.add(newA); vectors.add(newB);
		
		return vectors;
	}
	
	/**
	 * Returns a mutation of the given genomes, based on a random value relative to this algorithms mutation rate.
	 * @param genomes
	 * @param rnd
	 * @return
	 */
	private Vector<Double> mutate(Vector<Double> genomes, Random rnd){
		Vector<Double> result = new Vector<Double>(genomes.size());
		for(int i = 0; i < genomes.size(); i++){
			if(rnd.nextFloat() < mutationRate){
				result.add(rnd.nextDouble() * MAX_PERTURBATION);
			}
			else{
				result.add(genomes.get(i));
			}
		}
		
		return result;
	}
	
	//---------------------Static Helper Methods-----------------------------------//
	
	/**
	 * Returns the N best genomes of the provided genome population.d
	 * @param nBest
	 * @param copies
	 * @param pop
	 * @return
	 */
	private static Vector<Genome> getNBest(int nBest, int copies, Vector<Genome> pop){
		Vector<Genome> result = pop;
		int size = pop.size();
		while(nBest-- > 0){
			for(int i = 0; i < copies; ++i){
				result.add(pop.get(size - 1 - nBest));
			}
		}
		
		return result;
	}
	
	/**
	 * Picks a genome using Roulette pick (chance of picking proportional to ratio of fitness to total.)
	 * @param rnd
	 * @param totalFitness
	 * @param population
	 * @return
	 */
	private static Genome getGenomeRoulette(Random rnd, double totalFitness, Vector<Genome> population){
		double r = rnd.nextDouble() * totalFitness;
		int size = population.size();
		Genome result = null;
		double total = 0;
		
		for(int i = 0; i < size; i++){
			total += population.get(i).fitness;
			
			if(total >= r){
				result = population.get(i);
				break;
			}
		}
		return result;
	}
	
	//--------------------------------Genome Class---------------------------------------//
	
	/**
	 * Contains single genome of a genetic algorithm.
	 * @author John Riley
	 *
	 */
	final static class Genome implements Comparable<Genome>{
		
		final double fitness;
		final Vector<Double> weights;
		
		/**
		 * Standard Constructor. Takes in a fitness value and a vector of weights.
		 * @param fitness
		 * @param weights
		 */
		Genome(double fitness, Vector<Double> weights){
			this.fitness = fitness;
			this.weights = weights;
		}

		public int compareTo(Genome other) {
			if(this.fitness < other.fitness){ return -1; }
			else if(this.fitness > other.fitness){ return 1; }
			else{ return 0; }
		}
	}
}
