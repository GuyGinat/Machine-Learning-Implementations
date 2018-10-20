package Knn;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

import Knn.Knn.DistanceCheck;

class DistanceCalculator {
    /**
    * We leave it up to you whether you want the distance method to get all relevant
    * parameters(lp, efficient, etc..) or have it has a class variables.
    */
    public static double distance(Instance one, Instance two, double p, DistanceCheck distanceCheck, double maxDistance) {
    	if (distanceCheck == DistanceCheck.Regular) {
    		if (p > 3) {
        		return lInfinityDistance(one, two);
        	} else {
        		return lpDistance(one, two, p); 
        	}
    	} else {
    		if (p > 3) {
        		return efficientLInfinityDistance(one, two, maxDistance);
    		} else {
    			return efficientLpDistance(one, two, p, maxDistance);
    		}
    	}
    }

    /**
     * Returns the Lp distance between 2 instances.
     * @param one
     * @param two
     * @param p = the dimension of the instances (=vectors)
     */
    private static double lpDistance(Instance one, Instance two, double p) {
    	double sumOfDifferencesPowP = 0.0;
    	for (int i = 0; i < (one.numAttributes() - 1); i++) {
    		sumOfDifferencesPowP += Math.pow(Math.abs(one.value(i) - two.value(i)), p);
    	}
        return Math.pow(sumOfDifferencesPowP, 1 / p);
    }

    /**
     * Returns the L infinity distance between 2 instances.
     * @param one
     * @param two
     * @return
     */
    private static double lInfinityDistance(Instance one, Instance two) {
    	double maxDifference = 0.0, temp = 0.0;
    	for (int i = 0; i < (one.numAttributes() - 1); i++) {
    		temp = Math.abs(one.value(i) - two.value(i));
    		if (maxDifference < temp) maxDifference = temp; 
    	}
        return maxDifference;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @param p = the dimension of the instances (=vectors)
     * @return
     */
    private static double efficientLpDistance(Instance one, Instance two, double p, double maxDistance) {
    	double sumOfDifferencesPowP = 0.0;
    	for (int i = 0; i < (one.numAttributes() - 1); i++) {
    		sumOfDifferencesPowP += Math.pow(one.value(i) - two.value(i), p);
    		if (sumOfDifferencesPowP > Math.pow(maxDistance, p)) return Math.pow(sumOfDifferencesPowP, (1/p));
    	}
        return Math.pow(sumOfDifferencesPowP, 1 / p);
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private static double efficientLInfinityDistance(Instance one, Instance two, double maxDistance) {
    	double maxDifference = 0.0, temp = 0.0;
    	for (int i = 0; i < (one.numAttributes() - 1); i++) {
    		temp = Math.abs(one.value(i) - two.value(i));
    		if (maxDifference < temp) maxDifference = temp; 
    		if (maxDifference < maxDistance) return maxDifference;
    	}
        return maxDifference;
    }
}

public class Knn implements Classifier {

    public enum DistanceCheck{Regular, Efficient};
    public enum weightingScheme{Unweighted, Weighted};
    
    private Instances trainingInstances;
    private int k;
    private double p;
    private double maxDistance;
    private weightingScheme scheme;
    private DistanceCheck distanceCheck;
    public double averageTime = 0.0, totalTime = 0.0;
    
    public Knn(Instances instances, int k, double p, weightingScheme scheme, DistanceCheck distanceCheck) {
    	this.trainingInstances = instances;
    	this.k = k;
    	this.p = p;
    	this.scheme = scheme;
    	this.distanceCheck = distanceCheck;
    	
    }
    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for 
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {

    }
    
    /**
     * wrapper for distance
     * @param one
     * @param two
     * @return
     */
    public double distance(Instance one, Instance two) {
    	return DistanceCalculator.distance(one, two, p, distanceCheck, maxDistance);
    }
    
    /**
     * Returns the knn prediction on the given instance.
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
    	PriorityQueue<Neighbour> nearestNeighbours = findNearestNeighbors(instance);
    	if (scheme == weightingScheme.Unweighted) {
    		return getAverageValue(nearestNeighbours);
    	} else return getWeightedAverageValue(instance, nearestNeighbours); 
    }

    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all insatnces.
     * @param insatnces
     * @return
     */
    public double calcAvgError (Instances instances){
    	double sum = 0;
        for (Instance x : instances) {
        	sum += Math.abs(regressionPrediction(x) - x.classValue());
        }
        return sum / (double) instances.numInstances();
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     * updates "totalTime" and "averageTime" fields of the object
     * @param instances Instances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances instances, int num_of_folds) {
    	double errorsSum = 0.0, startingTime, totalTime = 0;
    	for (int i = 0; i < num_of_folds; i++) {
    		trainingInstances = instances.trainCV(num_of_folds, i);
    		startingTime = System.nanoTime();
    		errorsSum += calcAvgError (instances.testCV(num_of_folds, i));
    		totalTime += (System.nanoTime() - startingTime);
    	}
    	this.totalTime = totalTime;
    	this.averageTime = totalTime / num_of_folds;
    	return (errorsSum / num_of_folds);
    }

    /**
     * Finds the k nearest neighbors.
     * @param instance
     */
    public PriorityQueue<Neighbour> findNearestNeighbors(Instance instance) {
    	PriorityQueue<Neighbour> nn = new PriorityQueue<>(k);
    	//Initialize our priority queue with infinite values
    	for (int i = 0; i < k; i++) {
    		nn.add(new Neighbour(Double.MAX_VALUE, null));
    	}
    	//Check the distance between each instance in our data to the desired instance
    	// and insert instances closer to our instance
    	for (Instance dataInstance : trainingInstances) {
    		Neighbour currNeighbour = new Neighbour(distance(instance, dataInstance), dataInstance);
    		if (currNeighbour.distance < nn.peek().distance) {
    			nn.remove();
    			nn.add(currNeighbour);
    		}
    	}
    	return nn;
    }

    /**
     * Cacluates the average value of the given elements in the collection.
     * @param
     * @return
     */
    public double getAverageValue (PriorityQueue<Neighbour> nearestNeighbours) {
    	double sum = 0.0;
        for (Neighbour n : nearestNeighbours) {
        	sum += n.instance.classValue();
        }
        return sum / (double) nearestNeighbours.size();
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     * @return
     */
    public double getWeightedAverageValue(Instance instance, PriorityQueue<Neighbour> nearestNeighbours) {
        double numSum = 0.0, denumSum = 0.0, Wi = 0.0;
        for (Neighbour n : nearestNeighbours) {
        	double dis = distance(instance, n.instance);
        	if(dis == 0) {
        		return n.instance.classValue();
        	}
        	Wi = 1 / Math.pow(dis, 2);
        	numSum += Wi * n.instance.classValue();
        	denumSum += Wi;
        }
        return numSum / denumSum;
    }


    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        // TODO Auto-generated method stub - You can ignore.
        return 0.0;
    }
}


/**
 * Neighbour class uesd to compare between instances 
 * to check in the priority queue
 * @author Guy Ginat
 * 
 */
class Neighbour implements Comparable<Neighbour> {
	double distance;
	Instance instance;
	public Neighbour(double distance,Instance instance) {
		this.distance = distance;
		this.instance = instance;
	}
	
	/**
	 * We override compareTo method with reverse order
	 * since we want our priority queue to act as a minHeap.
	 */
	@Override
	public int compareTo(Neighbour other) {
		return Double.compare(other.distance, this.distance);
	}
}