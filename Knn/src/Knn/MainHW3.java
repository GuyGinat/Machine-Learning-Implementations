package Knn;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import Knn.Knn.DistanceCheck;
import Knn.Knn.weightingScheme;
import weka.core.Instances;

public class Main {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main(String[] args) throws Exception {
		
		Instances data = loadData("auto_price.txt");
		Knn knn;
		int minK = 0;
		String majorityFun = "";
		double minP = 0.0, CrossValidationError = Double.MAX_VALUE, minError = Double.MAX_VALUE;
		data.randomize(new Random());
		int [] num_of_folds = new int [] {data.numInstances(), 50, 10, 5, 3};	
		double averageTime, totalTime;
		
		//unscaled data
        for (int k = 1; k <= 20; k++) {
        	for (int p = 1; p <= 4; p++) {
        		//checking for scheme = Weighted
        		knn = new Knn (data, k, p, weightingScheme.Weighted, DistanceCheck.Regular);
        		CrossValidationError = knn.crossValidationError(data, 10);
        		if (minError > CrossValidationError) {
        			minError = CrossValidationError;
        			minK = k;
        			minP = (double) p;
        			majorityFun = "weighted";
        		}
        		//checking for scheme = Unweighted = uniform
        		knn = new Knn (data, k, p, weightingScheme.Unweighted, DistanceCheck.Regular);
        		CrossValidationError = knn.crossValidationError(data, 10);
        		if (minError > CrossValidationError) {
        			minError = CrossValidationError;
        			minK = k;
        			minP = (double) p;
        			majorityFun = "uniform";
        		}
        	}
        }
        					 
        System.out.println("-----------------------------" + "\n" + "Results for original dataset: " + "\n" + "-----------------------------" +
        "\n" + "Cross validation error with K = " + minK + ", lp = " + minP + ", majority function = " + 
        majorityFun + " for auto_price data is: " + minError);
        
        //scaled data
        data = FeatureScaler.scaleData(data);
        CrossValidationError = Double.MAX_VALUE;
        minError = Double.MAX_VALUE;
        for (int k = 1; k <= 20; k++) {
        	for (int p = 1; p <= 4; p++) {
        		//checking for scheme = Weighted
        		knn = new Knn (data, k, p, weightingScheme.Weighted, DistanceCheck.Regular);
        		CrossValidationError = knn.crossValidationError(data, 10);
        		if (minError > CrossValidationError) {
        			minError = CrossValidationError;
        			minK = k;
        			minP = (double) p;
        			majorityFun = "weighted";
        		}
        		//checking for scheme = Unweighted = uniform
        		knn = new Knn (data, k, p, weightingScheme.Unweighted, DistanceCheck.Regular);
        		CrossValidationError = knn.crossValidationError(data, 10);
        		if (minError > CrossValidationError) {
        			minError = CrossValidationError;
        			minK = k;
        			minP = (double) p;
        			majorityFun = "uniform";
        		}
        	}
        }
        					 
        System.out.println("\n-----------------------------" + "\n" + "Results for scaled dataset: " + "\n" + "-----------------------------" +
        "\n" + "Cross validation error with K = " + minK + ", lp = " + minP + ", majority function = " + 
        majorityFun + " for auto_price data is: " + minError);
        
        //in case minimum validation error was accepted from weighted scheme
        if (majorityFun.equals("weighted")) {
        	for (int i = 0; i < num_of_folds.length; i++) {
        		System.out.println("\n----------------------------\n Results for " + num_of_folds[i] + " folds: \n ---------------------------");
        		//regular distance check
        		knn = new Knn (data, minK, minP, weightingScheme.Weighted, DistanceCheck.Regular);
        		minError = knn.crossValidationError(data, num_of_folds[i]);
        		averageTime = knn.averageTime; 
        		totalTime = knn.totalTime;
        		System.out.println("Cross validation error of regular knn on auto_price dataset is " + minError
        							+ " and the average elapsed time is " + averageTime
                					+ "\nThe total elapsed time is: " + totalTime );
        		//efficient distance check
        		knn = new Knn (data, minK, minP, weightingScheme.Weighted, DistanceCheck.Efficient);
        		minError = knn.crossValidationError(data, num_of_folds[i]);
        		averageTime = knn.averageTime; 
        		totalTime = knn.totalTime;
        		System.out.println("\nCross validation error of efficient knn on auto_price dataset is " + minError
        							+ " and the average elapsed time is " + averageTime
                					+ "\nThe total elapsed time is: " + totalTime );
        	}
        }
      //in case minimum validation error was accepted from unweighted scheme
        else {
        	for (int i = 0; i < num_of_folds.length; i++) {
        		System.out.println("\n----------------------------\n Results for " + num_of_folds[i] + " folds: \n ---------------------------");
        		//regular distance check
        		knn = new Knn (data, minK, minP, weightingScheme.Unweighted, DistanceCheck.Regular);
        		minError = knn.crossValidationError(data, num_of_folds[i]);
        		averageTime = knn.averageTime; 
        		totalTime = knn.totalTime;
        		System.out.println("Cross validation error of regular knn on auto_price dataset is " + minError
        							+ " and the average elapsed time is " + averageTime
                					+ "\nThe total elapsed time is: " + totalTime );
        		//efficient distance check
        		knn = new Knn (data, minK, minP, weightingScheme.Unweighted, DistanceCheck.Efficient);
        		minError = knn.crossValidationError(data, num_of_folds[i]);
        		averageTime = knn.averageTime; 
        		totalTime = knn.totalTime;
        		System.out.println("\nCross validation error of efficient knn on auto_price dataset is " + minError
        							+ " and the average elapsed time is " + averageTime
                					+ "\nThe total elapsed time is: " + totalTime );
        	}
        }
	}
}