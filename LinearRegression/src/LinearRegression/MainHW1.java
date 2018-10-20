package LinearRegression;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class MainHW1 {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
		
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	
	
	public static void main(String[] args) throws Exception {
		//load data
		Instances training = loadData("wind_training.txt");
		Instances testing = loadData("wind_testing.txt");
		
		//find best alpha and build classifier with all attributes
		LinearRegression LR = new LinearRegression();
		LR.buildClassifier(training);
		
		double best_alpha = LR.getAlpha();
		System.out.println("The chosen alpha is: " + best_alpha);
		double trainingError = LR.calculateMSE(training);
		System.out.println("Training error with all features is: " + trainingError);
		double testingError = LR.calculateMSE(testing);
		System.out.println("Test error with all features is: " + testingError);
		
   		//build classifiers with all 3 attributes combinations
		Instances threeTrain = training; 	//Initialize an Instances object to hold three attributes each time.
		Instances bestThree = training;  	//An Instances object to contain the best three attributes considered so far.
		
		//Use the remover to single out three different attributes.
		Remove rm = new Remove();  
		//String used to tell which features are the three best (to be used in the Remove object).
		String finalInd = "";
		
		//keep track of your errors
		double currErr, minErr = Double.MAX_VALUE;
		int[] best = new int[3];
		for (int i = 0; i < 3; i++) {
			best[i] = 0;
		}
		
		//Loop through all available combinations of attributes
		for (int i = 1; i < training.numAttributes(); i++) {
			for (int j = i + 1; j < training.numAttributes(); j++) {
				for (int k = j + 1; k < training.numAttributes(); k++) {
					String ind = "" + i + "," + j + "," + k + ", " + training.numAttributes();
					rm.setAttributeIndices(ind);
					rm.setInvertSelection(true);
					rm.setInputFormat(training);
					threeTrain = Filter.useFilter(training, rm);
					LR.buildClassifier(threeTrain);
					
					currErr = LR.calculateMSE(threeTrain);
					
					if (currErr < minErr) {
						finalInd = ind;
						minErr = currErr;
						best[0] = i;
						best[1] = j;
						best[2] = k;
						bestThree = threeTrain;
					}
					System.out.println(currErr); /*
					threeTrain.attribute(0).name() + ", " + 
							  		   threeTrain.attribute(1).name() + ", " +
							  		   threeTrain.attribute(2).name() + " Error is: " + */
				}
			}
		}
		System.out.println("Training error of the best three features is:  " + training.attribute(best[0] - 1).name() + ", "
														     				 + training.attribute(best[1] - 1).name() + ", "
														     				 + training.attribute(best[2] - 1).name() + " Error is: "+ minErr);
		
		//Since we have allready found the best three features, use them
		//with the same LinearRegression object initialized before
		LR.buildClassifier(bestThree);
		rm.setAttributeIndices(finalInd);
		rm.setInputFormat(testing);
		rm.setInvertSelection(true);
		Instances threeTestError = Filter.useFilter(testing, rm);
		
		double testError = LR.calculateMSE(threeTestError);
		
		System.out.println("Test error of the best three features is: " + testing.attribute(best[0] - 1).name() + ", "
																		+ testing.attribute(best[1] - 1).name() + ", "
																		+ testing.attribute(best[2] - 1).name() + " Error is: "+ testError);
	}
}