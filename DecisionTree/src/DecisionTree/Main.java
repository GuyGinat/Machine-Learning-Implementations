package DecisionTree;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

import weka.core.Instance;
import weka.core.Instances;

public class Main {

	private static Node rootNode;

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
		
		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");
		
		double bestValidationError = Double.MAX_VALUE, currentError;
		int bestP_value = 0;
		
		//Constructing two trees using Entropy and Gini impurity measures
		DecisionTree DT = new DecisionTree();
		DT.buildTree(trainingCancer, false);
		double EntropyError = DT.calcAvgError(validationCancer);
		DT.PrintTree();
		DT.buildTree(trainingCancer, true);
		double GiniError = DT.calcAvgError(validationCancer);
		
		
		//Set our impurity depending on best validation error.
		if (GiniError < EntropyError) {
			DT.useGini = true;
		}
		//Build the tree according to the measure
		DT.buildTree(trainingCancer, DT.useGini);

		System.out.println("Validation error using Entropy: " + EntropyError);
		System.out.println("Validation error using Gini: " + GiniError);
		
		//For each p_value build tree and calculate errors.
		for (int i = 0; i < DT.p_value_map.length; i++) {
			DT.buildTree(trainingCancer, DT.useGini);
			System.out.println("----------------------------------------------------");
			System.out.println("Decision Tree with p_value of: " + DT.p_value_map[i]);
			DT.p_value++;
			System.out.println("The train error of the decision tree is " + DT.calcAvgError(trainingCancer));
			System.out.println("Max height on validation data: " + DT.treeHeight(DT.getRoot()));
			currentError = DT.calcAvgError(validationCancer);
			System.out.println("Average height on validation data: " + DT.avgHeight);
			System.out.println("The validation error of the decision tree is " + currentError);
			//Keep track of best error and the p_value that gives it.
			if (currentError < bestValidationError) {
				bestValidationError = currentError;
				bestP_value = i;
			}
		}
		
		System.out.println("----------------------------------------------------");
		System.out.println("Best validation error at p_value = " + DT.p_value_map[bestP_value]);
		DT.p_value = bestP_value;
		DT.buildTree(trainingCancer, DT.useGini);
		double testError = DT.calcAvgError(testingCancer);
		Node root = DT.getRoot();
		System.out.println("Test\terror\twith\tbest\ttree: " + testError);
		DT.PrintTree();
		
	}
}
