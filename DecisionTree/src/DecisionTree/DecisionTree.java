package DecisionTree;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.gui.Main.ChildFrameMDI;



class Node {
	List<Node> children;
	Node parent;
	//What attribute sent this instance up the tree
	int attributeIndex;
	//What classification do we assign to the given node 
	double returnValue;
	//All instances will be removed, used for building the tree
	Instances instances;
	//What value from the parent best attribute sent this instance up the tree
	int brancedFromValue = -1;
	
	/**
	 * Constructor for a new node.
	 * 
	 * @param parent - Set the Node parent.
	 * @param attributeIndex - By what attribute the node is to be split.
	 * @param instances - Used only for construction and building of the tree.
	 * @param brancedFromValue - What value of the spliting attribute sent this instance up the tree.
	 */
	public Node (Node parent, int attributeIndex, Instances instances, int brancedFromValue) {
		this.children = new ArrayList<Node>();
		this.parent = parent;
		this.attributeIndex = attributeIndex;
		this.instances = instances;
		this.brancedFromValue = brancedFromValue;
		this.returnValue = parent.returnValue;
	}
	
	/**
	 * Empty constructor
	 */
	public Node() {
		
	}
} 

public class DecisionTree implements Classifier {
	private final static double[][] CHI_SQUARE_DISTRIBUTION = {
			//p: 1,  0.75,   0.5,  0.25,  0.05, 0.005
				{0, 0.102, 0.455, 1.323, 3.841, 7.879}, 
				{0, 0.575, 1.386, 2.773, 5.991, 10.597},
				{0, 1.213, 2.366, 4.108, 7.815, 12.838},
				{0, 1.923, 3.357, 5.385, 9.488, 14.860}, 
				{0, 2.675, 4.351, 6.626, 11.070, 16.750},
				{0, 3.455, 5.348, 7.841, 12.592, 18.548},
				{0, 4.255, 6.346, 9.037, 14.067, 20.278},
				{0, 5.071, 7.344, 10.219, 15.507, 21.955}, 
				{0, 5.899, 8.343, 11.389, 16.919, 23.589}, 
				{0, 6.737, 9.342, 12.549, 18.307, 25.188}, 
				{0, 7.584, 10.341, 13.701, 19.675, 26.757} 
	};
	/* p_value recieves the following values:
	 * p_value = 0 -> 1
	 * p_value = 1 -> 0.75
	 * p_value = 2 -> 0.5
	 * p_value = 3 -> 0.25
	 * p_value = 4 -> 0.05
	 * p_value = 5 -> 0.005
	 */
	static double[] p_value_map = {1, 0.75, 0.5, 0.25, 0.05, 0.005};
	//Current p_value index
	public static int p_value;
	public  static double avgHeight;
	private static Node rootNode;
	public boolean useGini = false;
	public static double sumOfInstancesHeight = 0;
	
	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
	
	/**
	 * Builds a decision tree from the training data.
	 * buildClassifier is separated from buildTree in order to allow us to do extra
	 * preprocessing before calling buildTree method or post processing after.
	 * @param arg0 - Instances object.
	 */
	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		p_value = 0;
	}
	
	/**
	 * Builds the decision tree on given data set using a queue algorithm.
	 * @param trainingData
	 * @param accordingToGini
	 */
	public static void buildTree(Instances trainingData, boolean accordingToGini) {
		//Initialize our Q and root node
		Queue<Node> q = new LinkedList<>();
		rootNode = new Node();
		rootNode.parent = null;
		rootNode.instances = trainingData;
		rootNode.children = new ArrayList<>();
		rootNode.returnValue = 1;
		q.add(rootNode);
		//Iterate as long as we have nodes in our Q
		while (!q.isEmpty()) {
			Node currentNode = q.peek();
			setClassification(currentNode);
			if (currentNode.instances.numInstances() == 0) {
				q.remove(currentNode);
				continue;
			}
			//If the node is perfectly classified extract it from the Q
			double classification = currentNode.instances.instance(0).classValue();
			boolean isMonochrome = true;
			for (Instance instance : currentNode.instances) {
				if (instance.classValue() != classification) {
					isMonochrome = false;
					//no need for further checking, node isn`t monochromatic
					break;
				}
			}
			if (isMonochrome) {
				currentNode.returnValue = classification;
				q.remove(currentNode);
				continue;
			}
			//Otherwise, we want to find the spliting attribute
			else {
				int bestAttribute = 0;
				double bestGain = 0, tempGain;
				// Find best gain (check per attribute)
				for (int i = 0; i < currentNode.instances.numAttributes() - 1; i++) {
					tempGain = calcGain(currentNode.instances, i, accordingToGini);
					if (tempGain > bestGain) {
						bestGain = tempGain;
						bestAttribute = i;
						currentNode.attributeIndex = i;
					}
				}
				//If best gain is 0 we reached a pure node.
				if (bestGain == 0) {
					q.remove(currentNode);
					continue;
				}
				
				//Split node to decendents (by the amount of different values of the spliting class)
				for (int i = 0; i < currentNode.instances.attribute(bestAttribute).numValues(); i++) {
					Instances blank = new Instances(trainingData, 0);
					Node child = new Node(currentNode, i, blank, i);
					currentNode.children.add(child);
				}
				//Move instances into related new nodes, remove them for parent.
				for (int i = 0; i < currentNode.instances.numInstances(); i++) {
					Instance instance = currentNode.instances.instance(i); 
					double valueOfAttribute = instance.value(bestAttribute);
					currentNode.children.get((int)valueOfAttribute).instances.add(instance);
				}
				//Add to our Q all children of current node that still have instances to examine.
				int df = 0;
				for (int i = 0; i < currentNode.children.size(); i++) {
					if (currentNode.children.get(i).instances.numInstances() != 0) {
						df++;
						q.add(currentNode.children.get(i));
					}
				}
				//Check for ChiSquare pruning depending on our current p_value
				double chiSquare = calcChiSquare(currentNode.instances, bestAttribute);
				if (chiSquare < CHI_SQUARE_DISTRIBUTION[df - 2][p_value]) {
					currentNode.children = new ArrayList<>();
					q.remove(currentNode);
					continue;
				}
				
				//Check for nodes that did not recieve instances and delete them from our children list.
				int numChildren = currentNode.children.size();
				for (int i = 0; i < numChildren; i++) {
					if (currentNode.children.get(i).instances.numInstances() == 0) {
						currentNode.children.remove(i);
						numChildren--;
						i--;
					}
				}
				//Overwrite all instances in our current node and remove it from the Q.
				currentNode.instances = new Instances(trainingData, 0);
				q.remove(currentNode);
			}
		}
	}

	
    /**
     * Return the classification of the instance.
     * @param instance
     * @return double number, 0 or 1, represent the classified class.
     */
    @Override
    public double classifyInstance(Instance instance) {
    	int n = 0;
    	Node nodeIterator = rootNode;
    	boolean childToMoveToFound;
    	while (nodeIterator.children.size() != 0) {
    		childToMoveToFound = false;
    		for (Node c : nodeIterator.children) {
    			if (c.brancedFromValue == instance.value(nodeIterator.attributeIndex)) {
    				nodeIterator = c;
    				n++;
    				childToMoveToFound = true;
    				break;
    			}
    		}
    		if (!childToMoveToFound) {
    			sumOfInstancesHeight += n;
    			return nodeIterator.returnValue;
    		}
    	}
    	sumOfInstancesHeight += n;
    	return nodeIterator.returnValue;
    }
    
	/**
	 * Calculate the average error on a given instances set (could be the training, test or validation set).
	 * The average error is the total number of classification mistakes on
	 * the input instances set divided by the number of instances in the input set.
	 * 
	 * @param data - the given dataset to calculate the avarge error 
	 * @return AvarageError
	 */
	public double calcAvgError(Instances data) {
		double errors = 0;
		for (Instance instance : data) {
			errors += Math.abs(classifyInstance(instance) - instance.classValue());
		}
		//Calculate the avarage height after we summed up
		//all the heights of new instances in the new data.  
		avgHeight = sumOfInstancesHeight / data.numInstances();
		//Zero the sum of heights for future data.
		sumOfInstancesHeight = 0;
		return errors / (double)data.numInstances();
	}
	
	/**
	 * calculates the gain (giniGain or informationGain depending on the impurity measure) 
	 * of splitting the input data according to the attribute.
	 * @param trainingSubset
	 * @param attributeIndex = the attribute to check
	 * @return
	 */
	public static double calcGain(Instances trainingSubset, int attIndex, boolean accordingToGini) {
		double sum = 0;
		Instances Sv;
		//treating data as array for comfort
		double [][] trainingSubsetAsNumbers = dataAsArray(trainingSubset);
		//for a single attribute, go over all values
		for (int i = 0; i < trainingSubset.attribute(attIndex).numValues(); i++) {
			Sv = new Instances(trainingSubset, 0);
			//for all instances
			for (int j = 0; j < trainingSubset.numInstances(); j++) {
				//if the j instance's attIndex's attribute is 'i' (the one we now check) then add the instance to Sv
				if ((double) i == trainingSubsetAsNumbers[j][attIndex]) {
					Sv.add(trainingSubset.instance(j));
				}
			}
			//calculating gain according to 'accordingToGini' boolean
			if (accordingToGini)
				sum += ((double)Sv.numInstances() / (double)trainingSubset.numInstances()) * calcGini(Sv);
			else 
				sum += ((double)Sv.numInstances() / (double)trainingSubset.numInstances()) * calcEntropy(Sv);
		}
		//return value is also determined according to 'accordingToGini' boolean
		if (accordingToGini)
			return calcGini(trainingSubset) - sum;
		else 
			return calcEntropy(trainingSubset) - sum;
	}
	
	/**
	 * Calculate gini for a given dataset
	 * @param instances
	 * @return
	 */
	private static double calcGini (Instances instances) {
		//If there are no instances there is no impurity.
		if (instances.numInstances() == 0) return 0;
		//Keep track of class value count.
    	double numOfClassYes = 0, numOfClassNo = 0;
    	for (int i = 0; i < instances.numInstances(); i++) {
    		if (instances.instance(i).classValue() == 1) numOfClassYes++;
    		else numOfClassNo++;
    	}
    	//Calculate Gini according to the formula.
    	double squaredPartialYes = Math.pow(numOfClassYes / (double) instances.numInstances(), 2);
    	double squaredPartialNo = Math.pow(numOfClassNo / (double) instances.numInstances(), 2);
    	
    	return (1 - (squaredPartialYes + squaredPartialNo));
    }
    
	/**
	 * Calculate entropy for a given dataset
	 * @param instances
	 * @return
	 */
    public static double calcEntropy (Instances instances) {
    	//If there are no instances there is no impurity.
    	if (instances.numInstances() == 0) return 0;
    	double numOfClassYes = 0, numOfClassNo = 0;
    	//Keep track of class value count.
    	for (int i = 0; i < instances.numInstances(); i++) {
    		if (instances.instance(i).classValue() == 1) numOfClassYes++;
    		else numOfClassNo++;
    	}
    	
    	//Calculate Entropy according to the formula.
    	double partialYes = numOfClassYes / (double) instances.numInstances();
    	double partialNo = numOfClassNo / (double) instances.numInstances();
    	//If one of the class counts is 0 we return 0 (to avoid dividing by 0)
    	if (partialYes == 0 || partialNo == 0) return 0;
    	
    	double partialYesLog2 = Math.log10(partialYes) / Math.log10(2);
    	double partialNoLog2 = Math.log10(partialNo) / Math.log10(2);
    	
    	return -((partialYes * partialYesLog2) + (partialNo * partialNoLog2));
    }
	
	/**
	 * Calculates the chi square statistic of splitting the
	 * data according to the splitting attribute as learned in class.
	 * @param trainingSubset - The dataset to check chisquare value
	 * @param attributeIndex - The index to be examined
	 * @return
	 */
	public static double calcChiSquare(Instances trainingSubset, int attIndex) {
		// P( Y = 0 )
		double probYes = probability(trainingSubset);
		// P( Y = 1 )
		double probNo = 1 - probYes;
		//Initialize variables for chiSquare calculation.
		double sum = 0, Y0, Y1, Df;
		for (int i = 0; i < trainingSubset.attribute(attIndex).numValues(); i++) {
			Y0 = 0;
			Y1 = 0;
			Df = 0;
			for (Instance instance: trainingSubset) {
				if (instance.value(attIndex) == i) {
					Df++;
					if (instance.classValue() == 0) Y0++;
					else Y1++;
				}
			}
			//In case we divide by 0 continue summing up attribute values.
			if (probYes * Df == 0 || probNo * Df == 0) continue;
			sum += Math.pow(Y0 - (Df * probYes), 2) / (Df * probYes) +
					Math.pow(Y1 - (Df * probNo), 2) / (Df * probNo);
		}
		return sum;
	}
	
	/**
	 * Calculate the probabilty P( Y = 0 )
	 * @param data
	 * @return
	 */
    public static double probability(Instances data) {
    	double sum = 0;
    	for (Instance instance: data) {
    		if (instance.classValue() == 0) sum++;
    	}
    	return sum / data.numInstances();
    }
	
	
	/**
     * A helper function to translate the data into an array that
     * maps an instance to an array containing all values for attributes
     *  
     * @param data
     * @return the given data as an array
     */
    public static double[][] dataAsArray(Instances data) { 
        double Data[][] = new double[data.numInstances()][data.numAttributes()]; 
        for (int i = 0; i < data.numInstances(); i++) { 
            for (int j = 0; j < data.numAttributes(); j++) { 
                Data[i][j] = data.instance(i).value(j); 
            } 
        } 
        return Data; 
    }
    
    /**
     * Use printTree from the root.
     * (Wrapper function)
     */
    public static void PrintTree() {
    	printTree(rootNode, "");
    }
    
    /**
     * Print the tree from a given node with indentations.
     * 
     * @param root - the node to start printing from
     * @param tabs - Amount of indentation to be done on the print
     */
    public static void printTree(Node root, String tabs) {
    	if(root.equals(rootNode)) System.out.println("Root");
    	System.out.println(tabs + "Returning value: " + root.returnValue);
    	tabs += "\t";
    	for (Node c : root.children) {
    		System.out.println(tabs + "If attribute " + c.parent.attributeIndex + " = " + c.brancedFromValue);
    		if (c.children.size() == 0) {
    	    	System.out.print("\t");
    			System.out.println(tabs + "Leaf. Returning value: " + c.returnValue);
    			continue;
    		}
    		printTree(c, tabs);
    	}
    }
    
    /**
     * Set the return value for a node depending on
     * the majority of instances gaven in the training set.
     * This function is to be called through buildTree function.
     * 
     * @param node - The node to alter.
     */
    public static void setClassification(Node node) {
    	double yes = 0;
    	for (Instance instance : node.instances) {
    		if (instance.classValue() == 1) yes++;
    	}
    	double classification = (yes / node.instances.numInstances());
    	if (classification > 0.5) node.returnValue = 1.0;
    	else node.returnValue = 0.0;
    }
    
    /**
     * Calaculate the max height in the tree in a recursive matter.
     * @param node - the node from which to start doing recursive calls, in our case the root.
     * @return
     */
    public static int treeHeight (Node node) {
    	if (node.children.size() == 0) return 0;
    	int heightestChild = 0;
    	for (Node c : node.children) {
    		if (heightestChild < treeHeight(c)) {
    			heightestChild = treeHeight(c);
    		}
    	}
    	return 1 + heightestChild;
    }
    
    /**
     * Getter method for rootNode
     * @return
     */
    public Node getRoot() {
    	return rootNode;
    }
}