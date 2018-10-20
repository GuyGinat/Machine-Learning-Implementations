package LinearRegression;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {
	
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;

	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		m_truNumAttributes = trainingData.numAttributes() - 1;
		m_coefficients = new double [m_truNumAttributes + 1];
		//setting all tetas to the value 1
		for (int i = 0; i < m_truNumAttributes; i++) {
			m_coefficients[i] = 1;
		}
		
		//relevant only for the first time for 'm_alpha'
		if (m_alpha == 0) {
			findAlpha(trainingData);
			for (int i = 0; i < m_truNumAttributes; i++) {
				m_coefficients[i] = 1;
			}
		}
		m_coefficients = gradientDescent(trainingData);
	}
	
	private void findAlpha(Instances data) throws Exception {
		double prevErr = Double.MAX_VALUE, currErr = Double.MAX_VALUE, minimumErr  = Double.MAX_VALUE, bestAlpha = 0;
		for (int i = -17; i <= 0; i++) {
			m_alpha = Math.pow(3, i);
			for (int j = 0; j <= m_truNumAttributes; j++) {
				m_coefficients[j] = 1;
			}
			prevErr = calculateMSE(data);
			for (int j = 1; j <= 20000; j++) {
				currErr = calculateMSE(data);
				updateTetas(data);
				if (j % 100 == 0) {
					currErr = calculateMSE(data);
					//if the current error is larger then the previous one, exit
					if (currErr >= prevErr) {
						break;
					}
					prevErr = currErr;
				}
			}
			//if the current error is the smallest to be found among all alphas so far, update 'minimumErr' and 'bestAlpha' respectively
			if (currErr < minimumErr) {
				minimumErr = currErr;
				bestAlpha = m_alpha;
			}
			System.out.println("i: " + i + " error is: " + currErr);
			//updating 'currErr' for the next iteration over 'i'
			currErr = Double.MAX_VALUE;
		}
		//updating 'm_alpha' to be the best alpha found
		m_alpha = bestAlpha;
	}
	
	/**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData) throws Exception {
		//starting values for the errors variables which enable the loop starting condition
		double prevError = 1;
		double curError = 0;
		//intializing all tetas to the value '1'
		for (int i = 0; i < m_coefficients.length; i ++) {
			m_coefficients[i] = 1;
		}
		//continue as long as the error difference is smaller than 0.003
		while (Math.abs(prevError - curError) > 0.003) {
			prevError = curError;
			//100 "updateTetas"
			for (int i = 0; i < 100; i++) {
				updateTetas(trainingData);
			}
			curError = calculateMSE(trainingData);
		}
		return m_coefficients;
	}
	
	/**
	 * Update tetas method is called each iteration of the
	 * find alpha method and the gradient decsent method.
	 * 
	 * @param trainingData
	 * @throws Exception
	 */
	private void updateTetas(Instances trainingData) throws Exception {
		//temporary array for the tetas
		double [] tempTetas = new double [m_truNumAttributes + 1];
		double derivative  = 0, derivativeSum = 0;
		for (int i = 0; i <= m_truNumAttributes; i++) {
				for (int j = 0; j < trainingData.numInstances(); j++) {
					//calculate the derivative according to the formula
					derivative  = regressionPrediction(trainingData.instance(j)) - trainingData.instance(j).value(m_ClassIndex);
					//for teta = 0 , no need to multiply by X0 (equals 1)
					if (i > 0) {
						derivative  *= trainingData.instance(j).value(i - 1);
					}
					derivativeSum += derivative;
				}
				//updating teta i
				tempTetas[i] = m_coefficients[i] - (m_alpha * derivativeSum / (double)trainingData.numInstances());
				derivativeSum = 0;
		}
		//updating 'm_coefficients' tetas array, corresponding to the temporary array
		for (int i = 0; i <= m_truNumAttributes; i++) {
			m_coefficients[i] = tempTetas[i];
		}
		return;
	}

	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
     *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		double innerProduct = m_coefficients[0];
		for (int i = 0; i < m_truNumAttributes; i++) {
			innerProduct += m_coefficients[i + 1] * instance.value(i);
		}
		//return [ X0 + X(i)*Teta(i) ] for each 1 <= i <= m_truNumAttributes
		return innerProduct;
	}
	
	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param testData
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {
		double cost = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			//sum up the errors of all predictions minus actual value (squared)
			cost += Math.pow((regressionPrediction(data.instance(i)) - (double)data.instance(i).value(m_ClassIndex)), 2.0);
		}
		//dividing by 2m 
		cost = cost / ((double)data.numInstances() * 2.0);
		return cost;
	}
    
    @Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
	}

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
	 * getter method so we can print m_alpha and keep it private 
	 * @return
	 */
	public double getAlpha() {
		return m_alpha;
	}
}