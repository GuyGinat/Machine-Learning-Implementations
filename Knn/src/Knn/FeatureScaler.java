package Knn;

import java.util.Random;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class FeatureScaler {
	/**
	 * Returns a scaled version (using standarized normalization) of the given dataset.
	 * @param instances The original dataset.
	 * @return A scaled instances object.
	 * @throws Exception 
	 */
	public static Instances scaleData(Instances instances) throws Exception {
		Standardize s = new Standardize();
        s.setInputFormat(instances);
        return Filter.useFilter(instances, s);
	}
}