package OD;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import Jama.Matrix;
import Utils.ArrayUtils;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class DataConstructor {

	
	private int nFeatures;
	private int nObjects;
	private int nValues;
	private int[] firstValueIndex;
	private int[] valueFrequency;
	private int[] valuefeatureMap;
	private int[] featureModeValueFrequency;
	private int[] featureModeValueLocalIndex;

	private int superModeFrequency;
	private int[][] coOccurrence;
	private double[][] conditionalPossibility;
	private int[] coOccurenceWithLabel;
	private double[] conditionalPossibilityWithLabel;
	//private double[] labelCP;
	private double[] simWithLabel;
	private double[][] similarityMatrix;
	private double[][] transitionMatrix;
	private double[][] distanceMatrix;
	
	private double[] nodeDegree;
	
	private double[][] normRemainedValueMatrix;
	private double[][] remainedValueMatrix;
    private List<String> listOfCalss = new ArrayList<>();
    private Instances instances;

	
	
	
	
	/**
	 * record 
	 * nFeatures, nObjects, firstValueIndex, nValues
	 * valueFrequency, coOccurrence, listOfClass
	 * 
	 * @throws IOException
	 */
	public void dataPrepareFromArff(String filePath) throws IOException{
				
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(filePath));
		
		Instances instances = loader.getDataSet();
		this.instances = instances;
		instances.setClassIndex(instances.numAttributes()-1);
		nFeatures = instances.numAttributes() - 1;
		nObjects = instances.numInstances();
		
		//record value number of each feature and calculate the sum of all possible values of all features
		firstValueIndex = new int[nFeatures + 1];
		firstValueIndex[0] = 0;
		for(int i = 1; i < nFeatures; i++) {
			firstValueIndex[i] = firstValueIndex[i-1] + instances.attribute(i-1).numValues();
		}
		firstValueIndex[nFeatures] = firstValueIndex[nFeatures - 1] + instances.attribute(nFeatures-1).numValues();
		nValues = firstValueIndex[nFeatures];
		
		valuefeatureMap = new int[nValues];
		int tmpCount = 0;
		for(int i = 0; i < nFeatures; i++) {
			int tmpFeatureNumber = instances.attribute(i).numValues();
			for(int j = 0; j < tmpFeatureNumber; j++) {
				valuefeatureMap[tmpCount] = i;
				tmpCount++;
			}
		}
		
		
		valueFrequency = new int[nValues];
		coOccurrence = new int[nValues][nValues];
	
		
		for(int i = 0; i < nObjects; i++) {
			Instance instance = instances.instance(i);

			//calculate value frequency
			for(int j = 0; j < nFeatures; j++) {
				int localValueIndex = (int) instance.value(j);
				int globalValueIndex = firstValueIndex[j] + localValueIndex;
				valueFrequency[globalValueIndex]++;			
			}
				
			//record class of each objects
			if(instance.value(nFeatures) == 0.0) {
				listOfCalss.add("outlier");
			}else {
				listOfCalss.add("normal");
			}
			
			//calculate co-occurrence
			for(int a = 0; a < nFeatures; a++) {
				for(int b = 0; b < nFeatures; b++) {
					int valueLocalIndex1 = (int) instance.value(a);
					int valueLocalIndex2 = (int) instance.value(b);
					int valueGlobalIndex1 = valueLocalIndex1 + firstValueIndex[a];
					int valueGlobalIndex2 = valueLocalIndex2 + firstValueIndex[b];
					coOccurrence[valueGlobalIndex1][valueGlobalIndex2]++;
				}
			}
				
		}
		
		//calculate conditional possibility matrix
		conditionalPossibility = new double[nValues][nValues];
		for(int i = 0; i < nValues; i++) {
			for(int j = 0; j < nValues; j++) {
				if(valueFrequency[i] != 0) {
					conditionalPossibility[i][j] = (double) coOccurrence[i][j] / (double) valueFrequency[i];
				}else {
					conditionalPossibility[i][j] = 0;
				}
					
			}
		}
		
		//calculate Similarity Matrix
		similarityMatrix = new double[nValues][nValues];
		for(int i = 0; i < nValues; i++) {
        	for(int j = 0; j < nValues; j++) {
        		double weight = 0.0;
        		
        		if(coOccurrence[i][j] != 0) {
            		weight = ((double) coOccurrence[i][j]) / Math.sqrt((double)valueFrequency[i] *  (double)valueFrequency[j]);
        		}
        		similarityMatrix[i][j] = weight;
        	}      
        }      
		
		
		nodeDegree = new double[nValues];
		transitionMatrix = new double[nValues][nValues];
		for(int i = 0; i < nValues; i++) {
			double degreeI = 0.0;
			for(int j = 0; j < nValues; j++) {
				degreeI += similarityMatrix[i][j];
			}
			nodeDegree[i] = degreeI;
		}
		for(int i = 0; i < nValues; i++) {
			for(int j = 0; j < nValues; j++) {
				transitionMatrix[i][j] = similarityMatrix[i][j] / nodeDegree[i];
			}
		}	

		featureModeValueFrequency = new int[nFeatures];
		featureModeValueLocalIndex = new int[nFeatures];
		superModeFrequency = Integer.MIN_VALUE; 
		for(int i = 0; i < nFeatures; i++) {
			int nFeatureValue = firstValueIndex[i+1] - firstValueIndex[i];
			int modeValue = Integer.MIN_VALUE;
			int localModeIndex = -1;
			for(int j = 0; j < nFeatureValue; j++) {
				if(valueFrequency[firstValueIndex[i] + j] > modeValue) {
					modeValue = valueFrequency[firstValueIndex[i] + j];
					localModeIndex = j;
				}
			}
			featureModeValueFrequency[i] = modeValue;
			featureModeValueLocalIndex[i] = localModeIndex;
			if(superModeFrequency < modeValue) {
				superModeFrequency = modeValue;
			}
		}
		
	}
	
	
	
	
	
	
	
	
	public void calCPWithLabel() {
		
		coOccurenceWithLabel = new int[nValues];
		conditionalPossibilityWithLabel = new double[nValues];
						
		for(int i = 0; i < nObjects; i++) {
			Instance instance = instances.instance(i);
			for(int a = 0; a < nFeatures; a++) {				
				int valueLocalIndex = (int) instance.value(a);
				int valueGlobalIndex1 =  valueLocalIndex + firstValueIndex[a];
				if(instance.value(nFeatures) == 0.0) {
					coOccurenceWithLabel[valueGlobalIndex1]++;
				}				
			}
		}
				
		for(int i = 0; i < nValues; i++) {
			conditionalPossibilityWithLabel[i] = (double)coOccurenceWithLabel[i] / (double)valueFrequency[i];
		}
	}
	
	

	
	
	
	
	public void calSimWithLabel() {
		
		coOccurenceWithLabel = new int[nValues];
		simWithLabel = new double[nValues];
		
		
		int nOutliers = 0;
		for(int i = 0; i < listOfCalss.size(); i++) {
			if(listOfCalss.get(i).equals("outlier")) {
				nOutliers++;
			}
		}
		
		
		for(int i = 0; i < nObjects; i++) {
			Instance instance = instances.instance(i);
			//calculate co-occurrence
			for(int a = 0; a < nFeatures; a++) {				
				int valueLocalIndex = (int) instance.value(a);
				int valueGlobalIndex1 =  valueLocalIndex + firstValueIndex[a];
				if(instance.value(nFeatures) == 0.0) {
					coOccurenceWithLabel[valueGlobalIndex1]++;
				}				
			}
		}
		
		
		for(int i = 0; i < nValues; i++) {
			simWithLabel[i] = (double)coOccurenceWithLabel[i] / 
					Math.sqrt((double)valueFrequency[i] * (double)nOutliers);
		}
	}
	
	
	
	
	/**
	 * write network information into a list using cosine Similarity as weight of edge
	 * @param filePath
	 * @throws IOException
	 */
	public List<String> getNetworkList(double threhold) throws IOException {
        List<String> networkList = new ArrayList<>();
    
        for(int i = 0; i < nValues; i++) {
        	for(int j = i + 1; j < nValues; j++) {
        		double weight = similarityMatrix[i][j];
        		if(weight != 0 && weight >= threhold) {
        			String weightString = String.valueOf(weight);
        			String edgeInfo = i + " " + j + " " + weightString;
        			networkList.add(edgeInfo);
        		}
        	}      
        }        
  
        return networkList;
	}
	
	
	
	
	public double[][] calcTransitionMatrixT(int T){
		double[][] transitionMatrixT = new double[nValues][nValues];
		Matrix matrix = new Matrix(transitionMatrix);
		Matrix tmpMatrix = matrix;
		if(T== 1) {
			return transitionMatrix;
		}else {
			for(int i = 1; i < T; i++) {				
				Matrix newMatrix = tmpMatrix.times(matrix);				
				if(ArrayUtils.isMatrixSame(newMatrix, tmpMatrix)) {
					System.out.println("converge：" + i);
					tmpMatrix = newMatrix;
					break;					
				}else {
					tmpMatrix = newMatrix;
				}
				
			}
		}
		
		for(int i = 0; i < nValues; i++) {
			for(int j = 0; j < nValues; j++) {
				transitionMatrixT[i][j] = tmpMatrix.get(i, j);
			}
		}
		
		return transitionMatrixT;
	}
	
	
	
	
	
	
	
	
	public Instances generateNewInstances(double[][] CPmatrix){
		int valueInstancesFeatureNum = CPmatrix[0].length;
		FastVector attributes = new FastVector();
		for(int i = 0; i < valueInstancesFeatureNum; i++){
			attributes.addElement(new Attribute("A" + i));
		}
		
		
		Instances valueInstances = new Instances("valueInstances", attributes, 0);
		
		for(int i = 0; i < nValues; i++){
			Instance instance = new Instance(attributes.size());
			for(int j = 0; j < valueInstancesFeatureNum; j++){
				instance.setValue(j, CPmatrix[i][j]);
			}
			valueInstances.add(instance);
		}
		
		//System.out.println(valueInstances);
		return valueInstances;
	}
	
	

	
	
	
	
	
	
	
	
	
	public Instances generateNewInstancesByRemainValue(double[][] similarityMatrix, List<Integer> remainingValue){
		int valueInstancesFeatureNum = remainingValue.size();
		FastVector attributes = new FastVector();
		for(int i = 0; i < valueInstancesFeatureNum; i++){	
			attributes.addElement(new Attribute("value" + remainingValue.get(i)));
		}
		
		
		remainedValueMatrix = new double[valueInstancesFeatureNum][valueInstancesFeatureNum];		
		for(int i = 0; i < remainingValue.size(); i++){
			int indexI = remainingValue.get(i);
			for(int j = 0; j < remainingValue.size(); j++){
				int indexJ = remainingValue.get(j);
				double simi = similarityMatrix[indexJ][indexI];
				remainedValueMatrix[i][j] = simi;
			}
		}

		
//		double[][] normalizedMatrix = new double[remainingValue.size()][remainingValue.size()];
//		double[] columnSum = new double[remainingValue.size()];
//		for(int i = 0; i < remainingValue.size(); i++){
//			double tmpSum = 0.0;
//			for(int j = 0; j < remainingValue.size(); j++){
//				double simi = remainedValueMatrix[j][i];
//				tmpSum += simi;
//			}
//			columnSum[i] = tmpSum;
//		}
//		
//		for(int i = 0; i < remainingValue.size(); i++){
//			for(int j = 0; j < remainingValue.size(); j++){
//				double simi = remainedValueMatrix[i][j];
//				normalizedMatrix[i][j] = simi / columnSum[j];
//			}
//		}
//		
//		normRemainedValueMatrix = normalizedMatrix;
		
		Instances valueInstances = new Instances("valueInstances", attributes, 0);
		
		for(int i = 0; i < remainingValue.size(); i++){
			Instance instance = new Instance(attributes.size());
			for(int j = 0; j < valueInstancesFeatureNum; j++){
				instance.setValue(j, remainedValueMatrix[i][j]);
			}
			valueInstances.add(instance);
		}
		
		//System.out.println(valueInstances);
		return valueInstances;
	}
	
	
	
	
	
	

	
	
	
	
	
	
	
	
	public void output(String path, int[] clusterResults, double[] conditionalPWithLabel) throws IOException{
		BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(path));
		for(int i = 0; i < nValues; i++){
			bufferedWriter.write(clusterResults[i] + "," + conditionalPWithLabel[i]);
			bufferedWriter.newLine();
		}
		bufferedWriter.flush();
		bufferedWriter.close();
	}
	
	
	
	public double[][] getSimilarityMatrix(){
		return similarityMatrix;
	}
	public double[] getConditionalPWithLabel(){
		return conditionalPossibilityWithLabel;
	}
	
	public int getNFeatures() {
		return nFeatures;
	}
	
	public int getNObjects() {
		return nObjects;
	}
	
	public int getNValues() {
		return nValues;
	}
	
	public int[] getFirstValueIndex() {
		return firstValueIndex;
	}
	
	public int[] getValueFeatureMap() {
		return valuefeatureMap;
	}
	
	
	public int[] getValueFrequency() {
		return valueFrequency;
	}

	public int[][] getCoOccurrence() {
		return coOccurrence;
	}

	public double[][] getConditionalPossibility() {
		return conditionalPossibility;
	}
	
//	public double[] getConditionalPossibilityWithLabel() {
//		return conditionalPossibilityWithLabel;
//	}

	
	public double[] getSimWithLabel() {
		return simWithLabel;
	}
	
	public double[][] getTransitionMatrix(){
		return transitionMatrix;
	}
	
	public double[] getNodeDegree() {
		return nodeDegree;
	}
	
	public double[][] getDistanceMatrix(){
		return distanceMatrix;
	}
	
	public double[][] getNormRemainedValueMatrix(){
		return normRemainedValueMatrix;
	}
	
	public double[][] getRemainedValueMatrix(){
		return remainedValueMatrix;
	}
	
	public int[] getFeatureModeValueIndex() {
		return featureModeValueLocalIndex;
	}
	
	public List<String> getListOfClass() {
		return listOfCalss;
	}
	
	public Instances getInstances() {
		return instances;
	}
	
	
	
	
//	public static void main(String[] args) throws Exception{
//		DataConstructor dataConstructor = new DataConstructor();
//		dataConstructor.dataPrepareFromArff("E:\\data\\MDODarff\\finalData\\08-CT.arff");
//		dataConstructor.calCPWithLabel();
//		
//		Instances instances = dataConstructor.generateNewInstances(dataConstructor.getSimilarityMatrix());
//		
//		
//		int[] clusterResult = Clustering.runEM(instances, 2);
//		double[] conditionalPWithLabel = dataConstructor.getConditionalPWithLabel();
//		//dataConstructor.output("E:\\results.csv", clusterResult, conditionalPWithLabel);
//	}
	
	
}
