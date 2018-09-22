package OD;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import weka.core.Instance;
import weka.core.Instances;


public class OutliernessEvaluator {

	public static double[] objectOutliernessScoreing(double[] valueOutlierness, Instances instances, int[] firstValueIndex) {
		int nObject = instances.numInstances();
		int nFeatures = instances.numAttributes() - 1;

		//calculate weight of each feature
		double[] relevances = new double[nFeatures];
		double relevanceSum = 0.0;
		for(int i = 0; i < nFeatures; i++) {
			double relevance = 0.0;
			for(int j = firstValueIndex[i]; j < firstValueIndex[i+1]; j++) {
				relevance += valueOutlierness[j];
			}
			relevances[i] = relevance;
			relevanceSum += relevance;
		}
		
		double[] weight = new double[nFeatures];
		for(int i = 0; i < nFeatures; i++) {
			weight[i] = relevances[i] / relevanceSum;			
		}
		
		double[] objectOutlierness = new double[nObject];
		for(int i = 0; i < nObject; i++) {
			double score = 0.0;
			Instance instance = instances.instance(i);
			for(int j = 0; j < nFeatures; j++) {
				double value = instance.value(j);
				double featureWeight = weight[j];
				int valueIndex = firstValueIndex[j] + (int) value;
				score += valueOutlierness[valueIndex] * featureWeight;
			}
			objectOutlierness[i] = score;
		}
		
		
		return objectOutlierness;
	}
	
	public static Hashtable<Integer, Double> GenerateObjectScoreMap(double[] objectScore){
		Hashtable<Integer, Double> objectScoreTable = new Hashtable<>();
		for(int i = 0; i < objectScore.length; i++) {
			objectScoreTable.put(i, objectScore[i]);
		}		
		return objectScoreTable;		
	}
	
	/**
	 * use function to calculate the initial outlierness of each value
	 * @param firstValueIndex
	 * @param valueFrequency
	 * @return
	 */
	public static double[] calInitValueOutlierness(int[] firstValueIndex, int[] valueFrequency) {
		int nValue = valueFrequency.length;
		int nFeatures = firstValueIndex.length - 1;
			
		int[] featureModeValueFrequency = new int[nFeatures];
		int superModeFrequency = Integer.MIN_VALUE; 
		for(int i = 0; i < nFeatures; i++) {
			int nFeatureValue = firstValueIndex[i+1] - firstValueIndex[i];
			int modeValue = Integer.MIN_VALUE;
			for(int j = 0; j < nFeatureValue; j++) {
				if(valueFrequency[firstValueIndex[i] + j] > modeValue) {
					modeValue = valueFrequency[firstValueIndex[i] + j];
				}
			}
			featureModeValueFrequency[i] = modeValue;
			if(superModeFrequency < modeValue) {
				superModeFrequency = modeValue;
			}
		}
		
		double[] initValueOutlierness = new double[nValue];
		for(int i = 0; i < nFeatures; i++) {
			int nFeatureValue = firstValueIndex[i+1] - firstValueIndex[i];
			for(int j = 0; j < nFeatureValue; j++) {
				int frequency = valueFrequency[firstValueIndex[i] + j];
				double valueOutlierness = ((double) featureModeValueFrequency[i] - (double) frequency) / (double) featureModeValueFrequency[i]
						+ ((double) superModeFrequency - (double) featureModeValueFrequency[i]) / (double) superModeFrequency;
				valueOutlierness = 0.5 * valueOutlierness;
				initValueOutlierness[firstValueIndex[i] + j] = valueOutlierness;	
			}
		}
		return initValueOutlierness;
	}


	/**
	 * calculate biased cluster-value couplings
	 * @param transitionMatrix
	 * @param biasVector
	 * @param clusterIndex
	 * @param clusterInfo
	 * @return
	 */
	public static double[] calcBiasedClusterValueCoupling(double[][] transitionMatrix, 
			double[] biasVector, int clusterIndex, int[] clusterInfo) {
		
		int nValue = transitionMatrix.length;
		double[] valueOutlierness = new double[nValue];

		for(int i = 0; i < nValue; i++) {
			double score = 0.0;
			for(int j = 0; j < nValue; j++) {				
				if(clusterInfo[j] == clusterIndex) {
					score += transitionMatrix[j][i] * biasVector[j];
				}
			}
			valueOutlierness[i] = score;
		}
		
		return valueOutlierness;
	}		
	
	
	
	/**
	 * Recognize an outlying value cluster by Biased Cluster-value Couplings
	 * @param transitionMatrix
	 * @param biasedVector
	 * @param clusterInfo
	 * @param normalClusterIndex
	 * @param nClusters
	 * @return
	 */
	public static double[] calcValueOutliernessByBCVc(double[][] transitionMatrix,
			double[] biasedVector,
			int[] clusterInfo,
			int normalClusterIndex,
			int nClusters) {		
		int nValue = transitionMatrix.length;
		double[] valueOutlierness = new double[nValue];		
		for(int i = 0; i < nClusters; i++) {
			if(i != normalClusterIndex) {
				double[] tmpValueOutlierness = calcBiasedClusterValueCoupling(transitionMatrix, biasedVector, i, clusterInfo);
				for(int j = 0; j < nValue; j++) {
					valueOutlierness[j] += tmpValueOutlierness[j];
				}
			}
		}	
		return valueOutlierness;
	}
	
	
	
	
	public static double[] valueOutliernessScoring(double[][] CPmatrix, 
			double[] lastValueOutlierness, List<Integer> selectedValueList) {
		int nValue = CPmatrix.length;
		double[] valueOutlierness = new double[nValue];
		
		double[] columnSum = new double[nValue];
		for(int i = 0; i < nValue; i++) {
			double sum = 0.0;
			for(int j = 0; j < nValue; j++) {
				sum += CPmatrix[j][i];
			}
			columnSum[i] = sum;
		}	
			
		double[][] normalizedMatrix = new double[nValue][nValue];
		for(int i = 0; i < nValue; i++) {
			for(int j = 0; j < nValue; j++) {
				if(columnSum[j] == 0) {
					normalizedMatrix[i][j] = 0;
				}else {
					normalizedMatrix[i][j] = CPmatrix[i][j] / columnSum[j];
				}			
			}
		}
		
		
		//score each value 
		for(int i = 0; i < nValue; i++) {
			double score = 0.0;
			for(int j = 0; j < selectedValueList.size(); j++) {
				int index = selectedValueList.get(j);
				score += normalizedMatrix[i][index] * lastValueOutlierness[index];
			}
			valueOutlierness[i] = score;
		}
		return valueOutlierness;
	}	
	
	

	

	public static List<Integer> getValueListByCluster(int[] clusterInfo, int index){
		List<Integer> list = new ArrayList<>();
		for(int i = 0; i < clusterInfo.length; i++) {
			if(clusterInfo[i] == index) {
				list.add(i);
			}
		}
		return list;
	}
	


	
	
	
	
	/**
	 * calculate outlierness of each cluster based on init outlierness of each value
	 * @param clusterInfo 
	 * @param nCluster
	 * @param valueFrequency
	 * @return double[] index is clusterId, content is the sum of value/node frequency
	 */
	public static double[] calcClusterOutlierness(int[] clusterInfo, int nCluster, double[] valueOutlierness) {
		double[] clusterOutlierness = new double[nCluster];
		int[] clusterSize = new int[nCluster];
		int nValues = valueOutlierness.length;
		
		//calc cluster avg
		for(int i = 0; i < clusterInfo.length; i++) {
			double outlierness = valueOutlierness[i];
			int clusterId = clusterInfo[i];
			clusterOutlierness[clusterId] += outlierness;
			clusterSize[clusterId]++;
		}		
		
		double tmpSum = 0.0;
		for(int i = 0; i < nCluster; i++) {
			clusterOutlierness[i] = clusterOutlierness[i] / (double) clusterSize[i];
			tmpSum += clusterOutlierness[i];
		}
		return clusterOutlierness;		
	}
	
	
	/**
	 * get selected value subset - the most normal cluster
	 * @param clusterInfo 
	 * @param nCluster
	 * @param valueFrequency
	 * @return double[] index is clusterId, content is the sum of value/node frequency
	 */
	public static int getNormalClusterId(double[] clusterOutlierness) {
		double minOutlierness = 1.0;
		int index = -1;
		for(int i = 0; i < clusterOutlierness.length; i++) {
			if(clusterOutlierness[i] < minOutlierness) {
				index = i;
				minOutlierness = clusterOutlierness[i];
			}
		}
		return index;		
	}
	
	
	/**
	 * get selected value subset - the most outlier cluster
	 * @param clusterInfo 
	 * @param nCluster
	 * @param valueFrequency
	 * @return double[] index is clusterId, content is the sum of value/node frequency
	 */
	public static int getOutlierClusterId(double[] clusterOutlierness, int normalClusterIndex) {
		double maxOutlierness = 0.0;
		int index = -1;
		for(int i = 0; i < clusterOutlierness.length; i++) {
			if(clusterOutlierness[i] > maxOutlierness && i != normalClusterIndex) {
				index = i;
				maxOutlierness = clusterOutlierness[i];
			}
		}
		return index;		
	}
	
	
	/**
	 * get selected value subset - the most outlier cluster by value2cluster distance
	 * @param clusterInfo 
	 * @param nCluster
	 * @param valueFrequency
	 * @return double[] index is clusterId, content is the sum of value/node frequency
	 */
	public static int getOutlierClusterId2(double[] clusterOutlierness, int normalClusterIndex) {
		double minOutlierness = Double.MAX_VALUE;
		int index = -1;
		for(int i = 0; i < clusterOutlierness.length; i++) {
			if(clusterOutlierness[i] < minOutlierness && i != normalClusterIndex) {
				index = i;
				minOutlierness = clusterOutlierness[i];
			}
		}
		return index;		
	}
	
	



	public static int[] reviseCluserInfo(int[] clusterInfo, List<Integer> remainedClusters, int normalIndex) {
		int[] newClusterInfo = new int[clusterInfo.length];
		for(int i = 0; i < clusterInfo.length; i++) {
			int clusterIndex = clusterInfo[i];
			
			if(clusterIndex == normalIndex) {
				newClusterInfo[i] = 0;
			}
			else {
				if(remainedClusters.contains(clusterIndex)) {
					newClusterInfo[i] = 1;
				}else {
					newClusterInfo[i] = 2;
				}
			}
		}
		
		return newClusterInfo;
	}
	
	

	
	
	
	
	//use value to outlier cluster distance as creterion
	public static int[] refineOutlyingValueSet(int[] clusterInfo, double[][] value2ClusterCoupling) {
		int nValues = clusterInfo.length;	
		int[] newClusterInfo = new int[nValues];
		
		//use normal and others avg distance as threshold to refine outlier cluster
		double distanceSum = 0.0;
		int count = 0;
		for(int i = 0; i < nValues; i++) {
			distanceSum += value2ClusterCoupling[i][1];
			count++;			
		}
		double creterion = distanceSum / (double)count;
			
		for(int i = 0; i < nValues; i++) {
			double tmp = value2ClusterCoupling[i][1];			
			if(clusterInfo[i] == 1) {
				if(tmp < creterion) {
					newClusterInfo[i] = 2;
				}else {
					newClusterInfo[i] = clusterInfo[i];
				}
			}else {
				if(clusterInfo[i] == 2) {
					if(tmp > creterion) {
						newClusterInfo[i] = 1;
					}else {
						newClusterInfo[i] = clusterInfo[i];
					}
				}
			}
		}
		return newClusterInfo;	
	}
	

	
	
	
	
	
	public static List<Integer> getClusterValueListWithoutNormalValue(int[] clusterInfo, 
			int normalIndex, int nCluster) {		
		int nValues = clusterInfo.length;		
		List<Integer> list = new ArrayList<>();

		for(int i = 0; i < nValues; i++) {
			if(clusterInfo[i] != normalIndex) {
				list.add(i);
			}
		}
		
		return list;
	}
	
	
	
	

	
	
	public static double[][] calcClusterValueCoupling(int[] clusterInfo, 
			int nClusters, double[][] transitionMatrix){
		int nValues = transitionMatrix.length;
		
		double[][] clusterNodeTransitionMatrix = new double[nClusters][nValues];
		
		for(int i = 0; i < nClusters; i++) {
			List<Integer> clusterValueList = getValueListByCluster(clusterInfo, i);
			int clusterSize = clusterValueList.size();
				
			for(int j = 0; j < nValues; j++) {
				double tmp = 0.0;
				for(int k = 0; k < clusterSize; k++) {
					int clusterValueIndex = clusterValueList.get(k);
					tmp += transitionMatrix[clusterValueIndex][j];					
				}
				clusterNodeTransitionMatrix[i][j] = tmp / (double)clusterSize;			
			}
		}
		return clusterNodeTransitionMatrix;
	}
	
	
	

	
	
	public static double[][] calcClusterDistance(int clusterInfo[], int nClusters, 
			double[][] transitionMatrix, double[][] clusterNodeTransitionMatrix,
			double[] degree){	
		int nValues = transitionMatrix.length;		
		double[][] clusterDistance = new double[nClusters][nClusters];
		
		
		for(int i = 0; i < nClusters; i++) {
			for(int j = 0; j < nClusters; j++) {
				double tmp = 0.0;
				for(int k = 0; k < nValues; k++) {
					tmp+= (clusterNodeTransitionMatrix[i][k] - clusterNodeTransitionMatrix[j][k]) * 
							(clusterNodeTransitionMatrix[i][k] - clusterNodeTransitionMatrix[j][k]) / degree[k];
				}
				tmp = Math.sqrt(tmp);
				clusterDistance[i][j] = tmp;
			}				
		}
	
		return clusterDistance;
	}
	
	
	
	
	public static double[][] calcValue2ClusterEdgeWeight(int revisedClusterInfo[], 
			double[][] similarityMatrix) {
		
		int nValues = revisedClusterInfo.length;

		int[] sizeCount = new int[3];
		double[][] value2clusterCoupling = new double[nValues][3];
		
		//0 is normal, 1 is outlier, 2 is others
		for(int i = 0; i < nValues; i++) {
			int clusterIndex = revisedClusterInfo[i];
			if(clusterIndex == 0) {
				sizeCount[0]++;
			}else {
				if(clusterIndex == 1) {
					sizeCount[1]++;
				}else {
					sizeCount[2]++;
				}
			}
		}
				
		int clusterCount = 0;
		if(sizeCount[2]!=0) {
			clusterCount = 3;
		}else {
			clusterCount = 2;
		}
		
		for(int i = 0; i < nValues; i++) {
			for(int j = 0;j < nValues; j++) {
				int jCluster = revisedClusterInfo[j];
				value2clusterCoupling[i][jCluster] += similarityMatrix[i][j];				
			}
		}
		
		for(int i = 0; i < nValues; i++) {
			for(int j = 0; j < clusterCount; j++) {
				value2clusterCoupling[i][j] = value2clusterCoupling[i][j] / (double) sizeCount[j];
			} 
		}
		

		return value2clusterCoupling;
	}
	
	
	





	
	
	public static List<Integer> determineOutlierCluster (double[][] clusterDistance,
			int normalClusterId, int outlierClusterId){
		
		int nClusters = clusterDistance.length;
		double normal2outlier = clusterDistance[normalClusterId][outlierClusterId];
		
		List<Integer> outlierClusterList = new ArrayList<>();
		
		for(int i = 0; i < nClusters; i++) {
			double distance2normal = clusterDistance[i][normalClusterId];
			double distance2outlier = clusterDistance[i][outlierClusterId];
					
			if(distance2outlier < normal2outlier) {
				outlierClusterList.add(i);
			}
			
		}
		return outlierClusterList;
	}
	
	
	
	public static List<Integer> getRemainingFeatureList(List<Integer> remainingValueList, int[] firstValueIndex) {
		List<Integer> remainingFeatureList = new ArrayList<>();
	
		for(int i = 0; i < remainingValueList.size(); i++) {
			int valueIndex1 = remainingValueList.get(i);
			int featureIndex1 = 0;
			for(featureIndex1 = 0; featureIndex1 < firstValueIndex.length-1; featureIndex1++) {
				int featureValueIndex = firstValueIndex[featureIndex1];

				if(valueIndex1 >= featureValueIndex) {
					continue;
				}else {
					break;
				}		
			}
			featureIndex1 = featureIndex1-1;
			//int localValueIndex1 = valueIndex1 - firstValueIndex[featureIndex1];
			if(!remainingFeatureList.contains(featureIndex1)) {
				remainingFeatureList.add(featureIndex1);
			}
			
		}
		
		return remainingFeatureList;
	}
	
	
	
	public static void featureAnalysis(List<Integer> remainingValueList, List<Integer> normalValueList, 
			int[] firstValueIndex) {
		int nFeatures = firstValueIndex.length-1;
		
		int featureContainOutlier = 0;
		int featureContainNormal = 0;
		int featureContainBoth = 0;
		int featureOnlyContainNoise = 0;
		
		int valueCount = 0;
		for(int i = 0; i < nFeatures; i++) {
			int featureSize = firstValueIndex[i+1] - firstValueIndex[i];
			boolean isFeatureContainOutlier = false;
			boolean isFeatureContainNormal = false;
			boolean isFeatureContainBoth = false;
			boolean isFeatureOnlyContainNoise = false;
			
			
			for(int j = 0; j < featureSize; j++) {
				if(normalValueList.contains(valueCount)) {
					isFeatureContainNormal = true;
				}
				if(remainingValueList.contains(valueCount)) {
					isFeatureContainOutlier = true;
				}
				valueCount++;			
			}
			
			if(isFeatureContainNormal && isFeatureContainOutlier) {
				isFeatureContainBoth = true;
			}
			if(isFeatureContainNormal == false && isFeatureContainOutlier == false) {
				isFeatureOnlyContainNoise = true;
			}
			
			if(isFeatureContainNormal) {
				featureContainNormal++;
			}
			if(isFeatureContainOutlier) {
				featureContainOutlier++;
			}
			if(isFeatureContainBoth) {
				featureContainBoth++;
			}
			if(isFeatureOnlyContainNoise) {
				featureOnlyContainNoise++;
			}
			
		}
		System.out.println();
		System.out.println("nfeautre," + nFeatures);
		System.out.println("featureContainOutlier," + featureContainOutlier);
		System.out.println("featureContainNormal," + featureContainNormal);
		System.out.println("featureContainBoth," + featureContainBoth);
		System.out.println("featureContainOnlyNoise," + featureOnlyContainNoise);	
	}
	
	
	
	public static void createNewInstancesByFS2(List<Integer> remainingFeatureList, 
			Instances oldInstances,
			int[] firstValueIndex,
			List<Integer> outlierValueList,
			List<Integer> normalValueList,
			int[] featureModeValueLocalIndex,
			String outPath) throws IOException {
		
		int nObjects = oldInstances.numInstances();		
		Instances newInstances = new Instances(oldInstances);				
		int oldFeatureNum = oldInstances.numAttributes();
		
		List<Integer> deletFeatureList = new ArrayList<>();
		for(int i = 0; i < oldFeatureNum-1; i++) {
			if(!remainingFeatureList.contains(i)) {
				deletFeatureList.add(i);
			}
		}
					
		int[] replaceValueLocalIndex = new int[oldFeatureNum-1];		
		for(int i = 0; i < oldFeatureNum - 1; i++) {
			int featureValueNum = firstValueIndex[i+1] - firstValueIndex[i];	
						
			boolean haveNormalValue = false;
			int normalValueLocalIndex = -1;
			int nonOutlierValueLocalIndex = -1;
			for(int j = 0; j < featureValueNum; j++) {
				int genralIndex = firstValueIndex[i] + j;
				
				if(normalValueList.contains(genralIndex)) {
					haveNormalValue = true;
					normalValueLocalIndex = j;
				}else {
					if(!outlierValueList.contains(genralIndex)){
						nonOutlierValueLocalIndex = j;
					}
				}
			}
						
			if(haveNormalValue) {
				replaceValueLocalIndex[i] = normalValueLocalIndex;
			}else {
				replaceValueLocalIndex[i] = nonOutlierValueLocalIndex;
			}			
		}
		
		
		for(int i = 0; i < nObjects; i++) {
			Instance instance = newInstances.instance(i);
			for(int j = 0; j < oldFeatureNum -1; j++) {

				if(!deletFeatureList.contains(j)) {
					//int modeValueIndex = featureModeValueLocalIndex[j];	
					int replaceValue = replaceValueLocalIndex[j];

					double value = instance.value(j);
					int generalIndex = firstValueIndex[j] + (int) value;
					
					if(!outlierValueList.contains(generalIndex)) {
						instance.setValue(j, (double)replaceValue);
					}
				}
			}
		}
				
		for(int i = deletFeatureList.size()-1; i >= 0; i--) {
			int index = deletFeatureList.get(i);
			newInstances.deleteAttributeAt(index);
		}
		//System.out.println(newInstances);	
		
		BufferedWriter writer = new BufferedWriter(new FileWriter(outPath));
		writer.write(newInstances.toString());
		writer.flush();
		writer.close();
	}
	
}
