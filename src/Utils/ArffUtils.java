package Utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class ArffUtils {

	public static void arff2csv(Instances instances, String outpath) throws IOException{
		int nObjects = instances.numInstances();
		int nFeatures = instances.numAttributes();

		BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outpath));

		
		for(int i = 0; i < nFeatures-1; i++){
			int index = i+1;
			bufferedWriter.write("A" + index + ",");
		}
		
		bufferedWriter.write("A" + nFeatures);
		bufferedWriter.newLine();
		
		
		for(int i = 0; i < nObjects; i++) {
			Instance instance = instances.instance(i);

			for(int j = 0; j < nFeatures-1; j++) {
				double value = instance.value(j);
				bufferedWriter.write(value + ",");
			}
			double value = instance.value(nFeatures-1);
			String valueStr = String.valueOf(value);
			bufferedWriter.write(valueStr);
	
		}
		
		bufferedWriter.flush();
		bufferedWriter.close();	
	}
	
	
	
	
//	public static void csv2InstancesNominal(String csvPath, String outpath) throws IOException{
//		
//		BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outpath));
//
//        File file = new File(csvPath);
//        BufferedReader buf = new BufferedReader(new FileReader(file));
//        
//        String head = buf.readLine();
//        int nFeatures = head.split(",").length - 1;
//        
//        
//        String featureVector = null;
//        while((featureVector = buf.readLine())!=null){
//            String[] values = featureVector.split(",");
//            for (int j = 0; j < nFeatures; j++) {
//            	
//            }
//            
//            
//        }
//        
//        
//        buf.close();
//
//		
//		for(int i = 0; i < nFeatures-1; i++){
//			int index = i+1;
//			bufferedWriter.write("A" + index + ",");
//		}
//		
//		bufferedWriter.write("A" + nFeatures);
//		bufferedWriter.newLine();
//		
//	
//		bufferedWriter.flush();
//		bufferedWriter.close();	
//	}
	
	
	

	

	
	public static Instances fsByValueSubset(
			List<Integer> selectedValueList, Instances orgInstances, 
			int[] valueFeatureMap,
			int[] firstValueIndex) {
		String name = orgInstances.relationName();
		int selectedValueNum = selectedValueList.size();
		int orgValueNum = valueFeatureMap.length;
		
		
		// construct retain feature list 
		// construct an array to indicate value number per features	
		List<Integer> retainFeatures = new ArrayList<>();
		Map<Integer, Integer> valueNumPerFeature = new HashMap<>();		
		for(int i = 0; i < selectedValueNum; i++) {
			int valueIndex = selectedValueList.get(i);
			int featureIndex = valueFeatureMap[valueIndex];
			if(!retainFeatures.contains(featureIndex)) {
				retainFeatures.add(featureIndex);
				valueNumPerFeature.put(featureIndex, 1);
			}else {
				int tmpNum = valueNumPerFeature.get(featureIndex);
				valueNumPerFeature.replace(featureIndex, tmpNum+1);
			}
						
		}
		int newFeatureNum = retainFeatures.size();
		
		
		//sorted retain feature
		int[] retainFArray = new int[retainFeatures.size()];
		for(int i = 0; i < retainFeatures.size(); i++) {
			retainFArray[i] = retainFeatures.get(i);
		}
		Arrays.sort(retainFArray);
		List<Integer> sortedRetainFeatures = new ArrayList<>();
		for(int i = 0; i < retainFArray.length; i++) {
			sortedRetainFeatures.add(retainFArray[i]);
		}
		
		//construct new selected value local index, other value is set as 0
		int[] newSelectedValueLocalIndex = new int[orgValueNum];		
		int lastFeature = -1;
		int lastLocalIndex = 1;
		for(int i = 0; i < selectedValueNum; i++) {
			int Iindex = selectedValueList.get(i);
			int feature = valueFeatureMap[Iindex];
			int localIndex = 1;
			if(feature == lastFeature) {
				localIndex = lastLocalIndex + 1;
			}
			newSelectedValueLocalIndex[Iindex] = localIndex;
			lastLocalIndex = localIndex;
			lastFeature = feature;		
		}
			
		// construct attributes of new instances
		FastVector newAttributes = new FastVector(newFeatureNum + 1);
		for(int i = 0; i < newFeatureNum; i++) {
			int orgFeatureIndex = sortedRetainFeatures.get(i);
			int thisValueNum = valueNumPerFeature.get(orgFeatureIndex) +1;
			//int thisValueNum = valueNumPerFeature[i] + 1;
			FastVector tmpFV = new FastVector(thisValueNum);
			for(int j = 0; j < thisValueNum; j++) {
				tmpFV.addElement(String.valueOf(j));
			}
			String attrName = "A" + String.valueOf(orgFeatureIndex); 
			Attribute tmpattr = new Attribute(attrName, tmpFV);
			newAttributes.addElement(tmpattr);
		}
		FastVector label = new FastVector(2);
		label.addElement(String.valueOf(1));
		label.addElement(String.valueOf(0));
		Attribute labelAttr = new Attribute("class", label);
		newAttributes.addElement(labelAttr);		
		Instances newInstances = new Instances(name + "-FS", newAttributes, 0);
		
		
		for(int i = 0; i < orgInstances.numInstances(); i++) {
			Instance instance = orgInstances.instance(i);
			//revise values
			for(int j = 0; j < instance.numAttributes() - 1; j++) {
				double value = instance.value(j);
				if(sortedRetainFeatures.contains(j)) {
					int generalValueIndex = firstValueIndex[j] + (int) value;
					instance.setValue(j, newSelectedValueLocalIndex[generalValueIndex]);
				}
			}			
			//delete other features
			Instance newInstance = new Instance(instance);
			for(int j = newInstance.numAttributes() - 2 ; j >= 0; j--) {
				if(!sortedRetainFeatures.contains(j)) {
					newInstance.deleteAttributeAt(j);
				}
			}
			newInstances.add(newInstance);
		}

//		System.out.println(newInstances);		
		return newInstances;
	}
	
	
	
	
	public static void fsByTwoValueSubsetToCSV(
			Instances oldInstances,
			int[] firstValueIndex,
			int[] valueFeatureMap,
			List<Integer> outlierValueList,
			List<Integer> normalValueList,
			String outPath) throws IOException {
		
		int nObjects = oldInstances.numInstances();		
		Instances newInstances = new Instances(oldInstances);				
		int oldFeatureNum = oldInstances.numAttributes();
		
		
		int outlierValueNum = outlierValueList.size();
		
		
		// construct retain feature list 
		// construct an array to indicate value number per features	
		List<Integer> retainFeatures = new ArrayList<>();
		Map<Integer, Integer> valueNumPerFeature = new HashMap<>();		
		for(int i = 0; i < outlierValueNum; i++) {
			int valueIndex = outlierValueList.get(i);
			int featureIndex = valueFeatureMap[valueIndex];
			if(!retainFeatures.contains(featureIndex)) {
				retainFeatures.add(featureIndex);
				valueNumPerFeature.put(featureIndex, 1);
			}else {
				int tmpNum = valueNumPerFeature.get(featureIndex);
				valueNumPerFeature.replace(featureIndex, tmpNum+1);
			}						
		}
		int newFeatureNum = retainFeatures.size();
		
		
		List<Integer> deletFeatureList = new ArrayList<>();
		for(int i = 0; i < oldFeatureNum-1; i++) {
			if(!retainFeatures.contains(i)) {
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
		
		
		int[][] newInstancesMatrix = new int[nObjects][newFeatureNum+1];
		for(int i = 0; i < nObjects; i++) {
			Instance instance = newInstances.instance(i);
			for(int j = 0; j < newFeatureNum+1; j++) {
				int localValueIndex = (int) instance.value(j);
				newInstancesMatrix[i][j] = localValueIndex;
			}
		}		
		matrix2csv(newInstancesMatrix, outPath);
				
//		BufferedWriter writer = new BufferedWriter(new FileWriter(outPath));
//		writer.write(newInstances.toString());
//		writer.flush();
//		writer.close();
	}
	
	

	
	
	
	public static void saveInstances(Instances instances, String path){
		ArffSaver saver = new ArffSaver();
	    saver.setInstances(instances);
	    try {
	    	saver.setFile(new File(path));
	        saver.writeBatch();
	    } catch (IOException e) {
	        e.printStackTrace();
	    }
	}
	
	
	
	public static void arff2csv(Instances instances, String outpath, List<Integer> selectedValue, int[] clusterInfo, double[] CPWithLabel) throws IOException{
		int nObjects = instances.numInstances();
		int nFeatures = instances.numAttributes();

		BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outpath));

		bufferedWriter.write("index" + ",");
		for(int i = 0; i < nFeatures; i++){
			int index = i+1;
			bufferedWriter.write("A" + index + ",");
		}		
		bufferedWriter.write("CPWithLabel" + ",");
		bufferedWriter.write("isChosen" + ",");
		bufferedWriter.write("cluster");
		bufferedWriter.newLine();
		
		
		for(int i = 0; i < nObjects; i++) {
			Instance instance = instances.instance(i);
			bufferedWriter.write(i + ",");

			for(int j = 0; j < nFeatures; j++) {
				double value = instance.value(j);
				String valueStr = String.valueOf(value);
				bufferedWriter.write(valueStr + ",");
			}
			
			double CP = CPWithLabel[i];
			String CPstr = String.valueOf(CP);
			bufferedWriter.write(CPstr + ",");
			
			if(selectedValue.contains(i)){
				bufferedWriter.write(1 + ",");
			}else {
				bufferedWriter.write(0 + ",");
			}
			
			bufferedWriter.write(clusterInfo[i] + "");
			

			bufferedWriter.newLine();
		}
		
		bufferedWriter.flush();
		bufferedWriter.close();
		
	}
	
	public static Instances matrix2Instances(double[][] matrix){
		int column = matrix[0].length;
		FastVector attributes = new FastVector();
		for(int i = 0; i < column; i++){
			attributes.addElement(new Attribute("A" + i));
		}
		int raw = matrix.length;
		
		Instances valueInstances = new Instances("valueInstances", attributes, 0);
			
		
		for(int i = 0; i < raw; i++){
			Instance instance = new Instance(attributes.size());
			for(int j = 0; j < column; j++){
				instance.setValue(j, matrix[i][j]);
			}
			valueInstances.add(instance);
		}
		
		return valueInstances;
	}

	
	
	public static void matrix2csv(int[][] matrix, String outpath) throws IOException{
		
		BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outpath));
		int nObjects = matrix.length;
		int nFeatures = matrix[0].length;
		
	
		for(int i = 0; i < nFeatures-1; i++) {
			bufferedWriter.write("A" + i +",");
		}
		bufferedWriter.write("class");
		bufferedWriter.newLine();
		
		
		for(int i = 0; i < nObjects; i++){
			for(int j = 0; j < nFeatures-1; j++){
				bufferedWriter.write(matrix[i][j] + ",");
			}
			bufferedWriter.write(matrix[i][nFeatures-1] + "");
			bufferedWriter.newLine();
		}		

		bufferedWriter.flush();
		bufferedWriter.close();
	}

	
	public static void array2csv(String outpath, double[] CPWithLabel, int[] clusterResult, double[][] value2OutlierClusterDistance) throws IOException{
		BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outpath));

		bufferedWriter.write("cp,cluster");
		bufferedWriter.newLine();
		
		
		for(int i = 0; i < CPWithLabel.length; i++){
			

			double distance1 = value2OutlierClusterDistance[i][0];
			double distance2 = value2OutlierClusterDistance[i][1];
			bufferedWriter.write(distance1 + ",");
			bufferedWriter.write(distance2 + ",");
		
			double CP = CPWithLabel[i];
			String CPstr = String.valueOf(CP);
			bufferedWriter.write(CPstr + ",");
			
			int cluster = clusterResult[i];
			String clusterStr = String.valueOf(cluster);
			bufferedWriter.write(clusterStr);

			

			bufferedWriter.newLine();
		}		

		bufferedWriter.flush();
		bufferedWriter.close();
	}
	
	
	public static void array2csv(String outpath, double[] CPWithLabel, int[] clusterResult) throws IOException{
		BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outpath));

		bufferedWriter.write("cp,cluster");
		bufferedWriter.newLine();
		
		
		for(int i = 0; i < CPWithLabel.length; i++){
			


		
			double CP = CPWithLabel[i];
			String CPstr = String.valueOf(CP);
			bufferedWriter.write(CPstr + ",");
			
			int cluster = clusterResult[i];
			String clusterStr = String.valueOf(cluster);
			bufferedWriter.write(clusterStr);

			

			bufferedWriter.newLine();
		}		

		bufferedWriter.flush();
		bufferedWriter.close();
	}
	
}
