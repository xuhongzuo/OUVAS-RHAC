package OD;


import java.io.File;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import Louvain.ModularityOptimizer;
import Utils.ArffUtils;



/**
 * The main class of RHAC
 * @author Hongzuo Xu
 * 2018.5
 *
 */
public class ODUtils{

	
	public static void main(String[] args) throws Exception{
		// path can be the directory of arff files or the single arff 
		// optionInt 0 for directly OD, 1 for Feature Selection
		// Please create a folder named RHFS in the data folder if you want to perform FS
		String path = "E:\\data\\CIKMarff\\";
		int optionInt = 0;
	
//		String path = args[0];
//		String option = args[1];
//		int optionInt = Integer.parseInt(option);
			
		if(path.endsWith(".arff")) {
			runRHAC(path, optionInt);
			//runDataIndicator(path);
						
		}else {
			List<String> datasetList = buildDataSetsPathList(path);
			for(int i = 0; i <datasetList.size(); i++) {
				runRHAC(datasetList.get(i),optionInt);
				//runDataIndicator(datasetList.get(i));
			}
		}	
	}
	
	
	
	
	/**
	 * 
	 * @param path
	 * @param options 0 for OD,1 for FS
	 * @throws Exception
	 */
	public static void runRHAC(String path, int options) throws Exception{
    	String name = getDatasetName(path);
    	System.out.print(name + ",");
    	
    	long beginTime = System.currentTimeMillis();
    	
		DataConstructor dataConstructor = new DataConstructor();
		dataConstructor.dataPrepareFromArff(path);
		dataConstructor.calCPWithLabel();

		
		double[][] transitionMatrix = dataConstructor.getTransitionMatrix();
		
		List<String> networkList =  dataConstructor.getNetworkList(0.0);		
		ModularityOptimizer modularityOptimizer = new ModularityOptimizer();
		int[] clusterInfo = modularityOptimizer.runLouvain(networkList);
		
		
		List<Integer> fullValueList = new ArrayList<>();
		for(int i = 0; i < dataConstructor.getNValues(); i++) {
			fullValueList.add(i);
		}
				
//    	System.out.print( dataConstructor.getNFeatures() + ",");
//    	System.out.print(dataConstructor.getNValues() + ",");
//    	System.out.print( dataConstructor.getNObjects() + ", ");
		
		
		//identify normal-value cluster
		double[] muValueOutlierness = OutliernessEvaluator.calInitValueOutlierness(
				dataConstructor.getFirstValueIndex(), 
				dataConstructor.getValueFrequency());		
		double[] clusterOutlierness = OutliernessEvaluator.calcClusterOutlierness(clusterInfo, 
				modularityOptimizer.getNClusters(), 
				muValueOutlierness);		
		int normalClusterIndex = OutliernessEvaluator.getNormalClusterId(clusterOutlierness);	

		
		//recognize an outlying value cluster
		double[] tauValueOutlierness = OutliernessEvaluator.calcValueOutliernessByBCVc(dataConstructor.getTransitionMatrix(), 
				muValueOutlierness, 
				clusterInfo, 
				normalClusterIndex, 
				modularityOptimizer.getNClusters());	
		double[] clusterOutlierness2 = OutliernessEvaluator.calcClusterOutlierness(clusterInfo, 
			modularityOptimizer.getNClusters(), 
			tauValueOutlierness); 
		int outlierClusterIndex = OutliernessEvaluator.getOutlierClusterId(clusterOutlierness2, normalClusterIndex);	
	
		//discover more outlying value cluster(s)
		double[][] clusterValueCouplingMatrix = OutliernessEvaluator.calcClusterValueCoupling(clusterInfo,
				modularityOptimizer.getNClusters(), transitionMatrix);
		double[][] clusterDistance = OutliernessEvaluator.calcClusterDistance(clusterInfo, 
				modularityOptimizer.getNClusters(), 
				transitionMatrix, clusterValueCouplingMatrix, dataConstructor.getNodeDegree());
		List<Integer> outlyingClusters = OutliernessEvaluator.determineOutlierCluster(clusterDistance, 
				normalClusterIndex, outlierClusterIndex);
				
		//revise cluster info to 1-outlier 0-normal 2-other		
		int[] standardClusterInfo = OutliernessEvaluator.reviseCluserInfo(clusterInfo, outlyingClusters, normalClusterIndex);		


		//value-level refinement
		double[][] value2ClusterEdgeWeight = OutliernessEvaluator.calcValue2ClusterEdgeWeight(standardClusterInfo, 
				dataConstructor.getSimilarityMatrix());
		int [] newClusterInfo = OutliernessEvaluator.refineOutlyingValueSet(standardClusterInfo, value2ClusterEdgeWeight);		
		List<Integer> outlyingValues = OutliernessEvaluator.getValueListByCluster(newClusterInfo, 1);	
		//System.out.print("outlyingValueSize," + outlyingValues.size() + ","); 
		
	
		//directly detect outliers
		if(options == 0) {
			double[] valueScore = OutliernessEvaluator.valueOutliernessScoring(dataConstructor.getConditionalPossibility(), 
					tauValueOutlierness, 
					outlyingValues);
			double[] objectScore = OutliernessEvaluator.objectOutliernessScoreing(valueScore, 
					dataConstructor.getInstances(),
					dataConstructor.getFirstValueIndex());
			
	    	long endTime = System.currentTimeMillis();

			Hashtable<Integer, Double> objectScoreTable = OutliernessEvaluator.GenerateObjectScoreMap(objectScore);		
	    	Evaluation evaluation = new Evaluation("outlier");
	    	double auc = evaluation.computeAUCAccordingtoOutlierRanking(dataConstructor.getListOfClass(), 
	    			evaluation.rankInstancesBasedOutlierScores(objectScoreTable));
	    	double presion = evaluation.computePresion(dataConstructor.getListOfClass(), 
	    			evaluation.rankInstancesBasedOutlierScores(objectScoreTable));	
	    	System.out.format("auc,%.4f," , auc);
	    	System.out.format("presion,%.4f," , presion);
	    	System.out.format("%.4fs%n", (endTime - beginTime) / 1000.0); 	    	
		}
		
				
		//Feature Selection
		else {
			List<Integer> remainValues = OutliernessEvaluator.getValueListByCluster(newClusterInfo, 1);	
			List<Integer> normalValues = OutliernessEvaluator.getValueListByCluster(newClusterInfo, 0);

			List<Integer> remainingFeatureList = OutliernessEvaluator.getRemainingFeatureList(remainValues, 
					dataConstructor.getFirstValueIndex());
			
			String outRootpath = "";
			String outPath = "";			
			//windows
			if(path.contains("\\")) {
				int index = path.lastIndexOf("\\");
				outRootpath = path.substring(0, index);
				outPath = outRootpath + "\\" + "RHFS\\" + name + "-FS_" + remainingFeatureList.size() + ".csv";
			}else {
				//linux
				int index = path.lastIndexOf("/");
				outRootpath = path.substring(0, index);
				outPath = outRootpath + "/" + "RHFS/" + name + "-FS_" + remainingFeatureList.size() + ".csv";
			}
				
			
			ArffUtils.fsByTwoValueSubsetToCSV(dataConstructor.getInstances(), 
					dataConstructor.getFirstValueIndex(), 
					dataConstructor.getValueFeatureMap(), 
					remainValues, 
					normalValues, 
					outPath);
			
			
//			Instances newInstances = ArffUtils.fsByValueSubset(remainValues, 
//					dataConstructor.getInstances(), 
//					dataConstructor.getValueFeatureMap(), 
//					dataConstructor.getFirstValueIndex());
//			
//			BufferedWriter writer = new BufferedWriter(new FileWriter(outPath));
//			writer.write(newInstances.toString());
//			writer.flush();
//			writer.close();									
//	        OutliernessEvaluator.createNewInstancesByFS2(remainingFeatureList, 
//	        		dataConstructor.getInstances(),
//	        		dataConstructor.getFirstValueIndex(),
//	        		remainValues,
//	        		normalValues,
//	        		dataConstructor.getFeatureModeValueIndex(), outPath);
	        
			
	        int usedValueNum = remainValues.size();
	        int remainedFeatureNum = remainingFeatureList.size();
	        int orgfeatureNum = dataConstructor.getInstances().numAttributes()-1;

	    	System.out.print("RemainedValueNum," +  usedValueNum + ",");
	    	System.out.print("OrgFeatureNum," + orgfeatureNum + ",");
	    	System.out.print("remainedFeatureNum," +  remainedFeatureNum + ",");
	    	System.out.println();
		}
	}
	

	
	public static void runDataIndicator(String path) throws Exception {
		DataConstructor dataConstructor = new DataConstructor();
		
    	String name = getDatasetName(path);
    	System.out.print(name + ",");
    	dataConstructor.dataPrepareFromArff(path);
		dataConstructor.calCPWithLabel();
		dataConstructor.calSimWithLabel();
    	
    	double separability = DataIndicator.calcSeperability(dataConstructor.getInstances(), 
    			dataConstructor.getValueFrequency(),
    			dataConstructor.getFirstValueIndex(), 
    			dataConstructor.getListOfClass());
    	
    	double noisyRate = DataIndicator.calcNoisyRate(dataConstructor.getInstances(), 
    			dataConstructor.getValueFrequency(),
    			dataConstructor.getFirstValueIndex(), 
    			dataConstructor.getListOfClass());
    	
    	System.out.format("mfe,%.4f," , separability);
    	System.out.format("fnl,%.4f," , noisyRate);
    	System.out.println();
	}
	
	

	
	
	public static String getDatasetName(String path) {
		String name = "";
		
		//windows
		if(path.contains("\\")) {
			String[] splitedString = path.split("\\\\");
			String full_name = splitedString[splitedString.length-1];
			name = full_name.substring(0, full_name.length()-5);
		}else {
			//linux
			String[] splitedString = path.split("/");
			String full_name = splitedString[splitedString.length-1];
			name = full_name.substring(0, full_name.length()-5);
//			name = splitedString[splitedString.length-1].split("\\.")[0];
		}
		
		return name;
	}
	
	
	
    /**
     * to store the file names contained in a folder
     * @param dataSetFilesPath the path of the folder
     */
    public static List<String> buildDataSetsPathList(String dataSetFilesPath)
    {
        File filePath = new File(dataSetFilesPath);
        String[] fileNameList =  filePath.list();
        int dataSetFileCount = 0;
        for (int count=0;count < fileNameList.length;count++)
        {
            // System.out.println(fileNameList[count]);
            if (fileNameList[count].toLowerCase().endsWith(".arff"))
            {
                dataSetFileCount = dataSetFileCount +1;
            }
        }
        List<String> dataSetFullNameList = new ArrayList<>();

        dataSetFileCount = 0;
        for (int count =0; count < fileNameList.length; count++)
        {
            if (fileNameList[count].toLowerCase().endsWith(".arff"))
            {
                if(dataSetFilesPath.contains("\\")) {
                    dataSetFullNameList.add(dataSetFilesPath+"\\"+fileNameList[count]);
                }else {
                    dataSetFullNameList.add(dataSetFilesPath+"/"+fileNameList[count]);

				}

                dataSetFileCount = dataSetFileCount +1;
            }
        }
        
        return dataSetFullNameList;
    }    
	
}
