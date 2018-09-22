package Louvain;

import java.io.BufferedReader;    
import java.io.BufferedWriter;    
import java.io.FileReader;    
import java.io.FileWriter;    
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;    
    
public class ModularityOptimizer    
{    
	
	public int cluster[];
	private int nCluster;
	private Map<Integer, Integer> clusterSizeMap;
	private double maxModularity;
	
    public static void main(String[] args) throws IOException {    	
    	String networkFilePath = "E:\\network2.txt";
    	Network network = readInputFile(networkFilePath);
    	ModularityOptimizer modularityOptimizer = new ModularityOptimizer();
    	modularityOptimizer.runLouvain(network);
    }    
	
    
	public int[] runLouvain(List<String> networkList) throws IOException {
		boolean  update;    
        
        double modularity, maxModularity, resolution2;    
        int  i, j, nClusters, nIterations, nRandomStarts;    
        int[] cluster;    
        Network network;    
        Random random;    

        nRandomStarts = 10;    
        nIterations = 5;    
  
        network = generateNetwork(networkList);
        

//        System.out.format("Number of nodes: %d%n", network.getNNodes());    
//        System.out.format("Number of edges: %d%n", network.getNEdges() / 2);    
//        System.out.println("Running Louvain algorithm" );    
            
        resolution2 = 1.0 / network.getTotalEdgeWeight();
    
        cluster = null;    
        nClusters = -1;    
        maxModularity = Double.NEGATIVE_INFINITY;    
        random = new Random(100);  
        for (i = 0; i < nRandomStarts; i++)    
        {    
            network.initSingletonClusters();    
            j = 0;    
            update = true;    
            do    
            {    
                update = network.runLouvainAlgorithm(resolution2, random);  
                j++;       
                modularity = network.calcQualityFunction(resolution2);    
            }    
            while ((j < nIterations) && update); 
            
            
            if (modularity > maxModularity) {                   
                cluster = network.getClusters();    
                nClusters = network.getNClusters();    
                maxModularity = modularity;    
            }             
        }    
        this.maxModularity = maxModularity;
        
        
        Map<Integer, Integer> clusterSizeMap = new HashMap<>();
        for(int o = 0; o < cluster.length; o++) {
        	int clusterId = cluster[o];
        	if(clusterSizeMap.containsKey(clusterId)) {
        		int count = clusterSizeMap.get(clusterId);
        		clusterSizeMap.replace(clusterId, count + 1);
        	}else {
        		clusterSizeMap.put(clusterId, 1);
        	}
        }
        this.clusterSizeMap = clusterSizeMap;
        
    
            
//        if (nRandomStarts == 1)    
//        {    
//            if (nIterations > 1)    
//                System.out.println();    
//            System.out.format("Modularity: %.8f%n", maxModularity);    
//        }    
//        else    
//            System.out.format("Maximum modularity in %d random starts: %f%n", nRandomStarts, maxModularity);    
        
        
        this.nCluster = nClusters;

        
        //writeOutputFile(outputFileName, cluster);  
        return cluster;
	}
	
	
	
	
	
	public int[] runLouvain(Network network) throws IOException {
		boolean  update;    
        
        double modularity, maxModularity, resolution2;    
        int  i, j, nClusters, nIterations, nRandomStarts;    
        int[] cluster;    
        Random random;    

        nRandomStarts = 10;    
        nIterations = 5;    

        System.out.format("Number of nodes: %d%n", network.getNNodes());    
        System.out.format("Number of edges: %d%n", network.getNEdges() / 2);    
        System.out.println("Running Louvain algorithm" );    
        System.out.println();    
            
        resolution2 = 1.0 / network.getTotalEdgeWeight();
    
        cluster = null;    
        nClusters = -1;    
        maxModularity = Double.NEGATIVE_INFINITY;    
        random = new Random(100);  
        for (i = 0; i < nRandomStarts; i++)    
        {    
            if (nRandomStarts > 1)    
                System.out.format("Random start: %d%n", i + 1);    
    
            network.initSingletonClusters();
    
            j = 0;    
            update = true;    
            do    
            {    
                if (nIterations > 1) {
                    System.out.format("Iteration: %d%n", j + 1);    
                }
                
                update = network.runLouvainAlgorithm(resolution2, random);  
                j++;    
    
                modularity = network.calcQualityFunction(resolution2);    
                
    
                if (nIterations > 1)    
                    System.out.format("Modularity: %.10f%n", modularity);    
            }    
            while ((j < nIterations) && update); 
            
            if (modularity > maxModularity) {                   
                cluster = network.getClusters();    
                nClusters = network.getNClusters();    
                maxModularity = modularity;    
            }    
    
//            if (nRandomStarts > 1)    
//            {    
//                  if (nIterations == 1)    
//                      System.out.format("Modularity: %.8f%n", modularity);    
//                  System.out.println();    
//            } 
            
        }    
        this.maxModularity = maxModularity;
        
        
        //count the node number of each cluster
        Map<Integer, Integer> clusterSizeMap = new HashMap<>();
        for(int o = 0; o < cluster.length; o++) {
        	int clusterId = cluster[o];
        	if(clusterSizeMap.containsKey(clusterId)) {
        		int count = clusterSizeMap.get(clusterId);
        		clusterSizeMap.replace(clusterId, count + 1);
        	}else {
        		clusterSizeMap.put(clusterId, 1);
        	}
        }
        this.clusterSizeMap = clusterSizeMap;
        
    
            
        if (nRandomStarts == 1)    
        {    
            if (nIterations > 1)    
                System.out.println();    
            System.out.format("Modularity: %.8f%n", maxModularity);    
        }    
        else    
            System.out.format("Maximum modularity in %d random starts: %f%n", nRandomStarts, maxModularity);    
        
        
        this.nCluster = nClusters;

        //network.printClusterStatus();
        
        System.out.println(clusterSizeMap);

//        for(i = 0; i < cluster.length; i++) {
//        	System.out.println(cluster[i] );
//        }
//        System.out.println();
        
        
        return cluster;
	}
	
	
	
	
	
	
	
	
	public int getNClusters() {
		return nCluster;
	}

    
	public Map<Integer, Integer> getClusterSizeMap() {
		return clusterSizeMap;
	}
	

	
	
	
	
	
	
	
	
    private static Network readInputFile(String fileName) throws IOException    
    {    
        BufferedReader bufferedReader;    
        double[] edgeWeight1, edgeWeight2, nodeWeight;    
        int i, j, nEdges, nLines, nNodes;    
        int[] firstNeighborIndex, neighbor, nNeighbors, node1, node2;    
        Network network;    
        String[] splittedLine;    
    
        
        /**
         * get the number of lines
         */
        bufferedReader = new BufferedReader(new FileReader(fileName));    
        nLines = 0;    
        while (bufferedReader.readLine() != null)    
            nLines++;        
        bufferedReader.close();
        //System.out.println("nLines: " + nLines);
        
        /**
         * get the node1 and node2 of each line
         * get the edgeWeight
         * get the number of node
         */
        bufferedReader = new BufferedReader(new FileReader(fileName));      
        node1 = new int[nLines];    
        node2 = new int[nLines];    
        edgeWeight1 = new double[nLines];    
        i = -1;    
        for (j = 0; j < nLines; j++)    
        {    
            splittedLine = bufferedReader.readLine().split(" ");    
            node1[j] = Integer.parseInt(splittedLine[0]);    
            if (node1[j] > i)    
                i = node1[j]; 
            
            node2[j] = Integer.parseInt(splittedLine[1]);    
            if (node2[j] > i)    
                i = node2[j];    
            edgeWeight1[j] = (splittedLine.length > 2) ? Double.parseDouble(splittedLine[2]) : 1;    
        }    
        nNodes = i + 1;       
        bufferedReader.close(); 
        //System.out.print("edgeWeight1: ");
//        for(int o = 0; o < edgeWeight1.length; o++) {
//        	System.out.print(edgeWeight1[o] + " ");
//        }
//        System.out.println();
//        System.out.println("nNodes: " + nNodes);
    
        
        
        /**
         * get the number of neighbors of each node
         */
        nNeighbors = new int[nNodes];    
        for (i = 0; i < nLines; i++) {
            if (node1[i] < node2[i]) {    
                nNeighbors[node1[i]]++;    
                nNeighbors[node2[i]]++;    
            }  
        }

//        System.out.print("nNeighbors: ");
//        for(int o = 0; o < nNeighbors.length; o++) {
//        	System.out.print(nNeighbors[o] + " ");
//        }
//        System.out.println();
  

        
        firstNeighborIndex = new int[nNodes + 1];    
        nEdges = 0;    
        for (i = 0; i < nNodes; i++)    
        {    
            firstNeighborIndex[i] = nEdges;    
            nEdges += nNeighbors[i];    
        }    
        firstNeighborIndex[nNodes] = nEdges; 
        
//        System.out.println("nEdges: "+ nEdges);
//        System.out.print("firstNeighborIndex ");
//        for(int o = 0; o < firstNeighborIndex.length; o++) {
//        	System.out.print(firstNeighborIndex[o] + " ");
//        }
//        System.out.println();
        
        
        
    
        //neighbor is the neighbor node index of each node
        //firstNeighborIndex is the first neighbor index of each node
        neighbor = new int[nEdges];    
        edgeWeight2 = new double[nEdges];    
        Arrays.fill(nNeighbors, 0);    
        for (i = 0; i < nLines; i++)    
            if (node1[i] < node2[i])    
            {    
                j = firstNeighborIndex[node1[i]] + nNeighbors[node1[i]];    
                neighbor[j] = node2[i];    
                edgeWeight2[j] = edgeWeight1[i];    
                nNeighbors[node1[i]]++;    
                j = firstNeighborIndex[node2[i]] + nNeighbors[node2[i]];    
                neighbor[j] = node1[i];    
                edgeWeight2[j] = edgeWeight1[i];    
                nNeighbors[node2[i]]++;    
            }    
    
       
        {    
            nodeWeight = new double[nNodes];    
            for (i = 0; i < nEdges; i++)    
                nodeWeight[neighbor[i]] += edgeWeight2[i];    
            network = new Network(nNodes, firstNeighborIndex, neighbor, edgeWeight2, nodeWeight);    
        }    
        
        
        
//        System.out.print("edgeWeight2: ");
//        for(int o = 0; o < edgeWeight2.length; o++) {
//        	System.out.print(edgeWeight2[o]+ " ");
//        }
//        System.out.println();
        
        
//        System.out.print("nodeWeight: "); 
//        for(int o = 0; o < nodeWeight.length; o++) {
//        	System.out.print(nodeWeight[o]+ " ");
//        }
//        System.out.println();
        

        return network;    
    }    
    
    
    public double getMaxModularity() {
		return maxModularity;
	}
		
	
    
    
    
    private static Network generateNetwork(List<String> edgeList) throws IOException    
    {    
        BufferedReader bufferedReader;    
        double[] edgeWeight1, edgeWeight2, nodeWeight;    
        int i, j, nEdges, nLines, nNodes;    
        int[] firstNeighborIndex, neighbor, nNeighbors, node1, node2;    
        Network network;    
        String[] splittedLine;    
    
        
        /**
         * get the number of lines
         */
        nLines = edgeList.size();
        
        
        /**
         * get the node1 and node2 of each line
         * get the edgeWeight
         * get the number of node
         */
        node1 = new int[nLines];    
        node2 = new int[nLines];    
        edgeWeight1 = new double[nLines];    
        i = -1;    
        for (j = 0; j < nLines; j++)    
        {    
            splittedLine = edgeList.get(j).split(" ");
            node1[j] = Integer.parseInt(splittedLine[0]);    
            if (node1[j] > i)    
                i = node1[j]; 
            
            node2[j] = Integer.parseInt(splittedLine[1]);    
            if (node2[j] > i)    
                i = node2[j];    
            edgeWeight1[j] = (splittedLine.length > 2) ? Double.parseDouble(splittedLine[2]) : 1;    
        }    
        nNodes = i + 1;       

        

        
        /**
         * get the number of neighbors of each node
         */
        nNeighbors = new int[nNodes];    
        for (i = 0; i < nLines; i++) {
            if (node1[i] < node2[i]) {    
                nNeighbors[node1[i]]++;    
                nNeighbors[node2[i]]++;    
            }  
        }


  

        
        firstNeighborIndex = new int[nNodes + 1];    
        nEdges = 0;    
        for (i = 0; i < nNodes; i++)    
        {    
            firstNeighborIndex[i] = nEdges;    
            nEdges += nNeighbors[i];    
        }    
        firstNeighborIndex[nNodes] = nEdges; 
        

        
    
        //neighbor is the neighbor node index of each node
        //firstNeighborIndex is the first neighbor index of each node
        neighbor = new int[nEdges];    
        edgeWeight2 = new double[nEdges];    
        Arrays.fill(nNeighbors, 0);    
        for (i = 0; i < nLines; i++)    
            if (node1[i] < node2[i])    
            {    
                j = firstNeighborIndex[node1[i]] + nNeighbors[node1[i]];    
                neighbor[j] = node2[i];    
                edgeWeight2[j] = edgeWeight1[i];    
                nNeighbors[node1[i]]++;    
                j = firstNeighborIndex[node2[i]] + nNeighbors[node2[i]];    
                neighbor[j] = node1[i];    
                edgeWeight2[j] = edgeWeight1[i];    
                nNeighbors[node2[i]]++;    
            }    
    
       
        {    
            nodeWeight = new double[nNodes];    
            for (i = 0; i < nEdges; i++)    
                nodeWeight[neighbor[i]] += edgeWeight2[i];    
            network = new Network(nNodes, firstNeighborIndex, neighbor, edgeWeight2, nodeWeight);    
        }    
        

        return network;    
    }    
    
    
    
    
    
    
//    private static void writeOutputFile(String fileName, int[] cluster) throws IOException    
//    {    
//        BufferedWriter bufferedWriter;    
//        int i;    
//    
//        bufferedWriter = new BufferedWriter(new FileWriter(fileName));    
//    
//        for (i = 0; i < cluster.length; i++)    
//        {    
//            bufferedWriter.write(Integer.toString(cluster[i]));    
//            bufferedWriter.newLine();    
//        }    
//    
//        bufferedWriter.close();    
//    }    
}