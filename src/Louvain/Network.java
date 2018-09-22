package Louvain;

import java.io.Serializable;
import java.util.Random;

public class Network implements Cloneable, Serializable {
	
	private static final long serialVersionUID = 1;

	private int nNodes;
	private int[] firstNeighborIndex;
	
	private int[] neighbor;
	private double[] edgeWeight;

	private double[] nodeWeight;
	private int nClusters;
	private int[] cluster;

	private double[] clusterWeight;
	private int[] nNodesPerCluster;
	private int[][] nodePerCluster;
	private boolean clusteringStatsAvailable;

	public Network(int nNodes, int[] firstNeighborIndex, int[] neighbor, double[] edgeWeight, double[] nodeWeight) {
		this(nNodes, firstNeighborIndex, neighbor, edgeWeight, nodeWeight, null);
	}

	public Network(int nNodes, int[] firstNeighborIndex, int[] neighbor, double[] edgeWeight, double[] nodeWeight,
			int[] cluster) {
		int i, nEdges;
		this.nNodes = nNodes;
		this.firstNeighborIndex = firstNeighborIndex;
		this.neighbor = neighbor;
		if (edgeWeight == null) {
			nEdges = neighbor.length;
			this.edgeWeight = new double[nEdges];
			for (i = 0; i < nEdges; i++)
				this.edgeWeight[i] = 1;
		} else
			this.edgeWeight = edgeWeight;

		if (nodeWeight == null) {
			this.nodeWeight = new double[nNodes];
			for (i = 0; i < nNodes; i++)
				this.nodeWeight[i] = 1;
		} else
			this.nodeWeight = nodeWeight;
	}

	public int getNNodes() {
		return nNodes;
	}

	public int getNEdges() {
		return neighbor.length;
	}

	public double getTotalEdgeWeight() 
	{
		double totalEdgeWeight;
		int i;
		totalEdgeWeight = 0;
		for (i = 0; i < neighbor.length; i++)
			totalEdgeWeight += edgeWeight[i];
		return totalEdgeWeight;
	}

	public double[] getEdgeWeights() {
		return edgeWeight;
	}

	public double[] getNodeWeights() {
		return nodeWeight;
	}

	public int getNClusters() {
		return nClusters;
	}

	public int[] getClusters() {
		return cluster;
	}
	
	public int[][] getNodePerCluster() {
		return nodePerCluster;
	}

	public void initSingletonClusters() {
		int i;
		nClusters = nNodes;
		cluster = new int[nNodes];
		for (i = 0; i < nNodes; i++)
			cluster[i] = i;

		deleteClusteringStats();
	}

	/**
	 * merge original cluster to new cluster
	 * size of newCluster is the number of original cluster
	 * @param newCluster
	 */
	public void mergeClusters(int[] newCluster) {
		int i, j, k;
		if (cluster == null)
			return;
		i = 0;
		for (j = 0; j < nNodes; j++) {
			k = newCluster[cluster[j]];
			if (k > i)
				i = k;
			cluster[j] = k;
		}
		nClusters = i + 1;
		deleteClusteringStats();
	}

	
	public Network getReducedNetwork() {
		double[] reducedNetworkEdgeWeight1, reducedNetworkEdgeWeight2;
		int i, j, k, l, m, reducedNetworkNEdges1, reducedNetworkNEdges2;
		int[] reducedNetworkNeighbor1, reducedNetworkNeighbor2;
		Network reducedNetwork;
		if (cluster == null)
			return null;
		if (!clusteringStatsAvailable)
			calcClusteringStats();

		reducedNetwork = new Network();
		reducedNetwork.nNodes = nClusters;
		reducedNetwork.firstNeighborIndex = new int[nClusters + 1];
		reducedNetwork.nodeWeight = new double[nClusters];
		
		reducedNetworkNeighbor1 = new int[neighbor.length];
		reducedNetworkEdgeWeight1 = new double[edgeWeight.length];
		
		reducedNetworkNeighbor2 = new int[nClusters - 1];
		reducedNetworkEdgeWeight2 = new double[nClusters];
		reducedNetworkNEdges1 = 0;
		
		for (i = 0; i < nClusters; i++) {
			reducedNetworkNEdges2 = 0;
			for (j = 0; j < nodePerCluster[i].length; j++) {
				//k is the node id of the jth node in the cluster i
				k = nodePerCluster[i][j]; 
				for (l = firstNeighborIndex[k]; l < firstNeighborIndex[k + 1]; l++) {
					m = cluster[neighbor[l]]; // m是k的在l位置的邻居节点所属的簇id
					if (m != i) {
						if (reducedNetworkEdgeWeight2[m] == 0) {
							reducedNetworkNeighbor2[reducedNetworkNEdges2] = m;
							reducedNetworkNEdges2++;
						}
						reducedNetworkEdgeWeight2[m] += edgeWeight[l];
					}
				}
				reducedNetwork.nodeWeight[i] += nodeWeight[k];
			}
			for (j = 0; j < reducedNetworkNEdges2; j++) {
				reducedNetworkNeighbor1[reducedNetworkNEdges1 + j] = reducedNetworkNeighbor2[j];
				reducedNetworkEdgeWeight1[reducedNetworkNEdges1
						+ j] = reducedNetworkEdgeWeight2[reducedNetworkNeighbor2[j]];
				reducedNetworkEdgeWeight2[reducedNetworkNeighbor2[j]] = 0;
			}
			reducedNetworkNEdges1 += reducedNetworkNEdges2;
			reducedNetwork.firstNeighborIndex[i + 1] = reducedNetworkNEdges1;
		}
		reducedNetwork.neighbor = new int[reducedNetworkNEdges1];
		reducedNetwork.edgeWeight = new double[reducedNetworkNEdges1];
		System.arraycopy(reducedNetworkNeighbor1, 0, reducedNetwork.neighbor, 0, reducedNetworkNEdges1);
		System.arraycopy(reducedNetworkEdgeWeight1, 0, reducedNetwork.edgeWeight, 0, reducedNetworkNEdges1);
		return reducedNetwork;
	}
	
	public double calcQualityFunction(double resolution) {
		double qualityFunction, totalEdgeWeight;
		int i, j, k;
		if (cluster == null)
			return Double.NaN;
		if (!clusteringStatsAvailable)
			calcClusteringStats();
		qualityFunction = 0;
		totalEdgeWeight = 0;
		for (i = 0; i < nNodes; i++) {
			j = cluster[i];
			for (k = firstNeighborIndex[i]; k < firstNeighborIndex[i + 1]; k++) {
				if (cluster[neighbor[k]] == j)
					qualityFunction += edgeWeight[k];
				totalEdgeWeight += edgeWeight[k];
			}
		}
		for (i = 0; i < nClusters; i++)
			qualityFunction -= clusterWeight[i] * clusterWeight[i] * resolution;
		

		qualityFunction /= totalEdgeWeight;
		return qualityFunction;
	}
	
	public boolean runLocalMovingAlgorithm(double resolution) {
		return runLocalMovingAlgorithm(resolution, new Random());
	}
	public boolean runLocalMovingAlgorithm(double resolution, Random random) {
		boolean update;
		double maxQualityFunction, qualityFunction;
		double[] clusterWeight, edgeWeightPerCluster;
		int bestCluster, i, j, k, l, nNeighboringClusters, nStableNodes, nUnusedClusters;
		int[] neighboringCluster, newCluster, nNodesPerCluster, nodeOrder, unusedCluster;
		if ((cluster == null) || (nNodes == 1))
			return false;
		
		update = false;
		
		//calculate cluster weight and nNodesPerCluster
		clusterWeight = new double[nNodes];
		nNodesPerCluster = new int[nNodes];
		for (i = 0; i < nNodes; i++) {
			clusterWeight[cluster[i]] += nodeWeight[i];
			nNodesPerCluster[cluster[i]]++;
		}
		
		
		//calculate unused cluster(cluster node number is 0)
		//nUnusedCluster is the number of unused cluster
		nUnusedClusters = 0;
		unusedCluster = new int[nNodes];
		for (i = 0; i < nNodes; i++)
			if (nNodesPerCluster[i] == 0) {
				unusedCluster[nUnusedClusters] = i;
				nUnusedClusters++;
			}
		
		
		//rerank the order of nodes
		nodeOrder = new int[nNodes];
		for (i = 0; i < nNodes; i++)
			nodeOrder[i] = i;
		for (i = 0; i < nNodes; i++) {
			j = random.nextInt(nNodes);
			k = nodeOrder[i];
			nodeOrder[i] = nodeOrder[j];
			nodeOrder[j] = k;
		}
		
		
		edgeWeightPerCluster = new double[nNodes];
		neighboringCluster = new int[nNodes - 1];
		nStableNodes = 0;
		i = 0;
		do {
			j = nodeOrder[i];
			nNeighboringClusters = 0;
			//for each neighbor of node j, record the neighboring cluster 
			for (k = firstNeighborIndex[j]; k < firstNeighborIndex[j + 1]; k++) {
				l = cluster[neighbor[k]];
				if (edgeWeightPerCluster[l] == 0) {
					neighboringCluster[nNeighboringClusters] = l;
					nNeighboringClusters++;
				}
				edgeWeightPerCluster[l] += edgeWeight[k];
			}
						
			//move node j out of the original cluster
			clusterWeight[cluster[j]] -= nodeWeight[j];
			nNodesPerCluster[cluster[j]]--;
			if (nNodesPerCluster[cluster[j]] == 0) {
				unusedCluster[nUnusedClusters] = cluster[j];
				nUnusedClusters++;
			}
			
			//initial maxQualityFunction is 0 to ensure that the modularity is increasing
			//bestCluster is the cluster id that the next destination of node j 
			bestCluster = -1;
			maxQualityFunction = 0;
			for (k = 0; k < nNeighboringClusters; k++) {
				l = neighboringCluster[k];
				qualityFunction = edgeWeightPerCluster[l] - nodeWeight[j] * clusterWeight[l] * resolution;
				if ((qualityFunction > maxQualityFunction)
						|| ((qualityFunction == maxQualityFunction) && (l < bestCluster))) {
					bestCluster = l;
					maxQualityFunction = qualityFunction;
				}
				edgeWeightPerCluster[l] = 0;
			}
			
			
			if (maxQualityFunction == 0) {
				bestCluster = unusedCluster[nUnusedClusters - 1];
				nUnusedClusters--;
			}
			
			
			clusterWeight[bestCluster] += nodeWeight[j];
			nNodesPerCluster[bestCluster]++;
			
			//if node j not move, then node j is stable
			if (bestCluster == cluster[j])
				nStableNodes++;
			else { 	
				cluster[j] = bestCluster;
				nStableNodes = 1;
				update = true;
			}
			i = (i < nNodes - 1) ? (i + 1) : 0;
		} while (nStableNodes < nNodes);
		
		
		newCluster = new int[nNodes];
		nClusters = 0;
		for (i = 0; i < nNodes; i++)
			if (nNodesPerCluster[i] > 0) {
				newCluster[i] = nClusters;
				nClusters++;
			}
		for (i = 0; i < nNodes; i++)
			cluster[i] = newCluster[cluster[i]];
		deleteClusteringStats();
		return update;
	}
	
	public boolean runLouvainAlgorithm(double resolution) {
		return runLouvainAlgorithm(resolution, new Random());
	}
	
	public boolean runLouvainAlgorithm(double resolution, Random random) {
		boolean update, update2;
		Network reducedNetwork;
		if ((cluster == null) || (nNodes == 1)) {
			return false;
		}
		
		update = runLocalMovingAlgorithm(resolution, random);
		
		if (nClusters < nNodes) {
			reducedNetwork = getReducedNetwork();	
			reducedNetwork.initSingletonClusters();
			update2 = reducedNetwork.runLouvainAlgorithm(resolution, random);

			if (update2) {
				update = true;
				mergeClusters(reducedNetwork.getClusters());
			}
		}
		deleteClusteringStats();
		return update;
	}
	
	
	private Network() {
	}
	
	private void calcClusteringStats() {
		int i, j;
		clusterWeight = new double[nClusters];
		nNodesPerCluster = new int[nClusters];
		nodePerCluster = new int[nClusters][];
		for (i = 0; i < nNodes; i++) {
			clusterWeight[cluster[i]] += nodeWeight[i];
			nNodesPerCluster[cluster[i]]++;
		}
		for (i = 0; i < nClusters; i++) {
			nodePerCluster[i] = new int[nNodesPerCluster[i]];
			nNodesPerCluster[i] = 0;
		}
		for (i = 0; i < nNodes; i++) {
			j = cluster[i];
			nodePerCluster[j][nNodesPerCluster[j]] = i;
			nNodesPerCluster[j]++;
		}
		clusteringStatsAvailable = true;
	}
	
	private void deleteClusteringStats() {
		clusterWeight = null;
		nNodesPerCluster = null;
		nodePerCluster = null;
		clusteringStatsAvailable = false;
	}
	
	public void printClusterStatus() {
		calcClusteringStats();
		
		System.out.print("cluster: ");
		for(int i = 0; i < cluster.length; i++) {
			System.out.print(cluster[i] + " ");
		}		
		System.out.println();
		
		System.out.print("clusterWeight: ");
		for(int i = 0; i < clusterWeight.length; i++) {
			System.out.print(clusterWeight[i] + " ");
		}		
		System.out.println();
		
		
		System.out.print("nNodesPerCluster: ");
		for(int i = 0; i < nNodesPerCluster.length; i++) {
			System.out.print(nNodesPerCluster[i] + " ");
		}
		System.out.println();
		
		
		System.out.println("nodePerCluster: ");
		for(int i = 0; i < nodePerCluster.length; i++) {
			for(int j = 0; j < nodePerCluster[i].length; j++) {
				System.out.print(nodePerCluster[i][j] + " ");
			}
			System.out.println();
		}		
		System.out.println();
	}
	
	
	public void printNetworkStatus() {
		
		System.out.println("/*********************NetworkStatus********************************/");
		System.out.println("nNodes: " + nNodes);

		
        System.out.print("edgeWeight: ");
	    for(int o = 0; o < edgeWeight.length; o++) {
	      	System.out.print(edgeWeight[o] + " ");
	      }
	    System.out.println();
	    	    
        System.out.print("nodeWeight: ");
	    for(int o = 0; o < nodeWeight.length; o++) {
	      	System.out.print(nodeWeight[o] + " ");
	      }
	    System.out.println();
	    
        System.out.print("firstNeighborIndex: ");
	    for(int o = 0; o < firstNeighborIndex.length; o++) {
	      	System.out.print(firstNeighborIndex[o] + " ");
	      }
	    System.out.println();
	    
        System.out.print("neighbor: ");
	    for(int o = 0; o < neighbor.length; o++) {
	      	System.out.print(neighbor[o] + " ");
	    }
	    System.out.println();
		System.out.println("/*****************************************************************/");

		
	}
}