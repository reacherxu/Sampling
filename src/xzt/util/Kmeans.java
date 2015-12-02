package xzt.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Kmeans {
	
	
	/**
	 * cluster the dataset into several clusters
	 * @param trainData 
	 * @return
	 */
	public static List<ArrayList<Point>> kmeans(List<Point> trainData,int K) {
		List<ArrayList<Point>> clusters = null;
		boolean clusterFlag = true;  //need to cluster
		
		while( clusterFlag ) {
			clusters = new ArrayList<ArrayList<Point>>();
			//initial the clusters with random points
			//the first point in the list is the mean
			List<Integer> hasUsed = new ArrayList<Integer>();
			Random r = new Random();
			int count = 0;
			while( count < K) {
				ArrayList<Point> tmpCluster = new ArrayList<Point>();

				int random = r.nextInt(trainData.size());
				if( !hasUsed.contains(random)) {
					hasUsed.add(random);				

					tmpCluster.add(trainData.get(random)); 
					clusters.add(tmpCluster);
					count ++;
				}
			}

			int iteration = 0;
			List<Point> oldClusters = new ArrayList<Point>();
			List<Point> newClusters = new ArrayList<Point>();
			do {
				oldClusters = newClusters;
				updateMeans(clusters,K);

				allocate(trainData,clusters);

				newClusters = getMeans(clusters);
				if( iteration % 50 == 0)
					System.out.println(" iteration :" + iteration);
				iteration++;
				//			System.out.println("oldClusters："+oldClusters);
				//			System.out.println("newClusters："+newClusters);

			} while( isEqual(oldClusters,newClusters) == false );

			//remove the means
			for(int i=0;i<clusters.size();i++)
				clusters.get(i).remove(0);
			
			// make the clusters more average
			int clusterSize ;
			for(clusterSize=0;clusterSize<clusters.size();clusterSize++) {
				if( clusters.get(clusterSize).size() < trainData.size() / 10 ) {
					clusterFlag = true;
					K = K-1;
					break;  //recluster
				}
			}
			if( clusterSize == clusters.size())
				clusterFlag = false;
		}
		return clusters;
	}

	/**
	 * judge if the two clusters are the same
	 * @param oldClusters
	 * @param newClusters
	 * @return
	 */
	private static boolean isEqual(List<Point> oldClusters,
			List<Point> newClusters) {
		if(oldClusters.size() == 0 || oldClusters == null)
			return false;
		for(int i=0;i<oldClusters.size();i++) {
			if( oldClusters.get(i).equal(newClusters.get(i)) == false)
				return false;
		}
		return true;
	}

	/**
	 * get the means 
	 * @param cluster
	 * @return
	 */
	private static List<Point> getMeans(List<ArrayList<Point>> cluster) {
		List<Point> means = new ArrayList<Point>();
		for(int i=0;i<cluster.size();i++)
			means.add(new Point(cluster.get(i).get(0)));
		return means;
	}


	/**
	 * update the means
	 * @param clusters 
	 * @param K
	 */
	private static void updateMeans(List<ArrayList<Point>> clusters, int K) {
		int len = clusters.get(0).get(0).getData().size();
		if(clusters.get(0).size() > 1) {
			for(int i=0;i<K;i++) {
				Point tmpMean = new Point(
						new double[len]);
				//calculate the mean and save
				for(int j=1;j<clusters.get(i).size();j++) {
					tmpMean.add(clusters.get(i).get(j));
				}
				for(int j=0;j<len;j++) {
					tmpMean.set(j, tmpMean.get(j)/(clusters.get(i).size() - 1));
				}
				clusters.get(i).clear();
				clusters.get(i).add(tmpMean);
			}
		}
	}

	/**
	 * 计算簇内变差
	 * @param arrayList
	 * @return
	 */
	@SuppressWarnings("unused")
	private static double getVar(List<ArrayList<Point>> c) {
		double argFun = 0;
		for(int index=0;index<c.size();index++) {
			double sum = 0;
			for(int i=1;i<c.get(index).size();i++) {
				sum += Math.pow(c.get(index).get(0).dist(c.get(index).get(i)), 2);
			}
			argFun += sum;
		}
	
		return argFun;
	}

	/**
	 * 分配到不同的簇，并更新中心
	 * @param trainData
	 * @param clusters
	 */
	private static void allocate(List<Point> trainData,
			List<ArrayList<Point>> clusters) {
		for(int i=0;i<trainData.size();i++ ) { //the i-th point
			double minDist = Double.MAX_VALUE;
			int index = 0;
			
			for(int j=0;j<clusters.size();j++) { // the j-th cluster
				double dist = trainData.get(i).dist(clusters.get(j).get(0));
				if (  dist < minDist ) {
					minDist = dist;
					index = j;
				}
			}
			clusters.get(index).add(trainData.get(i));
		}
		
	}

}
