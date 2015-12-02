package xzt.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class Knn {


	/**
	 * select point p's k nearest neighborhood
	 * @param p which means the p-th element
	 * @return  bucket
	 */
	/*public static List<Point> knn(List<Point> pData,int p) {
		List<Point> bucket = new ArrayList<Point>();

		Map<Double,Point> map = new HashMap<Double,Point>();
		for(int i=0;i<pData.size();i++ ) {
			if( i!=p ) { //self-exclusive
				if( map.size() < K ) 
					map.put(pData.get(p).dist(pData.get(i)), pData.get(i));
				else {
					for(Map.Entry<Double,Point> entry : map.entrySet()) {
						if( entry.getKey() > pData.get(p).dist(pData.get(i))) {
							map.remove(entry.getKey());
							map.put(pData.get(p).dist(pData.get(i)), pData.get(i));
							break;  //once match,break
						}
					}
				}
			}
		}
		for(Point point : map.values()) {
			bucket.add(point);
		}
		//System.out.println("bucket size is: "+ bucket.size());
		return bucket;
	} */
	public static List<Point> knn(List<Point> pData,int p) {
		List<Point> bucket = new ArrayList<Point>();

		Map<Double,Point> tree = new TreeMap<Double,Point>();
		for(int i=0;i<pData.size();i++ ) {
			if( i!=p ) { 
				tree.put(pData.get(p).dist(pData.get(i)), pData.get(i));
			}
		}
		int countPoint = 0;
		for(Map.Entry<Double, Point> entry : tree.entrySet()) {
			if(countPoint < Parameter.K) {
				bucket.add(entry.getValue());
				countPoint ++;
			} else {
				break;
			}
		}
		return bucket;
	}
}
