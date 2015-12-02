package xzt.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class DataSet {
	
	//mapping  tested_positive  tested_negative 
	//mapping  headlamps  others
	//mapping grass others
	public String label = "";
	
	//读数据的时候自动确定 数据集的行数和列数
	public static int ROW;
	public static int COL; 

	public static List<Point> dataSet;
	public static List<Point> pData;
	public static List<Point> nData;
	public static List<Point> testSet;
	
	public DataSet(String label) {
		this.label = label;
		dataSet = new ArrayList<Point>();
		pData = new ArrayList<Point>();
		nData = new ArrayList<Point>();
		testSet = new ArrayList<Point>(); 
	}
	
	public static List<Point> getTestSet() {
		return testSet;
	}

	public static void setTestSet(List<Point> testSet) {
		DataSet.testSet = testSet;
	}

	public static int getROW() {
		return ROW;
	}

	public static void setROW(int rOW) {
		ROW = rOW;
	}

	public static int getCOL() {
		return COL;
	}

	public static void setCOL(int cOL) {
		COL = cOL;
	}

	public static List<Point> getDataSet() {
		return dataSet;
	}

	public static void setDataSet(List<Point> dataSet) {
		DataSet.dataSet = dataSet;
	}

	public static List<Point> getpData() {
		return pData;
	}

	public static void setpData(List<Point> pData) {
		DataSet.pData = pData;
	}

	public static List<Point> getnData() {
		return nData;
	}

	public static void setnData(List<Point> nData) {
		DataSet.nData = nData;
	}

	public void setup(String trainFile,String testFile) {
		//preprocess
		Long start = System.currentTimeMillis();
		mapping(trainFile,testFile);
		
		normalize(dataSet,testSet);
		long end = System.currentTimeMillis();
		System.out.println("数据处理所用时间为："+ (end-start)*1.0/1000 + "s");
	}
	

	/**
	 * read file and mapping the string to double into matrix dataSet
	 * store the positive class
	 * recognize 
	 * 			the dimensions of the features i.e. COL
	 * 			the size of the dataSet i.e. ROW
	 */
	public  void mapping(String trainFile,String testFile) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(trainFile));
			int trainCountLine = 0,trainCountP = 0;
			while( reader.ready() ) {
				String line = reader.readLine();
				String tmp[] = line.split(",");
				COL = tmp.length; //recognize the dimensions of the features
				
				//mapping  tested_positive  tested_negative 
				//mapping  headlamps  others
				//mapping grass others
				if( tmp[tmp.length-1].equals(label) || tmp[tmp.length-1].equals("0.0")) {
					tmp[tmp.length-1] = "0.0";
					Point tmpPoint = new Point(tmp);
					dataSet.add(tmpPoint);
					pData.add(tmpPoint);
					trainCountP ++;
				} else {
					tmp[tmp.length-1] = "1.0";
					Point tmpPoint = new Point(tmp);
					dataSet.add(new Point(tmpPoint));
					nData.add(tmpPoint);
				}
				trainCountLine ++;
 			}
			
			ROW = trainCountLine;
			System.out.println("trainCountP :"+trainCountP);
			System.out.println("trainCountLine :"+trainCountLine);
			reader.close();
			
			//if the test file exists
			if(testFile != "" && testFile != null) {  
				reader = new BufferedReader(new FileReader(testFile));
				int testCountLine = 0,testCountP = 0;
				while( reader.ready() ) {
					String line = reader.readLine();
					String tmp[] = line.split(",");

					if( tmp[tmp.length-1].equals(label)  || tmp[tmp.length-1].equals("0.0")) {
						tmp[tmp.length-1] = "0.0";
						Point tmpPoint = new Point(tmp);
						testSet.add(tmpPoint);
						testCountP ++;
					} else {
						tmp[tmp.length-1] = "1.0";
						Point tmpPoint = new Point(tmp);
						testSet.add(new Point(tmpPoint));
					}
					testCountLine ++;
//					if(testCountLine % 100000 == 0)
//						System.out.println(testCountLine);
				}

				System.out.println("testCountP :"+testCountP);
				System.out.println("testCountLine :"+testCountLine);
			}
			
			System.out.println("mapping finished!");
			reader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	/**
	 * normalize the data to [0,1]
	 * @param dataSet
	 */
	public  void normalize(List<Point> dataSet,List<Point> testSet) {
		//read every column of the matrix and compute the maximum and the minimum value 
		for(int col=0;col<COL-1;col++) {
//			if(col % 10 == 0)
//				System.out.println(col);
			
			List<Double> data = new ArrayList<Double>();	
			for(int row=0;row<dataSet.size();row++) {
				data.add(dataSet.get(row).get(col));
			}
			double []extreme = computeExtreme(data);
			for(int row=0;row<dataSet.size();row++) {
				if( extreme[0] == extreme[1])  // constant
					dataSet.get(row).set(col, 0.5) ; 
				else
					dataSet.get(row).set(col, (dataSet.get(row).get(col) - extreme[1]) / (extreme[0] - extreme[1])) ; 
			}
		}
		System.out.println("normalize train data finished!");
		
		//normalize the testSet if the testSet exists
		if(testSet.size() > 0) {
			for(int col=0;col<COL-1;col++) {
				List<Double> data = new LinkedList<Double>();	
				for(int row=0;row<testSet.size();row++) {
					data.add(testSet.get(row).get(col));
				}
				double []extreme = computeExtreme(data);
				for(int row=0;row<testSet.size();row++) {
					if( extreme[0] == extreme[1])  // constant
						testSet.get(row).set(col, 0.5) ; 
					else
						testSet.get(row).set(col, (testSet.get(row).get(col) - extreme[1]) / (extreme[0] - extreme[1])) ; 
				}
			}
		}
		System.out.println("normalize test data finished!");
	}
	
	/**
	 * find the maximum and minimum value 
	 * @param data   this is two-dimensional
	 * @return
	 */
	private  double[] computeExtreme(List<Double> data) {
		double []extreme = new double[2];
		double max = Double.MIN_VALUE;
		double min = Double.MAX_VALUE;
		
		for(int i=0;i<data.size();i++) {
			if( data.get(i) > max )
				max = data.get(i);
			if( data.get(i) < min)
				min = data.get(i);
		}
		extreme[0] = max;
		extreme[1] = min;
		return extreme;
	}

	public static void sampling(String objectFile,double rate,int fold) throws Exception {
			BufferedWriter writer = new BufferedWriter(new FileWriter(objectFile+fold+".arff"));
			
			int pSum = (int)(pData.size() * rate);
			int nSum = pSum * fold;
			System.out.println("pSum:"+pSum);
			System.out.println("nSum:"+nSum);
			
			Random r = new Random();
			ArrayList<Integer> pHasSelected = new ArrayList<Integer>();
			ArrayList<Integer> nHasSelected = new ArrayList<Integer>();
			while(pHasSelected.size() < pSum) {
				int p = r.nextInt( pData.size() );
				if( !pHasSelected.contains(p)) {
					pHasSelected.add(p);
//					System.out.println(pData.get(p).toString());
					writer.write(pData.get(p).toString());
					writer.newLine();
				}
			}
			while(nHasSelected.size() < nSum) {
				int n = r.nextInt( nData.size() );
				if( !nHasSelected.contains(n)) {
					nHasSelected.add(n);
					writer.write(nData.get(n).toString());
					writer.newLine();
				}
			}

			writer.flush();
			writer.close();
			System.out.println("sample finished!");
	}

	public static void main(String[] args) throws Exception {
		//TODO 设置类标签
		DataSet ds = new DataSet("");
		String trainFile = "D:\\IID\\MINE\\kddcup_10_percent";
		String testFile = "D:\\IID\\MINE\\test_corrected";
		String objectFile = "D:\\IID\\MINE\\kddcup_";
		ds.setup(trainFile+".arff",testFile+".arff");
		sampling(objectFile,0.1,5);
	}
	
}
