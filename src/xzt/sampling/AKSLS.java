package xzt.sampling;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import xzt.util.DataSet;
import xzt.util.Generate;
import xzt.util.Kmeans;
import xzt.util.Knn;
import xzt.util.Parameter;
import xzt.util.Point;
/**
 * 0 is the positive instances i.e.the minority instances
 * @author Reacher
 *
 */
public class AKSLS {
	//name diabetes glass
	public static String NAME = "satimage";
	//mapping diabetes  tested_positive  tested_negative 
	//mapping glass headlamps  others
	//mapping segment grass others
	//mapping satimage 4 others
	//mapping haberman 2 others
	//mapping vehicle       opel saab bus  van  
	//mapping yeast 0 1
	public static String label = "4";
	public static String fileName = "D:\\百度云同步盘\\wordbench\\imbalanced data\\experiment\\test\\"+NAME+"\\";
//	public static String fileName = "D:\\百度云同步盘\\wordbench\\imbalanced data\\experiment\\test\\glass\\";
	public static double OVERRATING = 5 ; //小类样本扩大倍数
	public static final int FOLD = 10; //k-cv
	public static final int K = 3; //初始聚类参数K
	public static final int KNN = Parameter.K; //K近邻参数K
	
	//读数据的时候自动确定 数据集的行数和列数
	public static int ROW;
	public static int COL; 
	
	//仅仅是数据集部分
	public static DataSet ds;
	public static List<Point> dataSet;
	public static List<Point> pData;
	public static List<Point> testSet;
	public static List<Point> pointSet;
	
	public static List<ArrayList<Point>> allData = new ArrayList<ArrayList<Point>>() ;
	
	public static Classifier classifier;
	//"weka.classifiers.trees.J48"  "weka.classifiers.functions.MultilayerPerceptron"  LibSVM
	public static String CLASSIFIERNAME = "weka.classifiers.trees.J48";
	
	static {
		//pre-process dataset
		ds = new DataSet(label);
		ds.setup(fileName+NAME+".arff",null);
//		ds.setup(fileName + NAME + "TrainSampled_0.01_5.arff",fileName + NAME + "TestSampled0.003.arff");
		
		ROW = DataSet.getROW();
		COL = DataSet.getCOL();
		
		dataSet = DataSet.getDataSet();
		pData = DataSet.getpData();
		testSet = DataSet.getTestSet();
		
		pointSet = new ArrayList<Point>();
	}
	
	
	
	/**
	 * synthesize an point between p and n using their distance 
	 * @param p
	 * @param n
	 * @param gap
	 */
	private static Point synthesize(Point p, Point n, double gap) {
		double []a = new double[COL];

		for(int i=0;i<p.getData().size();i++) {
			a[i] = p.getData().get(i) + gap * (n.getData().get(i) - p.getData().get(i));
		}

		return new Point(a);
	}
	
	/**
	 * calculate the safe-level value of a point p in set
	 * @param p
	 * @return
	 */
	private static int safeLevel(Point point,List<Point> set) {
		List<Point> neighbors = Knn.knn(set, set.indexOf(point));
		
		int count = 0;
		for (Point p : neighbors) {
			if(p.getLabel() == point.getLabel())
				count ++;
		}
		return count;
	}
	
	/**
	 * use method safe-level to smote points
	 * @param minority  小类样本数据
	 * @param all  全局数据
	 * @param numbers  要生成的小类样本的个数
	 */
	private static void safeLevelSmote(List<Point> minority,List<Point> all,int numbers) {
		if(numbers == 0)
			return ;
		
		Random r = new Random();
		List<Point> points = new ArrayList<Point>();
		while( points.size() < numbers ) {
//			if( pointSet.size() % 400 == 0 && pointSet.size() != 0)
//				System.out.println("this is "+pointSet.size()+"-th point......");
			
			int p = r.nextInt( minority.size() );

			List<Point> neighbors = Knn.knn(minority, p);//同类k近邻
			//TODO  if the neighbor is null
			Point n = neighbors.get(r.nextInt(neighbors.size()));

			int sl_p = safeLevel(minority.get(p),all);
			int sl_n = safeLevel(n,all);
			
			if(sl_p == 0 || (sl_p == 0 && sl_n == 0)) {//both are noisy 
				continue;
			} else if( sl_p != 0 && sl_n == 0) { // n is noisy,duplicate p
				points.add(new Point(minority.get(p)));
			} else {
				double ratio = 1.0 * sl_p / sl_n;
				double gap = 0;
				if(ratio == 1) {
					if(sl_p < Parameter.K / 2)
						continue;
					gap = r.nextDouble();
				} else if( ratio < 1 ) {
					gap = Math.random() * ratio + (1 - ratio);
				} else {
					gap = Math.random() * 1/ratio ;
				}
				
				Point newPoint = synthesize(minority.get(p),n,gap);
				
				all.add(newPoint);
				if( safeLevel(newPoint,all) == 0)  {//noisy 
					all.remove(all.size()-1);
					continue;
				}
				all.remove(all.size()-1);
				points.add(newPoint);
			}
		}
		pointSet.addAll(points);
	}
	

	/**
	 * record the rate of majority class samples to minority class samples
	 * @param arrayList
	 * @return
	 */
	private static double record(ArrayList<Point> cluster) {
		int majority = 0,minority = 0; 
		for(int i=1;i<cluster.size();i++) {
			if(cluster.get(i).getLabel() == 0)
				minority ++;
			else 
				majority ++;
		}
		System.out.println("majority:"+majority+"\t"+" minority:"+minority);
		return minority*1.0 / (majority + 1);// in case that majority is 0
	}
	
	//regular DM method 
	private static void regular(int type) {
		long start = System.currentTimeMillis();
		List<Point> positiveTrainData = new ArrayList<Point>();
		
		//select the minority class instances
		for (Point point : dataSet) {
			if(point.getLabel() == 0)
				positiveTrainData.add(point);
		}
		System.out.print("train data :"+dataSet.size() + "\t");
		System.out.println("train positive :"+positiveTrainData.size());

		//cluster the trainData,here, dataSet is the trainData
		//先聚类，再生成小类样本，再分类
		List<ArrayList<Point>> clusters = Kmeans.kmeans(dataSet,K);

		double sumRecord = 0;
		List<Double> rates = new ArrayList<Double>();
		List<Integer> select = new ArrayList<Integer>();
		for(int index=0; index < K; index++) {
			double rate = record(clusters.get(index));
			sumRecord += rate;
			rates.add(rate);
		}
		System.out.println("rates :"+rates);
		for(int index=0; index < K; index++) {
			select.add( (int)((OVERRATING-1) * positiveTrainData.size() * rates.get(index) / sumRecord));
		}
		System.out.println("select :"+select);

		for(int index=0; index < K; index++) {
			//select the minority class instances in the index-th cluster
			List<Point> positiveClusterData = new ArrayList<Point>();
			for (Point point : clusters.get(index)) {
				if(point.getLabel() == 0)
					positiveClusterData.add(point);
			}
			safeLevelSmote(positiveClusterData,clusters.get(index),select.get(index));
		}
		
		//generate new dataset
		String trainFileName = NAME + "SLSTrain"+".arff";
		String testFileName = NAME + "SLSTest"+".arff";
		Generate.generate(dataSet,pointSet,COL,fileName,trainFileName);
		Generate.generate(testSet,new ArrayList<Point>(),COL,fileName,testFileName);
//		pointSet.clear();
		long endGenerating = System.currentTimeMillis();
		System.out.println("产生数据所用时间为："+ (endGenerating-start)/1000 + "s");
		
		
	/*	//不进行任何处理
		trainFileName = NAME + "TrainWS"+".arff";
		testFileName = NAME + "TestWS"+".arff";
		Generate.generate(dataSet,new ArrayList<Point>(),COL,fileName,trainFileName);
		Generate.generate(testSet,new ArrayList<Point>(),COL,fileName,testFileName);
//		pointSet.clear();
*/		
		
		
		classify(trainFileName,testFileName);
		
		long endPredicting = System.currentTimeMillis();
		System.out.println("预测数据所用时间为："+ (endPredicting-start)/1000 + "s");
	}
	
	/**
	 * 用分类器测试
	 * @param trainFileName
	 * @param testFileName
	 */
	public static void classify(String trainFileName,String testFileName) {
		try {
			File inputFile = new File(fileName+trainFileName);// 训练语料文件
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances instancesTrain = atf.getDataSet(); // 读入训练文件

			//设置类标签类
			inputFile = new File(fileName+testFileName);// 测试语料文件
			atf.setFile(inputFile);
			Instances instancesTest = atf.getDataSet(); // 读入测试文件

			instancesTest.setClassIndex(instancesTest.numAttributes()-1);
			instancesTrain.setClassIndex(instancesTrain.numAttributes()-1);

//			classifier = (Classifier) Class.forName(
//					"weka.classifiers.trees.J48").newInstance();
			classifier = (Classifier) Class.forName(CLASSIFIERNAME).newInstance();
//			classifier = (Classifier) new WLSVM();
			classifier.buildClassifier(instancesTrain);

			Evaluation eval = new Evaluation(instancesTrain);
			//  第一个为一个训练过的分类器，第二个参数是在某个数据集上评价的数据集
			eval.evaluateModel(classifier, instancesTest);

			System.out.println(eval.toClassDetailsString());  
			System.out.println(eval.toSummaryString());  
			System.out.println(eval.toMatrixString());  
			System.out.println("precision is :"+(1-eval.errorRate()));

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	

	/**
	 * oversampling the minority samples
	 * @param minority
	 * 			the minority class point samples
	 */
	@SuppressWarnings("unused")
	private static void smote(List<Point> minority) {
		//step 1
		for(int i=0;i<(OVERRATING-1)*minority.size();i++) {
			Random r = new Random();
			int p = r.nextInt(minority.size());
			List<Point> neighbors = Knn.knn(minority, p);
			//step 2,3
			Point s = synthesize(neighbors,p);
			pointSet.add(s);
		}
		System.out.println("pointSet size is: "+ pointSet.size());
	}
	/**
	 * synthesize an element between p and one of its neighborhoods
	 * @param neighbors
	 */
	private static Point synthesize(List<Point> neighbors,int p) {
		Random r = new Random();
		Point point = new Point(pData.get(p));
		Point selected = neighbors.get(r.nextInt(neighbors.size()));
		double []a = new double[COL];
		
		for(int i=0;i<point.getData().size();i++) {
			a[i] = point.getData().get(i) + r.nextDouble() * (selected.getData().get(i) - point.getData().get(i));
		}
		
		return new Point(a);
	}
	/**
	 * divide the dataset into k folds
	 */
	public static void crossValidation() {
		
		//---------------------分为k折-----------------------------
		//初始化为k fold
		for(int i=0;i<FOLD;i++) {
			ArrayList<Point> tmp = new ArrayList<Point>();
			allData.add(tmp);
		}
		//选一个  删一个
		List<Integer> chosen = new ArrayList<Integer>();
		for(int i=0;i<dataSet.size();i++) {
			chosen.add(i);
		}
		
		
		for(int i=0;i<FOLD;i++) { 
			int choose = 0;
			while( choose < ROW/FOLD && i != FOLD-1) {
				int rand = new Random().nextInt(dataSet.size());
				if( chosen.contains(rand)) {
					chosen.remove(new Integer(rand));
					choose ++;
					allData.get(i).add(dataSet.get(rand));
				}
			}
			//最后一折全部添加
			if( i == FOLD-1) {
				for (Integer o : chosen) {
					allData.get(i).add(dataSet.get(o));
				}
			}
				
		}
		
		//------------------取一折为测试，其余为训练集-----------------------------
		for(int fold=0;fold<FOLD;fold++) {
			long start = System.currentTimeMillis();
			
			List<Point> trainData = new ArrayList<Point>();
			List<Point> testData = new ArrayList<Point>();
			List<Point> positiveTrainData = new ArrayList<Point>();
			List<Point> positiveTestData = new ArrayList<Point>();
			
			testData.addAll(allData.get(fold));
			for (List<Point> ps : allData) {
				if( ps != allData.get(fold))
					trainData.addAll(ps);
			}
			//select the minority class instances
			for (Point point : trainData) {
				if(point.getLabel() == 0)
					positiveTrainData.add(point);
			}
			System.out.print("train data :"+trainData.size() + "\t");
			System.out.println("train positive :"+positiveTrainData.size());
			for (Point point : testData) {
				if(point.getLabel() == 0 )
					positiveTestData.add(point);
			}
			System.out.print("test data :"+testData.size() + "\t");
			System.out.println("test positive :"+positiveTestData.size());
			
			kickNoisy(positiveTrainData,trainData);
//			smote(positiveTrainData);
//			safeLevelSmote(positiveTrainData,trainData,(OVERRATING-1) * positiveTrainData.size());
			//cluster the trainData
			//先聚类，再生成小类样本，再分类
			List<ArrayList<Point>> clusters = Kmeans.kmeans(trainData,K);

			int clusterSize = clusters.size();
			double sumRecord = 0;
			List<Double> rates = new ArrayList<Double>();
			List<Integer> select = new ArrayList<Integer>();
			for(int index=0; index < clusterSize; index++) {
				double rate = record(clusters.get(index));
				sumRecord += rate;
				rates.add(rate);
			}
			System.out.println("rates :"+rates);
			for(int index=0; index < clusterSize; index++) {
				select.add( (int)((OVERRATING-1) * positiveTrainData.size() * rates.get(index) / sumRecord));
			}
			System.out.println("select :"+select);

			for(int index=0; index < clusterSize; index++) {
				//select the minority class instances in the index-th cluster
				List<Point> positiveClusterData = new ArrayList<Point>();
				for (Point point : clusters.get(index)) {
					if( point.getLabel() == 0)
						positiveClusterData.add(point);
				}
				safeLevelSmote(positiveClusterData,clusters.get(index),select.get(index));
			}
			
			
			//generate new dataset
			String trainFileName = NAME + "SLSTrain"+fold+".arff";
			String testFileName = NAME + "SLSTest"+fold+".arff";
			//TODO  dataSet is a bug
			Generate.generate(trainData,pointSet,COL,fileName,trainFileName);
			Generate.generate(testData,new ArrayList<Point>(),COL,fileName,testFileName);
			pointSet.clear();
			long endGenerating = System.currentTimeMillis();
			System.out.println("产生数据所用时间为："+ (endGenerating-start)*1.0/1000 + "s");
			
			
		/*	//不进行任何处理
			trainFileName = NAME + "TrainWS"+".arff";
			testFileName = NAME + "TestWS"+".arff";
			Generate.generate(dataSet,new ArrayList<Point>(),COL,fileName,trainFileName);
			Generate.generate(testSet,new ArrayList<Point>(),COL,fileName,testFileName);
//			pointSet.clear();
	*/		
			
			
			classify(trainFileName,testFileName);
			long endClassifying = System.currentTimeMillis();
			System.out.println("产生数据所用时间为："+ (endClassifying-start)*1.0/1000 + "s");
		}
	}

	/**
	 * 注意改变数据集的标签   二分类时，分类要注意去DateSet中去配置
	 * @param args
	 */
	public static void main(String[] args) {
		int type = 0;
		
//		kickNoisy();
		//k-fold cross-validation
		if( !NAME.equals("kdd") ) {
//			danger();
			crossValidation();
		} else //use trainSet and testSet
			regular(type);
	}
	
	//kick the noisy points in positiveTrainData and trainData
	private static void kickNoisy(List<Point> positiveTrainData,List<Point> trainData) {
		List<Point> noise = new ArrayList<Point>();
		for (Point p : positiveTrainData) {
			if( 0 == safeLevel(p, trainData) ) {
				noise.add(p);
			}
		}
		for (Point point : noise) {
			positiveTrainData.remove(point);
			trainData.remove(point);
		}
		noise.clear();
	}

	@SuppressWarnings("unused")
	private static void danger() {
		int danger = 0,noisy = 0,safe = 0;
		
		for (Point point : pData) {
			
			int sl = safeLevel(point,dataSet);
			if( sl >= 1 && sl < KNN) {
				danger ++;
			}
			if( sl == KNN)
				safe ++;
			if( sl == 0)
				noisy ++;
		}
		
		System.out.println("共"+pData.size()+"个点");
		System.out.println("danger的点个数为：  "+danger);
		System.out.println("noisy的点个数为：  "+noisy);
		System.out.println("safe的点个数为：  "+safe);
	}
	
}
