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
import xzt.util.Knn;
import xzt.util.Parameter;
import xzt.util.Point;
/**
 * 0 is the positive instances i.e.the minority instances
 * @author Reacher
 *
 */
public class Borderline {
	//name diabetes glass
	public static String NAME = "satimage";
	//mapping diabetes  tested_positive  tested_negative 
	//mapping glass headlamps  others
	//mapping segment grass cement others
	//mapping satimage 4 others
	//mapping haberman 2 others
	//mapping vehicle       opel saab bus  van  
	public static String label = "4";
	public static String fileName = "D:\\百度云同步盘\\wordbench\\imbalanced data\\experiment\\test\\"+NAME+"\\";
//	public static String fileName = "D:\\百度云同步盘\\wordbench\\imbalanced data\\experiment\\test\\glass\\";
	public static double OVERRATING = 5; //小类样本扩大倍数
	public static final int FOLD = 8; //k-cv
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
	//"weka.classifiers.trees.J48"  "weka.classifiers.functions.MultilayerPerceptron"
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
	 * calculate the safe-level value of a point p
	 * @param p
	 * @return
	 */
	private static int safeLevel(List<Point> neighbor) {
		int count = 0;
		for (Point point : neighbor) {
			if(point.getLabel() != 0)
				count ++;
		}
		return count;
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
		
		/*//按照原有比例采样
		for(int i=0;i<FOLD;i++) { 
			int choose = 0;
			while( choose < ROW/FOLD && i != FOLD-1) {
				int p = pData.size() / FOLD;  //采样的小类样本的个数
				int rand = new Random().nextInt(dataSet.size());
				if( choose < p) {
					if( chosen.contains(rand) && dataSet.get(rand).getLabel() == 0) {
						chosen.remove(new Integer(rand));
						choose ++;
						allData.get(i).add(dataSet.get(rand));
					}
				} else {
					if( chosen.contains(rand) && dataSet.get(rand).getLabel() != 0) {
						chosen.remove(new Integer(rand));
						choose ++;
						allData.get(i).add(dataSet.get(rand));
					}
				}
			}
			//最后一折全部添加
			if( i == FOLD-1) {
				for (Integer o : chosen) {
					allData.get(i).add(dataSet.get(o));
				}
			}
				
		}*/
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
				if(point.getLabel() == 0)
					positiveTestData.add(point);
			}
			System.out.print("test data :"+testData.size() + "\t");
			System.out.println("test positive :"+positiveTestData.size());
			
			borderline(positiveTrainData,trainData);
			
			//generate new dataset
			String trainFileName = NAME + "BLTrain"+fold+".arff";
			String testFileName = NAME + "BLTest"+fold+".arff";
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

	private static void borderline(List<Point> positiveTrainData,
			List<Point> trainData) {
		List<Point> danger = dangerPoints(positiveTrainData,trainData);
		
		Random r = new Random();
		for (int i = 0; i < danger.size(); i++) {
			List<Point> neighbors = Knn.knn(positiveTrainData, positiveTrainData.indexOf(danger.get(i)));
			int S = (int) (positiveTrainData.size() * (OVERRATING-1) / danger.size());
			for (int j = 0; j < S; j++) { //generate S points
				Point p = synthesize(neighbors.get(r.nextInt(neighbors.size())),danger.get(i),Math.random());
				pointSet.add(p);
			}
		}
	}

	/**
	 * 计算Danger数据集
	 * @param positiveTrainData
	 * @param trainData
	 * @return
	 */
	private static List<Point> dangerPoints(List<Point> positiveTrainData,
			List<Point> trainData) {
		List<Point> danger = new ArrayList<Point>();
		for (int i = 0; i < positiveTrainData.size(); i++) {
			List<Point> neighbors = Knn.knn(trainData,trainData.indexOf(positiveTrainData.get(i)));
			int sl = safeLevel(neighbors);
			if( sl >= Math.ceil(KNN / 2.0) && sl < KNN)
				danger.add(positiveTrainData.get(i));
		}
		return danger;
	}

	/**
	 * 注意改变数据集的标签   二分类时，分类要注意去DateSet中去配置
	 * @param args
	 */
	public static void main(String[] args) {
		//k-fold cross-validation
		if( !NAME.equals("kdd") ) {
//			danger();
			crossValidation();
		} 
	}

	@SuppressWarnings("unused")
	private static void danger() {
		int danger = 0,noisy = 0,safe = 0;
		
		List<Point> positive = new ArrayList<Point>();
		for (Point point : dataSet) {
			if( point.getLabel() == 0)
				positive.add(point);
		}
		for (Point point : positive) {
			List<Point> neighbors = Knn.knn(dataSet, dataSet.indexOf(point));
			int sl = safeLevel(neighbors);
			if( sl >= 1 && sl < KNN) {
				danger ++;
			}
			if( sl == KNN)
				noisy ++;
			if( sl == 0)
				safe ++;
		}
		
		System.out.println("共"+positive.size()+"个点");
		System.out.println("danger的点个数为：  "+danger);
		System.out.println("noisy的点个数为：  "+noisy);
		System.out.println("safe的点个数为：  "+safe);
	}




	
}
