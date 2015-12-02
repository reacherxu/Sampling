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
import xzt.util.Point;
/**
 * 0 is the positive instances i.e.the minority instances
 * @author Reacher
 *
 */
public class SMOTE {
	//name diabetes glass
	public static String NAME = "satimage";
	//mapping diabetes tested_positive  tested_negative 
	//mapping glass headlamps  others
	//mapping segment grass others
	//mapping satimage 4 others
	//mapping haberman 2 others
	//iris Iris-setosa,Iris-versicolor,Iris-virginica
	//mapping vehicle       opel saab bus  van  
	//mapping yeast 0 1 
	public static String label = "4";
	public static String fileName = "D:\\百度云同步盘\\wordbench\\imbalanced data\\experiment\\test\\"+NAME+"\\";
//	public static String fileName = "D:\\百度云同步盘\\wordbench\\imbalanced data\\experiment\\test\\glass\\";
	public static double OVERRATING = 6; //小类样本扩大倍数
	public static final int FOLD =  8; //k-cv

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
	public static String CLASSIFIERNAME = "weka.classifiers.functions.MultilayerPerceptron";
	
	static {
		//pre-process dataset
		ds = new DataSet(label);
		ds.setup(fileName+NAME+".arff",null);
//		ds.setup(fileName + NAME + "-train.arff",fileName + NAME + "-test.arff");
		
		ROW = DataSet.getROW();
		COL = DataSet.getCOL();
		
		dataSet = DataSet.getDataSet();
		pData = DataSet.getpData();
		testSet = DataSet.getTestSet();
		
		pointSet = new ArrayList<Point>();
	}
	
	/**
	 * oversampling the minority samples
	 * @param minority
	 * 			the minority class point samples
	 */
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
		
//		double sum = 0;
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
			
			//不进行任何处理
//			System.out.println("----------------Original result-----------------------");
			String trainFileName = NAME + "OriginalTrain"+fold+".arff";
			String testFileName = NAME + "OriginalTest"+fold+".arff";
//			Generate.generate(trainData,new ArrayList<Point>(),COL,fileName,trainFileName);
//			Generate.generate(testData,new ArrayList<Point>(),COL,fileName,testFileName);
//			
//			classify(trainFileName,testFileName);
			
			
			//generate new dataset
			//将训练集进行smote操作
			smote(positiveTrainData);
			
			System.out.println("----------------SMOTE result-----------------------");
			trainFileName = NAME + "SMOTETrain"+fold+".arff";
			testFileName = NAME + "SMOTETest"+fold+".arff";
			Generate.generate(trainData,pointSet,COL,fileName,trainFileName);
			Generate.generate(testData,new ArrayList<Point>(),COL,fileName,testFileName);
			pointSet.clear();
			
			classify(trainFileName,testFileName);
			
			
			System.out.println("-----------------------------------------------");
		}
//		System.out.println("average precision is :" + sum/FOLD);
	}

	//regular DM method 
	private static void regular() {
		List<Point> positiveTrainData = new ArrayList<Point>();
		
		//select the minority class instances
		for (Point point : dataSet) {
			if(point.getLabel() == 0)
				positiveTrainData.add(point);
		}
		System.out.print("train data :"+dataSet.size() + "\t");
		System.out.println("train positive :"+positiveTrainData.size());
		
		
		//将训练集进行smote操作
		smote(positiveTrainData);
		
		//generate new dataset
		String trainFileName = NAME + "SMOTETrain"+".arff";
		String testFileName = NAME + "SMOTETest"+".arff";
		Generate.generate(dataSet,pointSet,COL,fileName,trainFileName);
		Generate.generate(testSet,new ArrayList<Point>(),COL,fileName,testFileName);
		pointSet.clear();
		classify(trainFileName,testFileName);
		
		//不进行任何处理
		trainFileName = NAME + "TrainWS"+".arff";
		testFileName = NAME + "TestWS"+".arff";
		Generate.generate(dataSet,new ArrayList<Point>(),COL,fileName,trainFileName);
		Generate.generate(testSet,new ArrayList<Point>(),COL,fileName,testFileName);
		classify(trainFileName,testFileName);
	}
	
	/**
	 * 注意改变数据集的标签   二分类时，分类要注意去DateSet中去配置
	 * @param args
	 */
	public static void main(String[] args) {
		int isCV = 0;
		//k-fold cross-validation
		if( isCV == 0  )
			crossValidation();
		else //use trainSet and testSet
			regular();
	}




	
}
