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
import xzt.util.Point;

public class DownSampling {
	public static String NAME = "satimage";
	//mapping  tested_positive  tested_negative 
	//mapping  headlamps  others
	//mapping grass others
	//mapping 4 others
	public static String label = "4";
	public static String fileName = "D:\\百度云同步盘\\wordbench\\imbalanced data\\experiment\\test\\"+NAME+"\\";
	
	public static DataSet ds;
	
	//读数据的时候自动确定 数据集的行数和列数
	public static int ROW;
	public static int COL; 

	public static List<Point> dataSet;
	public static List<Point> testSet;
	public static List<Point> pData;
	public static List<Point> nData;
	public static List<Point> pointSet;
	public static List<ArrayList<Point>> allData = new ArrayList<ArrayList<Point>>() ;
	
	public static final int FOLD = 3; //k-cv
	public static double DOWNSAMPLING = 1.5; 
	public static int K = 2;//聚类个数
	
	public static Classifier classifier;
	
	static {
		//pre-process dataset
		ds = new DataSet(label);
		ds.setup(fileName+NAME+".arff",null);
//		ds.setup(fileName + NAME + "TrainSampled_0.01_5.arff",fileName + NAME + "TestSampled0.003.arff");
		
		ROW = DataSet.getROW();
		COL = DataSet.getCOL();
		
		dataSet = DataSet.getDataSet();
		testSet = DataSet.getTestSet();
		pData = DataSet.getpData();
		nData = DataSet.getnData();
		
		pointSet = new ArrayList<Point>();
	}
	
	//sampling majority class samples according to the minority class samples
	public static void downsampling(List<Point> list,int numbers) {
		if(numbers == 0)
			return ;
		
		List<Integer> selected = new ArrayList<Integer>();
		while( selected.size() < numbers ) {
			Random r = new Random();
			int n = r.nextInt(list.size());
			if( !selected.contains(n) && list.get(n).getLabel() == 1) {
				selected.add(n);
				pointSet.add(list.get(n));
			}
		}
	}
	
	/**
	 * divide the dataset into k folds
	 */
	public static void crossValidation( int type) {
		
		double sum = 0;
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
			List<Point> negativeTrainData = new ArrayList<Point>();  //used to downsampling
			List<Point> positiveTestData = new ArrayList<Point>();
			
			testData.addAll(allData.get(fold));
			for (List<Point> ps : allData) {
				if( ps != allData.get(fold))
					trainData.addAll(ps);
			}
			
			//select the minority class instances to systhesize
			//select out majorith class instances to downsampling
			for (Point point : trainData) {
				if(point.getLabel() == 0)
					positiveTrainData.add(point);
				else
					negativeTrainData.add(point);
			}
			System.out.print("train data :"+trainData.size() + "\t");
			System.out.println("train positive :"+positiveTrainData.size());
			for (Point point : testData) {
				if(point.getLabel() == 0)
					positiveTestData.add(point);
			}
			System.out.print("test data :"+testData.size() + "\t");
			System.out.println("test positive :"+positiveTestData.size());
			
			//cluster the trainData
			if(type == 0) {  //先聚类 在抽取  在分类
				List<ArrayList<Point>> clusters = Kmeans.kmeans(trainData,K);

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
					select.add( (int)(DOWNSAMPLING * positiveTrainData.size() * rates.get(index) / sumRecord));
				}
				System.out.println("select :"+select);

				//将训练集进行downsampling操作
				for(int index=0; index < K; index++) {
					downsampling(clusters.get(index),select.get(index));
				}
			}
			else {  //random sampling
				downsampling(negativeTrainData,(int)DOWNSAMPLING * positiveTrainData.size());
			}
			
			//generate new dataset
			String trainFileName = NAME + "DownSamplingTrain"+fold+".arff";
			String testFileName = NAME+"DownSamplingTest"+fold+".arff";
			Generate.generate(pointSet,positiveTrainData,COL,fileName,trainFileName);
			Generate.generate(testData,new ArrayList<Point>(),COL,fileName,testFileName);
			System.out.println(pointSet.size());
			pointSet.clear();
			
			//用分类器测试
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
				
				classifier = (Classifier) Class.forName(
						"weka.classifiers.trees.J48").newInstance();
				classifier.buildClassifier(instancesTrain);
				
				Evaluation eval = new Evaluation(instancesTrain);
				//  第一个为一个训练过的分类器，第二个参数是在某个数据集上评价的数据集
				eval.evaluateModel(classifier, instancesTest);
				
				System.out.println(eval.toClassDetailsString());  
		        System.out.println(eval.toSummaryString());  
		        System.out.println(eval.toMatrixString());  
				System.out.println("precision is :"+(1-eval.errorRate()));
				
				sum += 1-eval.errorRate();
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			System.out.println("-----------------------------------------------");
		}
		System.out.println("average precision is :" + sum/FOLD);
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
		return majority*1.0 / (minority + 1);// in case that minority is 0
	}

	
	@SuppressWarnings("unused")
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
		if(type == 0) {  //先聚类，再生成小类样本，再分类
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
				select.add( (int)(DOWNSAMPLING * positiveTrainData.size() * rates.get(index) / sumRecord));
			}
			System.out.println("select :"+select);

			//TODO 将训练集进行downsampling操作
//			for(int index=0; index < K; index++) {
//				downsampling(clusters.get(index),select.get(index));
//			}
		}
		else {  
//			downsampling(negativeTrainData,(int)DOWNSAMPLING * positiveTrainData.size());
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
		
		//用分类器测试
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
			
			classifier = (Classifier) Class.forName(
					"weka.classifiers.functions.MultilayerPerceptron").newInstance();
			classifier.buildClassifier(instancesTrain);
			
			Evaluation eval = new Evaluation(instancesTrain);
			//  第一个为一个训练过的分类器，第二个参数是在某个数据集上评价的数据集
			eval.evaluateModel(classifier, instancesTest);
			
			System.out.println(eval.toClassDetailsString());  
	        System.out.println(eval.toSummaryString());  
	        System.out.println(eval.toMatrixString());  
			System.out.println("precision is :"+(1-eval.errorRate()));
			
			long endPredicting = System.currentTimeMillis();
			System.out.println("预测数据所用时间为："+ (endPredicting-start)/1000 + "s");
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	public static void main(String[] args) {
		int tpye = 0; //确定哪一种向下采样
		crossValidation(tpye);
	}
}
