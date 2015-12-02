package xzt.preprocess;

import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class TestA {
	public static String fileName = "D:\\IID\\MINE\\";
	public static String trainFileName = "kddcup_10_percent.arff";
	public static String testFileName = "test_corrected.arff";
	public static Classifier classifier;
	
	public static void launch() { 
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

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		launch();
	}
}
