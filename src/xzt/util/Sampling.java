package xzt.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Random;

public class Sampling {

	public static void sample(String file,double rate) throws Exception {
		BufferedReader reader = new BufferedReader(new FileReader(file+".arff"));
		BufferedWriter writer = new BufferedWriter(new FileWriter(file+rate+"sampled.arff"));
		
		ArrayList<String> samples = new ArrayList<String>();
		while( reader.ready()) {
			samples.add(reader.readLine());
		}
		
		Random r = new Random();
		ArrayList<Integer> hasSelected = new ArrayList<Integer>();
		while(hasSelected.size() < samples.size() * rate) {
			int s = r.nextInt( samples.size() );
			if( !hasSelected.contains(s)) {
				hasSelected.add(s);
//				System.out.println(pData.get(p).toString());
				writer.write(samples.get(s));
				writer.newLine();
			}
		}

		reader.close();
		writer.flush();
		writer.close();
		System.out.println("sample finished!");
	}
	
	public static void main(String[] args) throws Exception {
		sample("D:\\IID\\MINE\\test_corrected",0.003);
	}
}
