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

public class Generate {
	public static List<String> stringlist ;
	public static String FILENAME;
	static {
		stringlist = new LinkedList<String>();
		FILENAME = "";
	}
	
	public static void readToken() {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(FILENAME+"\\token.txt"));
			while(reader.ready()) {
				stringlist.add(reader.readLine());
			}
			reader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * write new dataset to local file
	 */
	public static void generate(List<Point> A,List<Point> B,int COL,String fileName,String name) {
		BufferedWriter writer;
		FILENAME = fileName;
		try {
			writer = new BufferedWriter(new FileWriter(fileName+name));
			if(stringlist.size() == 0)
				readToken();
			for (String s : stringlist) {
				writer.write(s);
				writer.newLine();
			}
			
			List<Point> newDataset = new ArrayList<Point>();
			newDataset.addAll(A);
			newDataset.addAll(B);
			
			while( newDataset.size() > 0) {
				Random r = new Random();
				int selected = r.nextInt(newDataset.size());
				for(int i=0;i<COL;i++) {
					if(i == COL-1) {
						writer.write(newDataset.get(selected).getData().get(i)+"");
						writer.newLine();
					} else {
						writer.write(newDataset.get(selected).getData().get(i)+",");
					}
				}
				newDataset.remove(selected);
			}
			System.out.println("write file "+name+" finished!");
			writer.flush();
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
}
