package xzt.preprocess;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class DataPreprocess {
	public static void preprocess(String fileName) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			BufferedWriter writer = new BufferedWriter(new FileWriter(fileName+"csv"));
			int countLine = 0,countP = 0,COL;
//			Map<String,Integer> classification = new HashMap<String,Integer>();
			
			while( reader.ready() ) {
				String line = reader.readLine();
				String tmp[] = line.split("\t");
				COL = tmp.length; //recognize the dimensions of the features
				
				if( tmp[COL-1].equals("ME2") ) {
					for(int i=1;i<COL;i++) {
						if( i == COL-1 ) {
							writer.write("0");
							writer.newLine();
						} else {
							writer.write(tmp[i].trim()+",");
						}
					}
					countP ++;
				} else  {
					for(int i=1;i<COL;i++) {
						if(countLine == 407)
							System.out.println(tmp);
						if( i == COL-1 ) {
							writer.write("1");
							writer.newLine();
						} else {
							writer.write(tmp[i].trim()+",");
						}
					}
				}
				countLine ++;
 			}
			
//			ROW = countLine;
			System.out.println("countP :"+countP);
			System.out.println("countLine :"+countLine+"\n\n");
			
			reader.close();
			writer.flush();
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public static void main(String[] args) {
		preprocess("D:\\百度云同步盘\\wordbench\\imbalanced data\\experiment\\yeast.data");
	}
	
}
