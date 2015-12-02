package xzt.util;

import java.util.LinkedList;
import java.util.List;

public class Point {
	private List<Double> data;
	
	public Point() {
		this.data = new LinkedList<Double>();  //garantee the order of the array
	}
	
	public Point(LinkedList<Double> data) {
		super();
		this.data = data;
	}
	
	//the real point 
	public Point(double []a) {
		this.data = new LinkedList<Double>();
		for(int i=0;i<a.length;i++ ) {
			data.add(a[i]);
		}
	}
	public Point(Point p) {
		this.data = p.data;
	}
	//the original data
	public Point(String []a) {
		this.data = new LinkedList<Double>();
		for(int i=0;i<a.length;i++ ) {
			data.add(Double.parseDouble(a[i]));
		}
	}

	public String toString() {
		String str = "";
		for(int i=0;i<this.data.size();i++) {
			if(i == this.data.size()-1)
				str += this.data.get(i);
			else 
				str += this.data.get(i)+",";
		}
		return str;
		
	}
	
	//return the i-th value
	public double get(int i) {
		return this.data.get(i);
	}
	public void set(int index,double value) {
		this.data.set(index, value);
	}
	
	//Euclidean distance
	public double dist(Point p) {
		double sum = 0;
		for(int i=0;i<p.data.size();i++) {
			sum += Math.pow(this.data.get(i)-p.get(i), 2);
		}
		return Math.sqrt(sum);
	}
	
	//point plus point
	public void add(Point point) {
		for(int i=0;i<this.data.size();i++) {
			this.data.set(i, this.data.get(i)+point.get(i));
		}
	}
	
	public List<Double> getData() {
		return data;
	}

	public void setData(List<Double> data) {
		this.data = data;
	}
	
	public double getLabel() {
		return this.data.get(data.size()-1);
	}

	/**
	 * judge if the two points are same
	 * @param point
	 * @return
	 */
	public boolean equal(Point point) {
		for(int i=0;i<this.data.size();i++) {
			if(this.data.get(i) != point.get(i))
				return false;
		}
		return true;
	}

	public static void main(String[] args) {
		double x[] = {1.1,2.2,2.3,3.2};
		double y[] = {1.0,2.2,2.3,3.20};
		Point a = new Point(x);
		Point b = new Point(y);
		System.out.println(a.equal(b));
	}
	
	
}
