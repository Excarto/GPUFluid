package Vector;

import static java.lang.Math.*;

public final class Vector{
	
	public static double[] mult(double scalar, double[] vector){
		double[] result = new double[vector.length];
		for (int i = 0; i < result.length; i++)
			result[i] = scalar*vector[i];
		return result;
	}
	
	public static double[] sum(double[] v1, double[] v2){
		double[] sum = new double[v1.length];
		for (int i = 0; i < sum.length; i++)
			sum[i] = v1[i] + v2[i];
		return sum;
	}
	
	public static double[] diff(double[] v1, double[] v2){
		double[] diff = new double[v1.length];
		for (int i = 0; i < diff.length; i++)
			diff[i] = v1[i] - v2[i];
		return diff;
	}
	
	public static double[] componentAlong(double[] v, double[] unit){
		double[] projection = new double[v.length];
		double dot = dot(v, unit);
		for (int i = 0; i < projection.length; i++)
			projection[i] = unit[i]*dot;
		return projection;
	}
	
	public static double dot(double[] v1, double[] v2){
		double product = 0.0;
		for (int i = 0; i < v1.length; i++)
			product += v1[i]*v2[i];
		return product;
	}
	
	public static double[] normalized(double[] vect){
		double[] normalized = vect.clone();
		double magnitudeInv = 1/magnitude(vect);
		for (int i = 0; i < normalized.length; i++)
			normalized[i] *= magnitudeInv;
		return normalized;
	}
	
	public static double magnitude(double[] vect){
		double magnitude = 0;
		for (int i = 0; i < vect.length; i++)
			magnitude += vect[i]*vect[i];
		return sqrt(magnitude);
	}
	
	public static double dist(double[] pos1, double[] pos2){
		double dist = 0.0;
		for (int i = 0; i < pos1.length; i++){
			double d = pos1[i]-pos2[i];
			dist += d*d;
		}
		return sqrt(dist);
	}
	
	public static double[] cross(double[] v1, double[] v2){
		return new double[]{
				v1[1]*v2[2] - v1[2]*v2[1],
				v1[2]*v2[0] - v1[0]*v2[2],
				v1[0]*v2[1] - v1[1]*v2[0]
		};
	}
	
	public static double[] matMult(double[][] mat, double[] vect){
		double[] out = new double[3];
		for (int row = 0; row < 3; row++){
			for (int col = 0; col < 3; col++)
				out[row] += mat[row][col]*vect[col];
		}
		return out;
	}
	
	public static double[][] getZYZMatrix(double a, double b, double c){
		a = toRadians(a);
		b = toRadians(b);
		c = toRadians(c);
		return new double[][]{
			{cos(a)*cos(b)*cos(c)-sin(a)*sin(c), -cos(c)*sin(a)-cos(a)*cos(b)*sin(c), cos(a)*sin(b)},
			{cos(a)*sin(c)+cos(b)*cos(c)*sin(a),  cos(a)*cos(c)-cos(b)*sin(a)*sin(c), sin(a)*sin(b)},
			{-cos(c)*sin(b),                      sin(b)*sin(c),                      cos(b)}
		};
	}
	
	public static double[][] getXYZMatrix(double a, double b, double c){
		a = toRadians(a);
		b = toRadians(b);
		c = toRadians(c);
		return new double[][]{
			{cos(b)*cos(c),                      -cos(b)*sin(c),                     sin(b)},
			{cos(a)*sin(c)+cos(c)*sin(a)*sin(b), cos(a)*cos(c)-sin(a)*sin(b)*sin(c), -cos(b)*sin(a)},
			{sin(a)*sin(c)-cos(a)*cos(c)*sin(b), cos(c)*sin(a)+cos(a)*sin(b)*sin(c), cos(a)*cos(b)}
		};
	}
	
}
