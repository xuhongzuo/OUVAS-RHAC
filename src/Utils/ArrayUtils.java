package Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.function.DoublePredicate;

import javax.swing.text.StyledEditorKit.ForegroundAction;

import org.omg.CORBA.DynamicImplementation;
import org.w3c.dom.ls.LSException;

import Jama.Matrix;

public class ArrayUtils {
	
	
	public static double getMax(double[] array){
		double max = 0.0;
		
		for(int i = 0; i < array.length; i++){
			if(array[i] > max){
				max = array[i];
			}
		}
				
		return max;
	}
	
	public static int getMaxIndex(double[] array){
		double max = Double.NEGATIVE_INFINITY;
		int index = -1;
		for(int i = 0; i < array.length; i++){
			if(array[i] > max){
				max = array[i];
				index = i;
			}
		}				
		return index;
	}
	
	
	
	public static double[][] calcMatrixPowerT(double[][] matrix, int T){		
		double[][] transitionMatrixT = new double[matrix.length][matrix.length];
		Matrix matrixj = new Matrix(matrix);
		Matrix tmpMatrix = matrixj;
		if(T == 1) {
			return matrix;
		}else {
			for(int i = 1; i < T; i++) {
				tmpMatrix = tmpMatrix.times(matrixj);
			}
		}

		for(int i = 0; i < matrix.length; i++) {
			for(int j = 0; j < matrix.length; j++) {
				transitionMatrixT[i][j] = tmpMatrix.get(i, j);
			}
		}
		
		return transitionMatrixT;
	}
	
	
	
	public static boolean isMatrixSame(Matrix matrix1, Matrix matrix2) {
		boolean flag = true;
		
		
		for(int i = 0; i < matrix1.getRowDimension(); i++) {
			for(int j = 0; j < matrix1.getColumnDimension(); j++) {
				double entry1 = matrix1.get(i, j);
				double entry2 = matrix2.get(i, j);
				if(entry1 != entry2) {
					flag = false;
					return flag;
				}
			}
		}
		return flag;
	}
	
	
	public static void printMatrix(double[][] matrix) {
		for(int i = 0; i < matrix.length; i++) {
			for(int j = 0; j < matrix[0].length; j++) {
				System.out.print(matrix[i][j] + " ");
			}
			System.out.println();
		}
	}
	
	
	public static void printArray(double[] array) {
		System.out.println();
		for(int i = 0; i < array.length; i++) {
			System.out.println(array[i] + ",");
		}
		System.out.println();
	}

	public static void printArray(int[] array) {
		System.out.println();
		for(int i = 0; i < array.length; i++) {
			System.out.print(array[i]);
		}
		System.out.println();
	}
	
	public static double getMedian(double[] array) {
		double middle = 0;
        int size = array.length;
        if(size != 0){
            Arrays.sort(array);
            if(size%2 == 0){
                middle = (array[size/2-1]+array[size/2])/2.0;
            }else{
                int inx = size/2;
                middle = array[inx];
            }
        }
        return middle;
	}
	
	
	public static double getMedian(List<Double> list) {
		double middle = 0;
		
		Double[] array = list.toArray(new Double[list.size()]);
		
		
        int size = array.length;
        
        if(size != 0){
            Arrays.sort(array);
            if(size%2 == 0){
                middle = (array[size/2-1]+array[size/2])/2.0;
            }else{
                int inx = size/2;
                middle = array[inx];
            }
        }
        return middle;
	}

	
	
	public static void main(String[] args) {
		double[][] matrix = {{1,1},{1,1}};
		double[][] matrix2 = {{1,1},{1,2}};
		double[][] result = ArrayUtils.calcMatrixPowerT(matrix, 2);
		for(int i = 0; i < result.length; i++) {
			for(int j = 0; j < result.length; j++) {
				System.out.print(result[i][j] + " ");
			}
			System.out.println();
		}
		
		Matrix matrixJ1 = new Matrix(matrix);
		Matrix matrixJ2 = new Matrix(matrix2);
		
		System.out.println(matrixJ1.getRowDimension());
		System.out.println(isMatrixSame(matrixJ1, matrixJ2));
		
		
		
		double arr[] = new double[]{2,5,4,3,6,8,7,1,9};
//      Integer arr[] = new Integer[]{1,2,3,4,5,6};
		List<Double> list = new ArrayList<>();
		list.add(2.0);
		list.add(3.0);
		list.add(1.0);
		list.add(4.0);
		
		

        double res = getMedian(list);
        System.out.println(res);
		
		
		
		
	}
	
	

}
