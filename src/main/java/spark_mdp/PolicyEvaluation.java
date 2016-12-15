package spark_mdp;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.mllib.linalg.distributed.*;
import org.apache.spark.storage.StorageLevel;

public class PolicyEvaluation {
	static private Long num_states;
	static private Long num_actions;
	static private Double discount;
	
	static void main(String[] args) throws IllegalArgumentException{
		JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("MDP"));
		
		num_states = Long.parseLong(args[0]);
		num_actions = Long.parseLong(args[1]);
		discount = Double.parseDouble(args[2]);
		
		checkAgruements(args);
		BlockMatrix Pi = new CoordinateMatrix(sc.textFile(args[3]).map(new Function<String, MatrixEntry>() {
			/**
			 * 
			 */
			private static final long serialVersionUID = 328024652664764883L;

			@Override
			public MatrixEntry call(String v1) throws Exception {
				// TODO Auto-generated method stub
				return null;
			}
			
		}).rdd(), num_states, num_states * num_actions).toBlockMatrix().persist(StorageLevel.MEMORY_ONLY_SER());
		
		
		BlockMatrix P = new CoordinateMatrix(sc.textFile(args[4]).map(new Function<String, MatrixEntry>() {
			/**
			 * 
			 */
			private static final long serialVersionUID = 328024652664764883L;

			@Override
			public MatrixEntry call(String v1) throws Exception {
				// TODO Auto-generated method stub
				return null;
			}
		}).rdd(), num_states * num_actions, num_states).toBlockMatrix();
		
		BlockMatrix r = new CoordinateMatrix(sc.textFile(args[5]).map(new Function<String, MatrixEntry>() {
			/**
			 * 
			 */
			private static final long serialVersionUID = -5784388060110611464L;

			@Override
			public MatrixEntry call(String v1) throws Exception {
				// TODO Auto-generated method stub
				return null;
			}
			
		}).rdd(), num_states * num_actions, 1).toBlockMatrix();
		
		BlockMatrix H = Pi.multiply(P).persist(StorageLevel.MEMORY_AND_DISK_SER());
		BlockMatrix vBar = Pi.multiply(r).persist(StorageLevel.MEMORY_AND_DISK_SER());
		
		sc.stop();
	}
	
	static void checkAgruements(String[] args) throws IllegalArgumentException {
		
	}
}
