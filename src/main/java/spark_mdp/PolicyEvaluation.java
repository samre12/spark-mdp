package spark_mdp;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.distributed.*;
import org.apache.spark.storage.StorageLevel;

public class PolicyEvaluation {
	static private Long num_states;
	static private Long num_actions;
	static private Double discount;
	static private Double epsilon;
	
	static void main(String[] args) throws IllegalArgumentException{
		JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("Policy Evaluation"));
		
		num_states = Long.parseLong(args[0]);
		num_actions = Long.parseLong(args[1]);
		discount = Double.parseDouble(args[2]);
		epsilon = Double.parseDouble(args[3]);
		
		final Broadcast<Long> num_actions_BroadCast = sc.broadcast(num_actions);
		final Broadcast<Double> discount_BroadCast = sc.broadcast(discount);
		
		checkAgruements(args);
		BlockMatrix Pi = new CoordinateMatrix(sc.textFile(args[4]).map(new Function<String, MatrixEntry>() {
			private static final long serialVersionUID = 328024652664764883L;

			@Override
			public MatrixEntry call(String s) throws Exception {
				String[] parts = s.split(",");
				Long state = Long.parseLong(parts[0]);
				Long action = Long.parseLong(parts[1]);
				return new MatrixEntry(state, (state - 1) * num_actions_BroadCast.value() + action, 1);
			}
			
		}).rdd(), num_states, num_states * num_actions).toBlockMatrix().persist(StorageLevel.MEMORY_ONLY_SER());
		
		
		BlockMatrix P = new CoordinateMatrix(sc.textFile(args[5]).map(new Function<String, MatrixEntry>() {
			private static final long serialVersionUID = 328024652664764883L;

			@Override
			public MatrixEntry call(String s) throws Exception {
				String[] parts = s.split(",");
				Long state1 = Long.parseLong(parts[0]);
				Long action = Long.parseLong(parts[1]);
				Long state2 = Long.parseLong(parts[2]);
				Double probability = Double.parseDouble(parts[3]);
				return new MatrixEntry((state1 - 1) * num_actions_BroadCast.value() + action, state2, discount_BroadCast.value() * probability);
			}
		}).rdd(), num_states * num_actions, num_states).toBlockMatrix();
		
		BlockMatrix r = new CoordinateMatrix(sc.textFile(args[5]).map(new Function<String, MatrixEntry>() {
			private static final long serialVersionUID = -5784388060110611464L;

			@Override
			public MatrixEntry call(String s) throws Exception {
				String[] parts = s.split(",");
				Long state = Long.parseLong(parts[0]);
				Long action = Long.parseLong(parts[1]);
				Double r = Double.parseDouble(parts[2]);
				return new MatrixEntry((state - 1) * num_actions_BroadCast.value() + action, 1, r);
			}
			
		}).rdd(), num_states * num_actions, 1).toBlockMatrix();
		
		BlockMatrix H = Pi.multiply(P).persist(StorageLevel.MEMORY_AND_DISK_SER());
		BlockMatrix vBar = Pi.multiply(r).persist(StorageLevel.MEMORY_AND_DISK_SER());
		//vBar also serves as the initial approximation of v
		
		sc.stop();
	}
	
	static void checkAgruements(String[] args) throws IllegalArgumentException {
		
	}
	
	static boolean checkDistance(BlockMatrix v1, BlockMatrix v2) {
		return false;
	}
}
