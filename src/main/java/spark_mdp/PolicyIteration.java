package spark_mdp;

import java.util.ArrayList;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.BlockMatrix;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;
import org.apache.spark.storage.StorageLevel;

import scala.Tuple2;

public class PolicyIteration {
	static private Long num_states;
	static private Long num_actions;
	static private Double discount;
	static private JavaSparkContext sc;  
	
	static void main(String[] args) throws IllegalArgumentException{
		sc = new JavaSparkContext(new SparkConf().setAppName("Policy Evaluation"));
		
		num_states = Long.parseLong(args[0]);
		num_actions = Long.parseLong(args[1]);
		discount = Double.parseDouble(args[2]);
		
		final Broadcast<Long> num_actions_BroadCast = sc.broadcast(num_actions);
		final Broadcast<Double> discount_BroadCast = sc.broadcast(discount);
	
		checkAgruements(args);
		
		BlockMatrix P = new CoordinateMatrix(sc.textFile(args[4]).flatMap(new FlatMapFunction<String, MatrixEntry>() {
			private static final long serialVersionUID = 328024652664764883L;
			
			private static final double error_bound = 1e-6;
			
			@Override
			public Iterable<MatrixEntry> call(String t) throws Exception {
				ArrayList<MatrixEntry> output = new ArrayList<MatrixEntry>();
				String[] parts = t.split("#");
				Long state1 = Long.parseLong(parts[0]);
				double total = 0;
				for (int i = 1; i < parts.length; i++) {
					String[] sub_parts = parts[i].split(",");
					Long action = Long.parseLong(sub_parts[0]);
					Long state2 = Long.parseLong(sub_parts[1]);
					Double probability = Double.parseDouble(sub_parts[2]);
					total += probability;
					
					output.add(new MatrixEntry((state1 - 1) * num_actions_BroadCast.value() + action, state2, discount_BroadCast.value() * probability));
				}
				if (Math.abs(1 - total) > error_bound) {
					throw new Exception(String.format(Strings.PROBABILITY_ERROR, state1));
				}
				return output;
			}
		}).rdd(), num_states * num_actions, num_states).toBlockMatrix().persist(StorageLevel.MEMORY_AND_DISK_SER());
		
		JavaPairRDD<Long, Tuple2<Long, Double>> rewards = sc.textFile(args[5]).mapToPair(new PairFunction<String, Long, Tuple2<Long, Double>>() {
			private static final long serialVersionUID = -5586934805066631406L;

			@Override
			public Tuple2<Long, Tuple2<Long, Double>> call(String t) throws Exception {
				String[] parts = t.split(",");
				Long state = Long.parseLong(parts[0]);
				Long action = Long.parseLong(parts[1]);
				Double r = Double.parseDouble(parts[2]);
				
				return new Tuple2<Long, Tuple2<Long, Double>>(state, new Tuple2<Long, Double>(action, r));
			}
			
		}).persist(StorageLevel.MEMORY_AND_DISK_SER());
		
		BlockMatrix r = new IndexedRowMatrix(rewards.map(new Function<Tuple2<Long, Tuple2<Long, Double>>, IndexedRow>() {
			private static final long serialVersionUID = 7303753406986104844L;

			@Override
			public IndexedRow call(Tuple2<Long, Tuple2<Long, Double>> v1) throws Exception {
				double[] value = {v1._2._2};
				return new IndexedRow((v1._1 - 1) * num_actions_BroadCast.value() + v1._2._1, new DenseVector(value));
			}
			
		}).rdd(), num_states * num_actions, 1).toBlockMatrix().persist(StorageLevel.MEMORY_AND_DISK_SER());
				
		//generate initial policy
		JavaPairRDD<Long, Tuple2<Long, Double>> initial = rewards.reduceByKey(new InitialReducer()).persist(StorageLevel.MEMORY_AND_DISK_SER());
		
		JavaPairRDD<Long, Long> policy = initial.mapToPair(new PairFunction<Tuple2<Long, Tuple2<Long, Double>>, Long, Long>() {
			private static final long serialVersionUID = -8189058367107771592L;

			@Override
			public Tuple2<Long, Long> call(Tuple2<Long, Tuple2<Long, Double>> t) throws Exception {
				return new Tuple2<Long, Long>(t._1, t._2._1);
			}
		});
		
		BlockMatrix v_initial = new IndexedRowMatrix(initial.map(new Function<Tuple2<Long, Tuple2<Long, Double>>, IndexedRow>() {
			private static final long serialVersionUID = 3919362531916209746L;

			@Override
			public IndexedRow call(Tuple2<Long, Tuple2<Long, Double>> v1) throws Exception {
				double[] value = {v1._2._2};
				return new IndexedRow(v1._1, new DenseVector(value));
			}
			
		}).rdd(), num_states, 1).toBlockMatrix();
		
	}
	
	static void checkAgruements(String[] args) throws IllegalArgumentException {
		
	}
	
	static class InitialReducer implements Function2<Tuple2<Long, Double>, Tuple2<Long, Double>, Tuple2<Long, Double>> {
		private static final long serialVersionUID = -2748441034957608709L;

		@Override
		public Tuple2<Long, Double> call(Tuple2<Long, Double> v1, Tuple2<Long, Double> v2) throws Exception {
			if (v2._2 > v1._2) {
				return v2;
			} else {
				return v1;
			}
		}
		
	}
}
