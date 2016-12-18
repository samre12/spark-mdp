package spark_mdp;

import java.util.ArrayList;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.*;
import org.apache.spark.storage.StorageLevel;

import scala.Tuple2;

public class PolicyEvaluation {
	static private Long num_states;
	static private Long num_actions;
	static private Double discount;
	static private Integer distance;
	static private Double epsilon;
	static private BlockMatrix negative_I;
	static private JavaSparkContext sc;
	
	static void main(String[] args) throws IllegalArgumentException{
		sc = new JavaSparkContext(new SparkConf().setAppName("Policy Evaluation"));
		
		num_states = Long.parseLong(args[0]);
		num_actions = Long.parseLong(args[1]);
		discount = Double.parseDouble(args[2]);
		distance = Integer.parseInt(args[3]);
		epsilon = Double.parseDouble(args[4]);
		
		final Broadcast<Long> num_actions_BroadCast = sc.broadcast(num_actions);
		final Broadcast<Double> discount_BroadCast = sc.broadcast(discount);
		
		checkAgruements(args);
		
		BlockMatrix P = new CoordinateMatrix(sc.textFile(args[5]).flatMap(new FlatMapFunction<String, MatrixEntry>() {
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
		}).rdd(), num_states * num_actions, num_states).toBlockMatrix();
		
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
			
		});
		
		BlockMatrix r = new IndexedRowMatrix(rewards.map(new Function<Tuple2<Long, Tuple2<Long, Double>>, IndexedRow>() {

			@Override
			public IndexedRow call(Tuple2<Long, Tuple2<Long, Double>> v1) throws Exception {
				double[] value = {v1._2._2};
				return new IndexedRow((v1._1 - 1) * num_actions_BroadCast.value() + v1._2._1, new DenseVector(value));
			}
			
		}).rdd(), num_states * num_actions, 1).toBlockMatrix();
		
		negative_I = new CoordinateMatrix(sc.textFile(args[5]).map(new Function<String, MatrixEntry>() {
			private static final long serialVersionUID = -1259242904008895467L;

			@Override
			public MatrixEntry call(String v1) throws Exception {
				String[] parts = v1.split("#");
				Long state = Long.parseLong(parts[0]);
				return new MatrixEntry(state, state, -1);
			}
		}).rdd(), num_states, num_states).toBlockMatrix().persist(StorageLevel.MEMORY_AND_DISK_SER());
		
		BlockMatrix Pi = new CoordinateMatrix(sc.textFile(args[7]).map(new Function<String, MatrixEntry>() {
			private static final long serialVersionUID = 328024652664764883L;

			@Override
			public MatrixEntry call(String s) throws Exception {
				String[] parts = s.split(",");
				Long state = Long.parseLong(parts[0]);
				Long action = Long.parseLong(parts[1]);
				return new MatrixEntry(state, (state - 1) * num_actions_BroadCast.value() + action, 1);
			}
			
		}).rdd(), num_states, num_states * num_actions).toBlockMatrix().persist(StorageLevel.MEMORY_ONLY_SER());
		
		BlockMatrix H = Pi.multiply(P).persist(StorageLevel.MEMORY_AND_DISK_SER());
		BlockMatrix vBar = Pi.multiply(r).persist(StorageLevel.MEMORY_AND_DISK_SER());
		//vBar also serves as the initial approximation of v
		
		BlockMatrix v_prev = vBar;
		BlockMatrix v_next = vBar.add(H.multiply(v_prev));
		
		int counter = 0; //counts number of iterations to convergence
		
		while(!checkDistance(v_next, v_prev)){
			v_prev = v_next;
			v_next = vBar.add(H.multiply(v_prev));
			counter++;
		}
		
		System.out.println(String.format("The Number of iterations required for convergenceare : %d", counter)); 
		
		JavaPairRDD<Long, Double> output = v_next.toIndexedRowMatrix().rows().toJavaRDD().mapToPair(new PairFunction<IndexedRow, Long, Double>() {
			private static final long serialVersionUID = 1885394162470517392L;

			@Override
			public Tuple2<Long, Double> call(IndexedRow t) throws Exception {
				return new Tuple2<Long, Double>(t.index(), t.vector().apply(0));
			}
		});
		output.saveAsTextFile(args[8]);
		sc.stop();
	}
	
	static void checkAgruements(String[] args) throws IllegalArgumentException {
		//checks whether the arguments input by the user are correct 
	}
	
	static JavaRDD<Double> convert(BlockMatrix v1, BlockMatrix v2) {
		JavaRDD<IndexedRow> rows = v1.add(v2.multiply(negative_I)).toIndexedRowMatrix().rows().toJavaRDD();
		JavaRDD<Double> output = rows.map(new Function<IndexedRow, Double>() {
			private static final long serialVersionUID = 2714620118336324819L;

			@Override
			public Double call(IndexedRow row) throws Exception {
				return row.vector().apply(0);
			}
			
		});
		return output;
	}
	
	static boolean checkDistance(BlockMatrix v1, BlockMatrix v2) {
		JavaRDD<Double> vector = convert(v1, v2);
		double norm;
		if (distance == 0) {
			norm = Distance.supNorm(vector);
		} else {
			norm = Distance.pNorm(vector, 2, sc);
		}
		if (norm < epsilon) {
			return true;
		} else {
			return false;
		}
	} 
}
