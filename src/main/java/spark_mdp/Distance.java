package spark_mdp;

import java.util.Comparator;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.distributed.BlockMatrix;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;

public class Distance {
	static JavaRDD<Double> covert(BlockMatrix v1, BlockMatrix v2) {
		JavaRDD<IndexedRow> rows = v2.toIndexedRowMatrix().rows().toJavaRDD();
		JavaRDD<Double> output = rows.map(new Function<IndexedRow, Double>() {
			private static final long serialVersionUID = 2714620118336324819L;

			@Override
			public Double call(IndexedRow row) throws Exception {
				return row.vector().apply(0);
			}
			
		});
		return output;
		
	}
	
	static Double supNorm(JavaRDD<Double> vector) {
		return vector.max(new Comparator<Double>() {
			@Override
			public int compare(Double o1, Double o2) {
				if (Math.abs(o1) < Math.abs(o2)) {
					return -1;
				} else if (Math.abs(o1) < Math.abs(o2)) {
					return 1;
				} else {
					return 0;
				}
			}
		});
	}
	
	static Double pNorm(JavaRDD<Double> vector, int p, JavaSparkContext sc) {
		final Broadcast<Integer> pBroadCast = sc.broadcast(p);
		return Math.pow(vector.map(new Function<Double, Double>() {
			private static final long serialVersionUID = 5486848485177887439L;

			@Override
			public Double call(Double v1) throws Exception {
				return Math.pow(v1, pBroadCast.value());
			}
			
		}).reduce(new Function2<Double, Double, Double>() {
			private static final long serialVersionUID = 7984983290020732838L;

			@Override
			public Double call(Double v1, Double v2) throws Exception {
				return v1 + v2;
			}
			
		}), 1 / ((double) p));
	}
}
