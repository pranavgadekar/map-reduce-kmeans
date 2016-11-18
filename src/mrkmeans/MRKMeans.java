package mrkmeans;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeMap;

import org.apache.commons.compress.utils.IOUtils;
import org.apache.commons.math3.linear.FieldDecompositionSolver;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.examples.pi.Combinable;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

class P1Mapper extends Mapper<LongWritable, Text, IntWritable, Text> {
	// List of Centroids
	private static ArrayList<ArrayList<Double>> centroids = new ArrayList<>();
	private Text exprString = new Text();

	// Setup function is used to read random centroids or previous centroids
	// stored in file
	@SuppressWarnings("deprecation")
	@Override
	protected void setup(Mapper<LongWritable, Text, IntWritable, Text>.Context context)
			throws IOException, InterruptedException {
		// TODO Auto-generated method stub

		Configuration conf = context.getConfiguration();
		FileSystem fs = FileSystem.get(conf);
		Path centroidP = new Path("centroidPoints.txt");

		if (fs.exists(centroidP)) {
			centroids.clear();
			// SequenceFile Reader is used to read and write to centroids file
			try (SequenceFile.Reader reader = new SequenceFile.Reader(fs, centroidP, conf)) {
				IntWritable key = new IntWritable();
				Text val = new Text();
				while (reader.next(key, val)) {
					ArrayList<Double> l = fillCentroids(key.get(), val.toString());
					centroids.add(l);
				}
			}
		} else {
			centroidP = new Path("centroidPoints2.txt");
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(centroidP)));
			String line;
			line = br.readLine();
			while (line != null) {
				fillCentroidMap(line);
				line = br.readLine();
			}
			br.close();
		}
	}

	// Centroid contains list of all expressions of centroids
	private static void fillCentroidMap(String s) {
		String[] split = s.split("\\t");
		ArrayList<Double> list = new ArrayList<Double>();

		for (int i = 2; i < split.length; i++) {
			list.add(Double.parseDouble(split[i]));
		}
		centroids.add(list);
	}

	private static ArrayList<Double> fillCentroids(int k, String string) {
		ArrayList<Double> output = new ArrayList<Double>();
		String newString = string.substring(1, string.length() - 1);
		String[] doubleVal = newString.split(",");
		for (int i = 0; i < doubleVal.length; i++) {
			output.add(Double.parseDouble(doubleVal[i]));
		}
		return output;
	}

	// mapper function takes in line of each file and emits<clusterID, gene_id +
	// expression values>
	//
	@Override
	protected void map(LongWritable key, Text value, Mapper<LongWritable, Text, IntWritable, Text>.Context context)
			throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		// super.map(key, value, context);
		String[] lineSplit = value.toString().trim().split("\\t");
		ArrayList<Double> exp = new ArrayList<Double>();
		int geneId = Integer.parseInt(lineSplit[0]);
		for (int i = 2; i < lineSplit.length; i++) {
			exp.add(Double.parseDouble(lineSplit[i]));
		}
		int min = -1;
		Double minValue = Double.MAX_VALUE;
		for (int i = 0; i < centroids.size(); i++) {

			Double tmp = calculateEucledianDistance(exp, centroids.get(i));
			if (minValue > tmp) {
				minValue = tmp;
				min = i;
			}
		}

		String op = "";
		for (int i = 0; i < lineSplit.length; i++) {
			if (i != 1)
				op += lineSplit[i] + "$";
		}
		exprString.set(op.substring(0, op.length() - 1));

		IntWritable clusterId = new IntWritable(min);
		context.write(clusterId, exprString);

	}

	// Calculates Eucledian Distance between 2 genes
	public static double calculateEucledianDistance(ArrayList<Double> list1, ArrayList<Double> list2) {
		int size = list2.size();

		double euclideanDist = 0;
		for (int i = 0; i < size; i++) {
			double val1 = list1.get(i);
			double val2 = list2.get(i);
			double diff = val1 - val2;
			diff = diff * diff;
			euclideanDist += diff;
		}
		return Math.sqrt(euclideanDist);
	}

}

// Reducer gets input in form <clusterID, List(gene_id + expression values)>
// it emits <clusterID, List(gene_id)>

class P1Reducer extends Reducer<IntWritable, Text, IntWritable, Text> {
	public static TreeMap<Integer, ArrayList<Double>> centroidMap = new TreeMap<>();
	public static TreeMap<Integer, ArrayList<Double>> prevMap = new TreeMap<>();
	private Text outStr = new Text();

	@Override
	protected void reduce(IntWritable arg0, Iterable<Text> arg1,
			Reducer<IntWritable, Text, IntWritable, Text>.Context arg2) throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		// super.reduce(arg0, arg1, arg2);

		HashMap<Integer, ArrayList<Double>> map = new HashMap<>();
		int expLength = -1;
		for (Text t : arg1) {
			String[] sA = t.toString().split("\\$");
			ArrayList<Double> list = new ArrayList<>();
			expLength = sA.length - 1;
			for (int i = 1; i < sA.length; i++) {
				list.add(Double.parseDouble(sA[i]));
			}
			map.put(Integer.parseInt(sA[0]), list);
		}

		int numberOfGenesInCluster = map.size();
		ArrayList<Double> centroidExpr = new ArrayList<>();
		for (int j = 0; j < expLength; j++) {
			double sum = 0;
			for (ArrayList<Double> ll : map.values()) {
				sum = sum + ll.get(j);
			}
			sum = sum / numberOfGenesInCluster;
			centroidExpr.add(sum);
		}
		centroidMap.put(arg0.get(), centroidExpr);

		// PRINT THIS TO CENTROID FILE
		outStr.set(map.keySet().toString());
		arg2.write(arg0, outStr);
	}

	// It checks whether previous clusters and newly generated clusters are same
	// if same then write into file and stop map reduce while loop
	@SuppressWarnings("deprecation")
	@Override
	protected void cleanup(Reducer<IntWritable, Text, IntWritable, Text>.Context context)
			throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		// super.cleanup(context);
		Configuration conf = context.getConfiguration();
		FileSystem fs = FileSystem.get(conf);

		Path centroids = new Path("centroidPoints.txt");

		if (fs.exists(centroids)) {
			// Read file
			prevMap.clear();
			try (SequenceFile.Reader reader = new SequenceFile.Reader(fs, centroids, conf)) {
				IntWritable key = new IntWritable();
				Text val = new Text();
				while (reader.next(key, val)) {
					ArrayList<Double> l = fillCentroidsIntoPrevMap(key.get(), val.toString());
					prevMap.put(key.get(), l);
				}
			}
			// store result into oldMap
			fs.delete(centroids, true);
		}
		// Compare centroidMap with oldmap

		if (AreTwoClustersEqual(prevMap, centroidMap)) {
			Path flag = new Path("flag.txt");
			try (SequenceFile.Writer out = SequenceFile.createWriter(fs, context.getConfiguration(), flag,
					IntWritable.class, Text.class)) {
				out.append(new IntWritable(1), new Text("Convereged"));
			}
		}

		// if true generate flag file
		try (SequenceFile.Writer out = SequenceFile.createWriter(fs, context.getConfiguration(), centroids,
				IntWritable.class, Text.class)) {
			for (Integer center : centroidMap.keySet()) {
				IntWritable value = new IntWritable(center);
				Text t = new Text();
				t.set(centroidMap.get(center).toString());
				out.append(value, t);
			}
		}
	}

	// Checks if two clusters are identical
	private static boolean AreTwoClustersEqual(TreeMap<Integer, ArrayList<Double>> prevCluster,
			TreeMap<Integer, ArrayList<Double>> clusters) {
		if (prevCluster.size() == 0 || clusters.size() == 0)
			return false;
		for (int i = 0; i < clusters.size(); i++) {
			ArrayList<Double> list1 = clusters.get(i);
			ArrayList<Double> list2 = prevCluster.get(i);
			if (!equalLists(list1, list2))
				return false;
		}

		return true;
	}

	// helper function to check two expression lists
	public static boolean equalLists(ArrayList<Double> one, ArrayList<Double> two) {
		if (one == null || two == null) {
			return false;
		}

		if (one.size() != two.size()) {
			return false;
		}

		one = new ArrayList<Double>(one);
		two = new ArrayList<Double>(two);

		// Collections.sort(one);
		// Collections.sort(two);
		return one.equals(two);
	}

	// centroids are saved in a map for checking next iteration
	private ArrayList<Double> fillCentroidsIntoPrevMap(int k, String string) {
		// TODO Auto-generated method stub
		ArrayList<Double> output = new ArrayList<Double>();
		String newString = string.substring(1, string.length() - 1);
		String[] doubleVal = newString.split(",");
		for (int i = 0; i < doubleVal.length; i++) {
			output.add(Double.parseDouble(doubleVal[i]));
		}
		return output;
	}

}

public class MRKMeans {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		int k = 15;
		int[] array = {};
		Path out = new Path(args[1]);
		Path in = new Path("iyer.txt");

		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		Path flag = new Path("flag.txt");
		Path oldCentroids = new Path("centroidPoints.txt");
		if (fs.exists(oldCentroids))
			fs.delete(oldCentroids, true);
		Path centroidFirst = new Path("centroidPoints2.txt");
		if (fs.exists(centroidFirst))
			fs.delete(centroidFirst, true);

		// FSDataOutputStream dataOutputStream = fs.create(centroidFirst, true);
		FileWriter dataOutputStream = new FileWriter("centroidPoints2.txt");
		ArrayList<String> list = giveStringArray(k, in, fs, array);
		for (String s : list) {
			dataOutputStream.write(s);
			dataOutputStream.write("\n");
		}
		dataOutputStream.close();

		while (true) {
			Job job = Job.getInstance(conf, "MRKMeans");
			// Path center = new Path("centroidPoints1.txt");
			// conf.set("centroidPath", center.toString());
			job.setJarByClass(MRKMeans.class);
			job.setMapperClass(P1Mapper.class);

			job.setReducerClass(P1Reducer.class);

			if (fs.exists(out))
				fs.delete(out, true);

			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(Text.class);
			FileInputFormat.addInputPath(job, in);
			FileOutputFormat.setOutputPath(job, new Path(args[1]));

			job.waitForCompletion(true);

			if (fs.exists(flag)) {
				fs.delete(flag, true);
				System.exit(job.waitForCompletion(true) ? 0 : 1);
			}
		}
	}

	// return initial centroids
	private static ArrayList<String> giveStringArray(int k, Path path, FileSystem fs, int[] array) throws IOException {
		ArrayList<String> result = new ArrayList<>();
		BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
		String line;
		line = br.readLine();
		int count = 0;
		while (line != null) {
			line = br.readLine();
			count++;

		}
		br.close();
		int size = count;
		br = new BufferedReader(new InputStreamReader(fs.open(path)));
		line = br.readLine();
		count = 0;
		ArrayList<Integer> ll = new ArrayList<>();
		if (array.length == 0) {
			ll = randomCentroids(k, size);
		} else {

			for (int i = 0; i < array.length; i++) {
				ll.add(array[i]);
			}
		}
		while (line != null) {
			line = br.readLine();
			count++;
			if (ll.contains(count)) {
				result.add(line);
			}
		}
		br.close();
		return result;

	}

	// assigns random centroids
	public static ArrayList<Integer> randomCentroids(int k, int size) {
		ArrayList<Integer> list = new ArrayList<>();
		Random randomNumberGenerator = new Random();
		for (int i = 0; i < k; i++) {
			int randomNumber = randomNumberGenerator.nextInt(size);
			if (randomNumber == 0) {
				randomNumber = randomNumberGenerator.nextInt(size);
			}
			list.add(randomNumber);
		}
		return list;
	}
}
