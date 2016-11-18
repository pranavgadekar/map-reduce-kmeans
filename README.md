<h2>Implementation of KMeans clustering algorithm of genetic data using Map Reduce in Hadoop</h2>

<h4><p>This is an implementation of KMeans Clustering algorithm in Map Reduce using Hadoop.<p>
<p>The input data has to be kept in a text file and has to be of the format {gene_id expression1 expression2 ...... expressionN}</p>
<p>The output of the Map Reduce jobs is given as follows:
<ul>
  <li>{cluster_ID_1 [gene_id1 gene_id2 ..... gene_idN]}</li>
  <li>{cluster_ID_2 [gene_id1 gene_id2 ..... gene_idN]}</li>
  ......
  <li>{cluster_ID_N [gene_id1 gene_id2 ..... gene_idN]}</li>
 </ul></p>
 
<p>The number of clusters to be formed can be changed by changing the value of parameter "N" in the main()</p>
 
</h4>
