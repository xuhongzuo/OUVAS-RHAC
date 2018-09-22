# OUVAS-RHAC

This is the source code of RHAC algorithm published in CIKM18.

The related paper is 
"Exploring a High-quality Outlying Feature Value Set for Noise-Resilient Outlier Detetection in Categorical Data"
Please cite our paper if you use it. 

The code is implemented in JAVA. Please find the main class "ODUtils" in package OD.
The input can be a single dataset file or a directory of multiple datasets. 
Note that the algorithm can be only performed on datasets in arff format.

RHAC can be directly detect outliers by setting "optionInt" as 0, or can be performed as a feature selection method by setting it as 1. 
