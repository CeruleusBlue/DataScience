# Big Data Mining Techniques and Implementation
# Assignment 2
## Task 1

### Dataset: 

The kddcup.data_10_percent dataset is a small subset of a larger benchmark problem. The benchmark problem aims at assisting researchers to develop methods that can reliably detect network intrusions in real time. A very brief description of the data, the names of attributes and classes, and more can be found on the corresponding source webpage. The classes are DOS (Denial of Service), Probe, R2L (Root 2 Local), U2R (User 2 Root) and Normal.

### Objective:

The objective of the task is to implement a Naive Bayes classifier from scratch without using any machine learning library.

## Task 2

### Dataset:
This dataset contains about 9664 webpages and their hyperlinks. The records with the “n” flag contain (unique) IDs and URLs of webpages. Those with the “e” flag represent hyperlinks between webpages. Essentially, this data set models a Web.

### Objective:
Performance explorative analysis of this data set with Apache Spark. The following definitions are used in this task: The out-degree of a webpage is the number of hyperlinks which that webpage has. The in-degree of a webpage is the number of webpages which have a hyperlink to that webpage.

## Task Requirements:

1. The webpages with largest out-degree
2. The webpages with largest in-degree
3. The average out-degree
4. The average in-degree
5. The count of webpages with out-degree value '0'.

## Task 3

### Dataset:
The same dataset used in **Task 2**.

### Task Requirements:

1. Generate a 5 × 5 block matrix for the transition matrix of the web graph, and a 5 × 1 block matrix for 
the initial rank vector (which is a uniform vector).

2.  Implement the sparse matrix formulation of the PageRank algorithm. Use a teleport probability β =
0.85. The computation of the (evolving) rank vector is iterated 10 times. Then, return the ranking 
scores of the first 20 webpages (i.e., webpages from ID 0 to ID 19). 

*Note: The specific details of each project can be found in their respective directories.*
