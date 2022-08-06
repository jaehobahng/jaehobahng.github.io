---
layout: single
title: "Data Preprocessing"
categories: R
tag: coding
# image: \assets\images\2022-07-29-first\TEST.png
---

# Data Preprocessing
## Basic Dataset properties

1.  <b>str(iris)</b><br/>
Column properties of a given dataset
<img src="\assets\images\2022-08-01-DataManipulation\str(iris).jpg" alt="Alt text">





## Basic overview of dataset columns
1. <b>psych::describeBy(iris[,1:2], iris$Species, mat=T)</b><br/>
Describe sepal length and sepal width by each Species.<br/>
Decriptions given(number, mean, standard deviation, median, trimmed, min, max, range, skew, kurtosis)
<img src="\assets\images\2022-08-01-DataManipulation\describeBy.jpg" alt="Alt text">