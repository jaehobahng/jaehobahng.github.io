---
layout: single
title: "Data Preprocessing"
categories: R
tag: coding
# image: \assets\images\2022-07-29-first\TEST.png
---

# Data Preprocessing
## library(dplyr)
### One of the most commonly used libraries for data preprocessing in R.<br/>The sign '%>%' can be used function after function to process data in an intuitive, and concise line of code<br/>

<b>1. filter</b><br/>
Function : filter certain characteristics from a given dataset
<br/>
<br/>
Code Example : iris %>% filter((Species=="setosa"|Species=="versicolor") & Sepal.Width > 3)
<br/>
<br/>
<i>Translation : Filter rows from dataset iris where species is setosa or versicolar and sepal_width is greater than three.</i>

<b>1. filter</b><br/>
Function : filter certain characteristics from a given dataset
<br/>
<br/>
Code Example : iris %>% filter((Species=="setosa"|Species=="versicolor") & Sepal.Width > 3)
<br/>
<br/>
<i>Translation : Filter rows from dataset iris where species is setosa or versicolar and sepal_width is greater than three.</i>





## library(tidyr)
1. <b>psych::describeBy(iris[,1:2], iris$Species, mat=T)</b><br/>
Describe sepal length and sepal width by each Species.<br/>
Decriptions given(number, mean, standard deviation, median, trimmed, min, max, range, skew, kurtosis)
<img src="\assets\images\2022-08-01-DataManipulation\describeBy.jpg" alt="Alt text">


## library(tidyr)
1. <b>psych::describeBy(iris[,1:2], iris$Species, mat=T)</b><br/>
Describe sepal length and sepal width by each Species.<br/>
Decriptions given(number, mean, standard deviation, median, trimmed, min, max, range, skew, kurtosis)
<img src="\assets\images\2022-08-01-DataManipulation\describeBy.jpg" alt="Alt text">

## library(plyr)
1. <b>psych::describeBy(iris[,1:2], iris$Species, mat=T)</b><br/>
Describe sepal length and sepal width by each Species.<br/>
Decriptions given(number, mean, standard deviation, median, trimmed, min, max, range, skew, kurtosis)
<img src="\assets\images\2022-08-01-DataManipulation\describeBy.jpg" alt="Alt text">


## basic given functions
1. <b>psych::describeBy(iris[,1:2], iris$Species, mat=T)</b><br/>
Describe sepal length and sepal width by each Species.<br/>
Decriptions given(number, mean, standard deviation, median, trimmed, min, max, range, skew, kurtosis)
<img src="\assets\images\2022-08-01-DataManipulation\describeBy.jpg" alt="Alt text">