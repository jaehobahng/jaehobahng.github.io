---
layout: splash
title: "Basic Dataset Summary"
categories: R
tag: coding
# image: \assets\images\2022-07-29-first\TEST.png
---

# Basic Dataset Summary functions
## Basic Dataset properties

1.  <b>str(iris)</b><br/>
Column properties of a given dataset<br/>
<img src="\assets\images\2022-08-01-DataManipulation\str(iris).jpg" alt="Alt text">

2. <b>dim(iris)</b><br/>
how many rows and columns does the dataset consist of?<br/>
<img src="\assets\images\2022-08-01-DataManipulation\dim(iris).jpg" alt="Alt text">

3. <b>head(iris,6)</b><br/>
Choose first 6 rows of a given dataset<br/>
<img src="\assets\images\2022-08-01-DataManipulation\head(iris).jpg" alt="Alt text">


## Check proportions of properties
1. <b>table(iris$Species)</b><br/>
Check how many of each Species there are in the 150 rows<br/>
<img src="\assets\images\2022-08-01-DataManipulation\table(iris)2.jpg" alt="Alt text">

   <b>1.1 table(iris$Species,round(iris$Sepal.Length,0))</b>
   <br/>
    Check how many rows there are of each species + distribution of the sepal length rounded to the ones place<br/>
<img src="\assets\images\2022-08-01-DataManipulation\table(iris).jpg" alt="Alt text">

2. <b>prop.table(table(iris$Species,round(iris$Sepal.Length,0)),1)</b>
<br/>
Calculate ratio of each combination found in the table function<br/>
prop.table(table(),1) = ratio in row<br/>
prop.table(table(),2) = ratio in column<br/>
<img src="\assets\images\2022-08-01-DataManipulation\prop.table(iris).jpg" alt="Alt text">



## Basic overview of dataset columns
1. <b>psych::describeBy(iris[,1:2], iris$Species, mat=T)</b><br/>
Describe sepal length and sepal width by each Species.<br/>
Decriptions given(number, mean, standard deviation, median, trimmed, min, max, range, skew, kurtosis)<br/>
<img src="\assets\images\2022-08-01-DataManipulation\describeBy.jpg" alt="Alt text">

2. <b>skimr::skim(Cars93)</b><br/>
Detailed overview of selected columns of a dataset. Details given differ depending on whether the column is a numeric or a factor column.<br/>
<img src="\assets\images\2022-08-01-DataManipulation\skim.jpg" alt="Alt text">
3. <b>aggregate</b><br/>
Group and summarize a numeric column by a factor column. Which function to use(mean, max, min etc) must be given manually<br/>
<img src="\assets\images\2022-08-01-DataManipulation\aggregate.jpg" alt="Alt text">

4. <b>pairs(iris[1:4], panel = panel.smooth, main = "iris data", col = iris$Species)</b><br/>
A visual scatter plot of how each numerical variable is correlated to one another with one another.<br/>
With an additional 'col' input, all relationships can be shown in respect to a given factor<br/>
<img src="\assets\images\2022-08-01-DataManipulation\pairs.jpg" alt="Alt text">