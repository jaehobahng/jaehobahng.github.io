---
layout: splash
title: "Correlation Coefficients"
categories: R
tag: coding
---

# Correlation Coefficients with R

<b>1. Correlation Coefficient</b><br/>
1. Pearson<br/>
 - The most commonly used type of correlation coefficient.<br/>
 - Linear relationship between two variables.<br/>
 - Can't tell the difference between independent and dependent variables. <br/>
 - The stronger the correlation between these two datasets, the closer it'll be to +1 or -1. <br/>
2. Spearman<br/>
 - Used to determine the monotonic relationship between two sets of data. <br/>
 - Based on the ranked values for each dataset and uses skewed or ordinal variables.<br/>
3. Kendall<br/>
 - measures the strength of dependence between two sets of data.<br/>

<br/>
<pre>
Code Example(Normality check) : 
    iris.cor <- cor(iris[1:4], use = "pairwise.complete.obs")
    cor(iris$Sepal.Length,iris$Sepal.Width, use = "pairwise.complete.obs")

use = how to deal with NA values
    - everything : pring NA when there are missing values
    - all.obs : print error message when there are missing values
    - complete.obs : exclude rows with missing values
    - pairwise.complete.obs : exclude missing values when paired
method = pearson, kendall, spearman

Code Example(Visualize) : 
    corrplot(iris.cor, method = "circle")
    corrplot.mixed(cor(variables))
     - Mixed plot used to show both number and visual effect
</pre>
<br/>
