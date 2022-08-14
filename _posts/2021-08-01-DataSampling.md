---
layout: splash
title: "Data Sampling"
categories: R
tag: coding
---

# Data Sampling
## Oversampling<br/>

<b>1. UBL::ADASYN</b><br/>
1. Creates a different number of samples for each observation
2. Uses a type of weight to choose the synthetic sample size and weight correlates to the number of majority class within the knn
3. This is done in order to prevent the machine from ignoring minority classes that are surrounded by majority classes
<img src="\assets\images\DataSampling\ADASYN.jpg" alt="Alt text">
source : https://givitallugot.github.io/articles/2021-07/Python-imbalanced-sampling-copy

<br/>
<pre>
Code Example(Numeric) : 
AdasynClassif(Species~.,
    iris[-c(45:75),],     
    baseClass="virginica",
    dist = "Euclidean",
    beta = 1)   #Optional

Code Example(Factors included) : 
AdasynClassif(AirBags~.,
    Cars93[complete.cases(Cars93),],
    baseClass = 'Driver only',
    dist = "HEOM",
    beta = 1)
</pre>

Distances to use
1. Numeric features(Euclidean, manhattan, canberra, chebyshev, p-norm)
2. Nominal features(Overlap)
3. Both numeric and nominal features(HEOM,HVDM)

<br/>

<b>1. UBL::SMOTE</b><br/>
1. Uses KNN method to find nearest minority observations from a preselected minority observation
2. Between the preselected observation and the knn-chosen observations, additional samples are created within the linear space between the two points
<img src="\assets\images\DataSampling\SMOTE.jpg" alt="Alt text">
source : https://givitallugot.github.io/articles/2021-07/Python-imbalanced-sampling-copy

<br/>
<pre>
Code Example(numeric) : 
dat <- iris[, c(1, 2, 5)]
dat$Species <- factor(ifelse(dat$Species == "setosa", "rare", "common")) 
newData <- SmoteClassif(Species ~ ., 
                        dat, 
                        C.perc = list(common = 1,rare = 6))

Code Example(nominal+numeric)
smotecar <- SmoteClassif(AirBags~.,
             Cars93[complete.cases(Cars93),],
             dist = "HEOM",
             C.perc = list('Driver & Passenger' = 2.2, 'None' = 1.4))
</pre>
<br/>

<b>3. Borderline SMOTE</b><br/>
1. Similar to the SMOTE method, but choose only the minority observations where the majority of the knn neighbors are the majority.
2. The minority observations are divided into safe(less than k/2 observaitons are majority), danger(k/2 to k observations are majority) and noise(all knn's are majority).
3. Observaitons that are classified as danger create samples between the minority and majority samples, thus creating an additional borderlike minority samples
<br/>
<br/>
<pre>
library(smotefamily)

iris.m <- iris[1:130,]
table(iris.m$Species)

BLSMOTE(iris.m[,1:4],
    iris.m$Species,
    K=2,                 #Number of nearest neighbors during sampling process
    dupSize=2,           #desired times of synthetic majority instances over original majority instances, 0 for duplicating until balanced
    C=4)                 #number of nearest neighbors during calculating safe-level process
</pre>
<img src="\assets\images\DataSampling\BorderlineSMOTE.png" alt="Alt text">
source : Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning(2005)<br/><br/>


## Undersampling<br/>

<b>1. unbalanced::Tomek Links</b><br/>
1. Choose a pair of minority and majority observations where the link is the shortest of all possible links from the minoirty observaiton
2. Eliminate the majority observations formed by the link so the model's boundaries are shifted more towards the majority observations, thus allowing the borderline observations to be classified more as the minority gorup.
<br/><img src="\assets\images\DataSampling\TomekLink.png" alt="Alt text">
source : https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets

<br/>
<pre>
Code Example: titanic dataset
library(unbalanced)
# make Tomek links
set.seed(123)

# split data (train, test)
train_idx<- createDataPartition(y = train$Survived, p=0.7, list = FALSE)
Tomek_train<- train[train_idx,]
Tomek_test<- train[-train_idx,]

# make Tomek data set
input = c('Age', 'SibSp', 'Parch', 'Fare', 'family_size') # Columns to use
Tomek<-ubTomek(X=Tomek_train[,input], Y= Tomek_train[,'Survived'])
# Tomek$id.rm --> removed data index
# 91 data(0 class removed. )

</pre>
<br/>