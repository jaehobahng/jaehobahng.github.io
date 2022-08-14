---
layout: splash
title: "Dimensionality Reduction Techniques"
categories: R
tag: coding
---

# Principle Component Analysis vs. Factor Analysis

<b>1. Principle Component Analysis</b><br/>
 1. Standardize the range of continuous variables
 2. Compute the covariance matrix to dentify correlations
 3. Compute the eigenvactors and eigenvalues of the covariance matrix
 4. Create feature vector to decide which principal components to keep
<br/><br/><br/>
 - Each component that is formed has a ratio of how much of the data is explained
 - The first component has the highest ratio and the last component has the lowest ratio
 - Normally, components that have a cumulative ratio of around 80% are chosen for the model
<br/>
<pre>
Code Example : 
    #Standardize dataset
    train.scale <- scale(train[, -1:-2])
    
    #Create princomp values on dataset
    pca <- principal(train.scale, rotate="none")   
    
    #plot dataset for eigenvalues(how many components to keep)
    plot(pca$values, type="b", ylab="Eigenvalues", xlab="Component")

    #Rotating pca for greater ratio of variance explained
    pca.non.rotate <- principal(train.scale, nfactors = 5, rotate = "none")
    pca.rotate <- principal(train.scale, nfactors = 5, rotate = "varimax")   #(varimax / quartimax / equimax)
    pca.scores <- data.frame(pca.rotate$scores)
    
    #Add dependent variable on pca dataset
    pca.scores$ppg <- train$ppg
</pre>
<br/>


<b>2. Factor Analysis</b><br/>
 1. Unlike Principal Component analysis, factor analysis has no rank of importance bewteen the factors
 2. Factors form a linear connection between original components.
 3. Choose factors where eigenvalue is greater than one

<br/>
<pre>
Code Example : 
    #Call Data
    library(ade4)
    data(olympic)

    #Seleciton of how many factors to use
    library(psych)
    fa.parallel(olympic$tab,fm="ml",     #ml=maximum likelihood
                fa="fa", n.iter=100)

    eigen(cor(olympic$tab))              #$values must be greater than one

    #Create factoranalysis model
    fa <- factanal(olympic$tab, 
                factors = 2, 
                scores = "regression")   #regression option must be used for score output

    #Plot model
    factor.plot(fa, labels = colnames(olympic$tab),
                pos=4, title = "Factor Plot")
    
    #Plot model as heatmap
    library(gplots)
    library(RColorBrewer)
    heatmap.2(abs(fa$loadings), col = brewer.pal(9, "Blues"),
            trace="none",key=FALSE, dendrogram="none",
            cexCol = 1.2 , main = "Factor Loadings")

</pre>
<br/>