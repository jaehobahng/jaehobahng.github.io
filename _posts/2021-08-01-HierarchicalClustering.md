---
layout: splash
title: "Hierarchical Clustering"
categories: R
tag: coding
---
# Hierarchical Clustering
<img src="\assets\images\HierarchicalClustering\Dendrogram.png" alt="Alt text">
 - Connecting datapoints using different distances and methods to form a dendogram
 - Choosing the right amount of clusters can be chosen after the model is complete
 - Model is slow when dataset is large
 - Sensitive to outliers
<br/>
 - Distances : 
     - Nominal : Euclidean, Manhattan, Minkowski, Statistical, Mahalanobis
     - Factoral : Jaccard, Soremsem-Dicd,Anderberg,Ochiai,Simple Matching, Rogers and Tanimoto
 - Methods : Single, Complete, Average, Centroid, Ward


<pre>
Code Example : 
library(cluster)               
library(compareGroups)         
library(HDclassif)             
library(NbClust)               
library(sparcl)                

    Import Data : 
        data(wine)
        str(wine)
        names(wine) <- c("Class", "Alcohol", "MalicAcid", "Ash", "Alk_ash",
                        "magnesium", "T_phenols", "Flavanoids", "Non_flav",
                        "Proantho", "C_Intensity", "Hue", "OD280_315", "Proline")
        names(wine)
        df <- as.data.frame(scale(wine[, -1])) #평균0, 표준편차 1
        summary(df)
        table(wine$Class)

    Complete Link Example : 
        Search for Optimal cluster number : 
            numComplete <- NbClust::NbClust(df, distance = "euclidean",
                                min.nc = 2, max.nc = 6,
                                method = "complete",
                                index = "all")
            
            numComplete$Best.nc  # To show which index recommended what number of clusters
            
        Create Model : 
            dis <- dist(df, method = "euclidean")
            hc <- hclust(dis, method = "complete")
            comp3 <- cutree(hc, 3) 
            
        Visualize Model : 
            plot(hc, 
                hang = -1,      #To match the graph to the bottom
                labels = FALSE, 
                main = "Complete-Linkage")

            ColorDendrogram(hc,                 
                            y = comp3,          
                            main = "Complete",  
                            branchlength = 100)  
            
        Evaluate model(In case of supervised learning) : 
            table(comp3)
            table(comp3, wine$Class)   
            (51+50+48)/178
        
        
        
        
    Ward Link Example :     
        Search for Optimal cluster number : 
            numWard <- NbClust(df,                    
                            diss = NULL,              #dissimilarity(default = NULL)
                            distance = "euclidean",   
                            min.nc = 2, 
                            max.nc = 6, 
                            method= "ward.D2",        
                            index = "all")
            
        Create Model : 
            dis <- dist(df, method = "euclidean")
            hcWard <- hclust(dis, method = "ward.D2")    
            ward3 <- cutree(hcWard, 3)
            plot(hcWard, hang = -1, labels = FALSE, main = "Ward's-Linkage")
        
        Visualize Model : 
            ColorDendrogram(hcWard,                 
                            y = ward3,              
                            main = "Complete",      # Title of Chart
                            branchlength = 100)     # Length of Branches


</pre>