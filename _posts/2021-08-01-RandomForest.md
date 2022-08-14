---
layout: splash
title: "Random Forest"
categories: R
tag: coding
---
# Random Forest
<img src="\assets\images\RandomForest\RandomForestDiagram.png" alt="Alt text">
 - A method to improve overfitting problem for decision trees
 - Bagging : dataset is divided into different partitions to 
 - Features are also randomized
 - For each set of dataset that is partitioned by the bagging method, different features are used for the decision tree used to for the model
 - Both dataset and features are randomized, and results from all trees are combined to reach a final result.
 - OOB : out of bag data is used to test accuracy of model
 - Both regression and classification datasets are usable for the model

<pre>
Code Example : 

Regression : 
    Import Dataset : 
        library(ElemStatLearn)
        data(prostate)
        prostate$gleason <- ifelse(prostate$gleason == 6, 0, 1)
        pros.train <- subset(prostate, train ==  TRUE)[, 1:9]
        pros.test = subset(prostate, train == FALSE)[, 1:9]
    
    Create Model : 
            set.seed(123)
            rf.pros <- randomForest(lpsa ~ ., data = pros.train)
            rf.pros
            plot(rf.pros)
        
        #Choose number of trees where mse is minimized
            which.min(rf.pros$mse)   
        
        set.seed(123)
            rf.pros.2 <- randomForest(lpsa ~ ., data = pros.train, ntree = 75)
            rf.pros.2
        
        #Visualize for importance of variables
            randomForest::varImpPlot(rf.pros.2, scale = TRUE,
                    main = "Variable Importance Plot - PSA Score")
            importance(rf.pros.2)
        
        #Predict and evaluate model with test dataset
            rf.pros.test <- predict(rf.pros.2, newdata = pros.test)
            plot(rf.pros.test, pros.test$lpsa)
            rf.resid <- rf.pros.test - pros.test$lpsa #calculate residual
            mean(rf.resid^2)


Classification : 
    Import Dataset : 
        data(biopsy)
        biopsy <- biopsy[, -1]
        names(biopsy) <- c("thick", "u.size", "u.shape", "adhsn", "s.size", "nucl", "chrom", "n.nuc", "mit", "class")
        biopsy.v2 <- na.omit(biopsy)
        set.seed(123) #random number generator
        ind <- sample(2, nrow(biopsy.v2), replace = TRUE, prob = c(0.7, 0.3))
        biop.train <- biopsy.v2[ind == 1, ] #the training data set
        biop.test <- biopsy.v2[ind == 2, ] #the test data set
        str(biop.test)

    Create Model : 
        set.seed(123)
        rf.biop <- randomForest(class ~ ., data = biop.train)
        rf.biop
        plot(rf.biop)

        #Choose number of trees where mse is minimized
            which.min(rf.biop$err.rate[, 1])
            set.seed(123)
            rf.biop.2 <- randomForest(class ~ ., data = biop.train, ntree = 19)
            rf.biop.2
        
        #Predict and evaluate model with test dataset
            rf.biop.test <- predict(rf.biop.2, 
                                    newdata = biop.test, 
                                    type = "response")
            table(rf.biop.test, biop.test$class)
            (139 + 67) / 209

        #Visualize important features
            varImpPlot(rf.biop.2)




</pre>