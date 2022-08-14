---
layout: splash
title: "Decision Tree"
categories: R
tag: coding
---
# Decision Tree
<img src="\assets\images\DecisionTree\Diagram.png" alt="Alt text">
 - A basic method which divides the data for each node until all datasets are classified
 - Tree must be pruned to prevent overfitting


<pre>
Code Example : 
    Regression Tree : 
        Import Data :     
            data(prostate)
            prostate$gleason <- ifelse(prostate$gleason == 6, 0, 1)

            pros.train <- subset(prostate, train ==  TRUE)[, 1:9]
            pros.test = subset(prostate, train == FALSE)[, 1:9]

        Create Model : 
            set.seed(123)
            tree.pros <- rpart::rpart(lpsa ~ ., data = pros.train)
            
            #Prune Trees to prevent overfitting
            tree.pros$cptable
            plotcp(tree.pros)
            cp <- tree.pros$cptable[5,1]
            prune.tree.pros <- prune(tree.pros, cp = cp)

            plot(partykit::as.party(tree.pros))
            plot(partykit::as.party(prune.tree.pros))
            party.pros.test <- predict(prune.tree.pros, 
                                    newdata = pros.test)
            rpart.resid <- party.pros.test - pros.test$lpsa #calculate residual
            mean(rpart.resid^2)  


    Classification Tree : 
        Import Data : 
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
            tree.biop <- rpart(class ~ ., data = biop.train)
            
            #Prune tree to prevent overfitting
            tree.biop$cptable
            cp <- min(tree.biop$cptable[3,1])
            prune.tree.biop = prune(tree.biop, cp <- cp)
            plot(as.party(prune.tree.biop))
            rparty.test <- predict(prune.tree.biop, newdata = biop.test,
                                type = "class")
            table(rparty.test, biop.test$class)
            (136+64)/209



</pre>