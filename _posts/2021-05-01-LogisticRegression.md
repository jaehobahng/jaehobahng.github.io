---
layout: splash
title: "Logistic Regression"
categories: R
tag: coding
---
# Logistic Regression
<img src="\assets\images\LogisticRegression\Diagram.jpg" alt="Alt text">
 - Commonly used for binary classification
 - Coefficients : 
     - Continuous variable : The number by which the log of odds increases for every variable unit
     - Discrete variable : log(odds ratio) that tells you how much having a certain trait increases or decreases the odds of being the dependent variable
 - p values must be statistically significant(<0.05) for the variables to be valid
 - The variable coefficients and y intercept are considered the slope and y intercept of the log(odds) value, and these values can be transformed into the probability of the logistic regression
 - Maximum Likelihood
     - The optimized logistic regression model is chosen when the likelihood is maximized
     - Likelihood = the multiplication of all probabilities of the data point be
         - When data point is considered class A : probability of being class A
         - When data point is considered class B : probability of being class B

<pre>
Code Example : 
library(MASS)
library(reshape2)
library(ggplot2)
library(corrplot)
library(InformationValue)

    Import Data : 
        data(biopsy)
        str(biopsy)
        head(biopsy)

        biopsy$ID = NULL
        names(biopsy) = c("thick", "u.size", "u.shape", "adhsn", 
                        "s.size", "nucl", "chrom", "n.nuc", "mit", "class")
        names(biopsy)
        biopsy.v2 <- na.omit(biopsy)
        y <- ifelse(biopsy.v2$class == "malignant", 1, 0)
        biop.m <- reshape2::melt(biopsy.v2, id.var = "class")

    Visualize Data for EDA : 
        ggplot(data = biop.m, aes(x = class, y = value)) + 
        geom_boxplot() +
        facet_wrap(~variable, ncol = 3)

        bc <- cor(biopsy.v2[ ,1:9]) #create an object of the features
        corrplot::corrplot.mixed(bc)

    Train/Test Data Split : 
        set.seed(123) #random number generator
        ind <- sample(2, 
                    nrow(biopsy.v2), 
                    replace = TRUE, 
                    prob = c(0.7, 0.3))
        train <- biopsy.v2[ind==1, ] #the training data set
        test <- biopsy.v2[ind==2, ] #the test data set
        str(test) #confirm it worked
        prop.table(table(train$class))
        prop.table(table(test$class))

    Create basic model : 
        full.fit <- glm(class ~ ., family = binomial, data = train)

        summary(full.fit)
        confint(full.fit)
        exp(coef(full.fit))   
        library(car)
        vif(full.fit) #Check for collinearity

        trainY <- y[ind==1]
        testY <- y[ind==2]

    Evaluate Train / Test dataset with model : 
        train.probs <- predict(full.fit, type = "response")
        InformationValue::confusionMatrix(trainY, train.probs)
        misClassError(trainY, train.probs)

        test.probs <- predict(full.fit, newdata = test, type = "response")
        misClassError(testY, test.probs)
        confusionMatrix(testY, test.probs)

    Selecting Optimal variable : 
        X <- train[, 1:9]
        Xy <- data.frame(cbind(X, trainY))

        Cross Validation : 
            bestglm::bestglm(Xy = Xy, 
                    IC = "CV",              #Information Criteria(AIC, BIC, BICg, BICq, LOOCV, CV)
                    CVArgs = list(Method = "HTF", K = 10, REP = 1), 
                    family=binomial)

            reduce.fit <- glm(class ~ thick + u.size + nucl, family = binomial, data = train)

            test.cv.probs = predict(reduce.fit, newdata = test, type = "response")
            misClassError(testY, test.cv.probs)
            confusionMatrix(testY, test.cv.probs)

        BIC : 
            bestglm(Xy = Xy, IC = "BIC", family = binomial)
            bic.fit <- glm(class ~ thick + adhsn + nucl + n.nuc, 
                        family = binomial, data = train)
            test.bic.probs = predict(bic.fit, newdata = test, type = "response")
            misClassError(testY, test.bic.probs)
            confusionMatrix(testY, test.bic.probs)





</pre>