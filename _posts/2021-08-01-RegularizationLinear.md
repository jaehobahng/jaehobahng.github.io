---
layout: splash
title: "Regularizaiton Linear Models"
categories: R
tag: coding
---

<img src="\assets\images\RegularizationLinear\RegularizaitonDiagram.png" alt="Alt text">
<img src="\assets\images\RegularizationLinear\RegularizationEquation.png" alt="Alt text">


<pre>
Data Preparation code : 

    library(ElemStatLearn)

    data(prostate)
    str(prostate)
    plot(prostate)
    plot(prostate$gleason, ylab = "Gleason Score")
    table(prostate$gleason)
    boxplot(prostate$lpsa ~ prostate$gleason, xlab = "Gleason Score", 
            ylab = "Log of PSA")

    prostate$gleason <- ifelse(prostate$gleason == 6, 0, 1)
    table(prostate$gleason)

    p.cor = cor(prostate)
    corrplot.mixed(p.cor)

    train <- subset(prostate, train == TRUE)[, 1:9]
    str(train)
    test = subset(prostate, train==FALSE)[,1:9]
    str(test)

    x <- as.matrix(train[,1:8])
    y <- train[,9]
</pre>


# Ridge Regression
<img src="\assets\images\RegularizationLinear\RidgeEquation.png" alt="Alt text">
 - L2-norm regularizaiton
 - Variable selection is impossible
 - There is a closed form solution(can be solved through differentials)
 - Performs well even when there is a high collinearity between variables
 - Has a tendency to reduce larger variables

 - Pitchers can be close to 0 but cannot be 0 and therefore all variables must be used
 - The shrinkage penalty(second term in equation) becomes smaller when β1, . . . , βp is close to 0 and this term allows estimations to be shrinked to 0
 
 - The tuning parameter λ regulates how much the shrinkage penalty effects the overall linear model
 - When λ = 0, the penalty term has no effect → the outcome is identical to the original RSS
 - When λ = infinite, the coefficients become close to zero.

<pre>
Code Example : 

Training Model : 
    library(glmnet)
    ridge <- glmnet(x,y,
                    family="gaussian",   #nominal : gaussian / factorial : binomial
                    alpha=0)             #0 = ridge / 1 = Lasso / 0~1 = Elastic

    print(ridge)
    plot(ridge, xvar='norm', label=TRUE)   
    plot(ridge, xvar='lambda', label=TRUE)    
    plot(ridge, xvar='dev', label=TRUE)    

    ridge.coef <- coef(ridge, s=0.1, exact = FALSE)

Testing Model : 
    newx <- as.matrix(test[,1:8])
    ridge.y <- predict(ridge,
                    newx = newx,
                    type = "response", 
                    s = 0.1)

    plot(ridge.y,test$lpsa,
        xlab = "predicted",
        ylab = "Actual",
        main = "Ridge Regression")

    ridge.resid <- ridge.y-test$lpsa  
    mse <- mean(ridge.resid^2)
    r2 <- sum((test$lpsa-ridge.y)^2) / sum((test$lpsa-mean(test$lpsa))^2)
    mae <- mean(abs(test$lpsa-ridge.y))


</pre>


# Lasso Regression
<img src="\assets\images\RegularizationLinear\LassoEquation.png" alt="Alt text">
 - L1-norm regularizaiton
 - Variable seleciton is possible
 - There is no closed form solution(solved through numerical optimization)
 - Performs relatively worse than ridge regression when there is a high collinearity between variables

 - There is a higher chance the optimized output will appear in the corner → meaningless variables will be close to 0
 - The same regularization is applied regardless of the size of the parameter → small parameters become 0 excluding variables from the model(makes model simple and easier to interpret)

<br/>
<pre>
Code Example : 

Training Model : 
    library(glmnet)
    lasso<-glmnet(x,y,
                family = "gaussian",
                alpha = 1)    
    print(lasso)

    plot(lasso, xvar='norm', label=TRUE)   #variable / coefficient based on L1
    plot(lasso, xvar='lambda', label=TRUE) #variable / coefficient based on Lambda
    plot(lasso, xvar='dev', label=TRUE)   

    lasso.coef <- coef(lasso, s = 0.045)

Testing Model : 
    lasso.y <- predict(lasso, 
                    newx = newx, 
                    type = "response",      #response,coefficients,nonzero,class
                    s = 0.045) 

    plot(lasso.y, test$lpsa)
    lasso.resid <- lasso.y - test$lpsa
    mean(lasso.resid^2)
</pre>
<br/>


# Elastic net
<img src="\assets\images\RegularizationLinear\ElasticDiagram.png" alt="Alt text">
<img src="\assets\images\RegularizationLinear\ElasticEquation.png" alt="Alt text">
Image source : https://towardsdatascience.com/from-linear-regression-to-ridge-regression-the-lasso-and-the-elastic-net-4eaecaf5f7e6
<br/>
 - Works well for big data sets
 - Has strengths fo both ridge and lasso models : reducing variables and minimzing variances are both possible
 - Uses both L1 and L2 norms and regulates λ1 and λ2 (λ1 + λ2 <= 1)

<br/>
<pre>
Code Example : 

Find best alpha lambda for model : 
    grid <- expand.grid(.alpha = seq(0,1,0.2),    
                        .lambda = seq(0,0.2,0.02))
    table(grid)

    library(caret)
    control <- trainControl(method = "LOOCV")

    enet.train <- train(lpsa ~.,
                        data = train,
                        method = "glmnet",
                        trControl = trainControl(method="LOOCV"),
                        tuneGrid = grid)
    min(enet.train$result[,3])
    subset(x=enet.train$result
        ,enet.train$result[,3] == min(enet.train$result[,3]))

Train and evaluate model : 
    enet <- glmnet(x,y,family = "gaussian", alpha = 0, lambda = 0.08)
    enet.coef <- coef(enet, s = 0.08)
    enet.coef

    enet.y <- predict(enet, 
                    newx = newx, 
                    type = "response",  
                    s = 0.08)

    plot(enet.y, test$lpsa,
        xlab = "Predicted", ylab = "Actual")

    enet.resid <- enet.y - test$lpsa
    mean(enet.resid^2)
</pre>
<br/>