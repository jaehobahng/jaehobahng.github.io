---
layout: splash
title: "Linear Regression"
categories: R
tag: coding
---

# How to create a model

<b>1. Principle Component Analysis</b><br/>

<br/>
<pre>
Code Example : 

    #Call data
    iris.df <- iris[,c(1,3)]


    #Create random model
    lm.iris.df <- lm(Sepal.Length ~ Petal.Length + I(Petal.Length^2), data = iris.df)

    #Predict outcome through created model
    plot.iris <- as.data.frame(predict(lm.iris.df,iris.df))

</pre>
<br/>

# How to optimize variables
<b>1. Step Method</b><br/>
<pre>
Code Example : 
    
    #What the step method consists of
    step(model, scope, direction, k)

    #Create  a step model where the lower limit is no variable and upper limit is two variables with Sepal Length and width
    #Direction both means variables can be either added or subtracted each time the model is iterated for a more accurate model
    #Summary function of the iris.step model allows us to see which model had the best accurace(check AIC)
    iris.lm <- lm(Petal.Width ~ 1, iris)
    iris.step <- step(iris.lm, 
                    scope = list(lower=~1, upper=~Sepal.Length + Sepal.Width), 
                    direction = "both")
    summary(iris.step)

</pre>
<br/>

# How to plot model
<pre>
Code Example : 
    
    #Split screen for 6 graphs
    par(mforw = c(2,3)) 

    #show six different graphs
    plot.lm(Cars.lm, which=c(1:6))
        1. Residuals vs Fitted
        2. Normal Q-Q
        3. Scale-Location
        4. Cook's distance
        5. Residuals vs Leverage
        6. Cook's dist vs Leverage
</pre>


# Other tests for the model
1. Multicollinearity
 - If the Variance inflation factor is greater than 5 or at most ten, multicollinearity exists
 - VIF = 1/(1-R2i)    R2i = when I variable is linear regressioned by the other variables
 - Find the variable where multicollinearity exists. Take the variable out of the model and if indicators like the R2 values do not change, take the variable out and adjust your model.
<pre>
Code Example : 
vif(model)
</pre>
<br/>

2. Homoskedascity of errors
 - if p<0.05, errors of the model do not have homoskedascity
<pre>
Code Example : 
    lmtest::bptest(model)
</pre>


# How to calculate MSE(Mean Square Error)(evaluation of model)
<pre>
Code Example : 
    resid.subfit <- actual value - predicted value
    mean(resid.subfit^2)
</pre>