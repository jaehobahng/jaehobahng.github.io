---
layout: splash
title: "Time Series"
categories: R
tag: coding
---
# What is a stationary time series?
 - A stationary time series is one whose properties do not depend on the time at which the series is observed
     - No seasonality / No trends
     - The average and variance are constant

 - Time series must be proprocessed to a stationary timeseries before analysis


# Time series decomposition
<pre>
# Decomose time series for seasonality, trend, and other remainding data
    decomp <- stl(co2, s.window="periodic")
    

# How to plot decomposed components of a time series
    plot(decomp, col = "red",col.range="blue")

# How to take out seasonality from time series 
    decomp$time.series[,'seasonal']
    plot(co2 - decomp$time.series[,'seasonal']-decomp$time.series[,'trend'])
</pre>

# Autocorrelation
 - A funciton on how the timeseries is related to its past values

1. ACF(Autocorrelation Function)
 - An overall effect on past values to current values(kind of like moving average)
 - If time series is stationary, the acf will converge to 0 very rapidly
 - Unstationary timeseries usually have a high positive value
 - Used to determine MA of the ARIMA fuction

<img src="\assets\images\TimeSeries\ACF(equation).jpg" alt="Alt text">
2. PACF(Partial Autocorrelation Function)
 - A 1:1 effect on a certain point in time to the current values
- Used to determine AR of the ARIMA fuction

<img src="\assets\images\TimeSeries\PACF(equation).jpg" alt="Alt text"><br/>
<img src="\assets\images\TimeSeries\PACF(equation2).jpg" alt="Alt text">



# Moving Average Model
 - NOT the moving average model where past datasets are smoothened to see the trend
<br/>
<img src="\assets\images\TimeSeries\MA(equation).jpg" alt="Alt text"><br/>
 - uses past forecast errors in a regression-like model
 - 

# Autoregressive Model
<img src="\assets\images\TimeSeries\AR(equation).jpg" alt="Alt text"><br/>
 - forecast the variable of interest using a linear combination of past values of the variable
 - 

# ARIMA(p,d,q)
<pre>
Code Example : 
    Visualize Time Series : 
        plot(fpp2::goog200, lwd = 2, xlab = "Day", ylab="Dollars",
            main = "Google Stock Prices")

    Check for stationarity of time series : 
        Method 1 : 
        forecast::Acf(goog200)

        Method 2 : 
        tseries::adf.test(dgoog200)

    If variance is not consistant : 
        lgoog200 <- log(goog200)

    If average is not consistant : 
        1. Find optimal differencing
        forecast::ndiffs(goog200)

        2. Difference time series
        dgoog200 <- diff(goog200,lag=1)

        3. Check for stationarity
        plot(dgoog200)   #Visualize
        acf(dgoog200)    #Use ACF

    Find optimal ARIMA(p,d,q) values : 
        Method 1 : ACF / PACF visualization
        Acf(dgoog200)
        pacf(dgoog200)

        Method 2 : autoarima method
        goog.arima <- arima(dgoog200, order = c(0,0,0))
        auto.arima(dgoog200)
        
    Check for acuracy of model
        accuracy(auto.arima(dgoog200))
        accuracy(arima(dgoog200, order = c(0,0,0)))

    Check for normal distribution of risiduals : 
        qqnorm(goog.arima$residuals,pch=21,col="black",bg="gold")
        qqline(goog.arima$residuals, col = "royalblue", lwd = 2)
        
    Check for autocorrelation of residuals
        Box.test(goog.arima$residuals, type = "Ljung-Box")

    Predict and visualize values
        goog.arima.pred <- forecast::forecast(goog.arima, h=10)
        plot(goog.arima.pred, shadecols = c("mistyrose","salmon"))
</pre>



# SARIMA
 - ARIMA with seasonality
     - Example : month trend in a year time series
<pre>
Code Example : 
    Create model : Two identical models
        auto.arima(gas)
        arima(gas, order = c(2,1,1), 
            seasonal = list(order=c(0,1,1), 
                            period=12))
    Predict values : 
        forecast(gas.arima,
                h=12*5)

    Plot predicted values : 
        plot(forecast(gas.arima,h=60),
            shadecols = c("lavender","skyblue"),
            fcol = "orangered",
            col="darkorange")
  

</pre>



# Moving Average
<pre>
Code Example : 
library(forecast)

Moving Average(None, 5, 10, 15)
    plot(Nile)
    plot(ma(Nile,5))
    plot(ma(Nile,10))
    plot(ma(Nile,15))   

</pre>