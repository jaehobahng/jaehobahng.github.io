---
layout: splash
title: "Statistical Hypothesis Test"
categories: R
tag: coding
---

# Statistical hypothesis test
## T-Test<br/>

<b>1. One Sample T-Test</b><br/>
1. Check data for normality
2. When normality is proven = T-Test<br/>When normality is not proven = Wilcox test
<br/>
<pre>
Code Example(Normality check) : 
    shapiro.test(iris$Sepal.Length)

Code Example(T-Test) : 
    t.test(iris$Sepal.Length,alternative = c("two.sided","less","greater"), mu=3.05)

Code Example(Wilcox Test) : 
wilcox.test(A,mu=m)
 - Null Hypothesis : Population mean = m / Alternative Hypothesis : Population mean ≠ m
wilcox.test(A,mu=m,alternative="less")
 - Null Hypothesis : Population mean = m / Alternative Hypothesis : Population mean < m
wilcox.test(A,mu=m,alternative="greater")
 - Null Hypothesis : Population mean = m / Alternative Hypothesis : Population mean > m

</pre>

<br/>

<b>2. Independent two sample t-test</b><br/>
1. Null Hypothesis = difference in average is equal to 0
2. There must be only 2 nominal properties matched to a quantitative data
3. The number data for the two nominal properties do not have to be the same
4. Wilcox test will be used if a non-parametric test must be used

<br/>
<pre>
Code Example(T-Test) : 
    t.test(Bwt~Sex, data=cats, alternative="two.sided",var.equal=TRUE)
    t.test(Bwt~Sex, data=cats, alternative="two.sided",var.equal=FALSE)

Code Example(Wilcox test) : 
    wilcox.test(x,y)

</pre>
<br/>

<b>3. Paired T-Test</b><br/>
1. Used in situations usually when a certain before and after affect must be measured on one dataset
2. Null Hypothesis = difference in average is equal to 0
3. There must be only 2 nominal properties matched to a quantitative data
4. There must be an equal amount of datasets for the 2 nominal properties(before and after)
5. Wilcox test will be used if a non-parametric test must be used

<br/>
<pre>
Code Example(T-Test) : 
    t.test(iris$Sepal.Length,iris$Sepal.Width, alternative = "two.sided", paired = TRUE)

Code Example(Wilcox test) : 
    wilcox.test(x, y, paired = TRUE, alternative = "greater")

</pre>
<br/>


## ANOVA(Analysis of Variance<br/>

<b>1. One-Way ANOVA</b><br/>
1. Homogeneity of variance test
2. ANOVA(whether there is a difference between the groups)
3. Post-hoc test(between what groups the difference exists)


<br/>
<pre>
Code Example(Homogeneity of variance test) : 
    car::leveneTest(score ~ group, 
            data=owa.df)
    car::bartlett.test(score ~ group, 
                data=owa.df)
    (leveneTest is more commonly used because the barlett test is sensitive to normality)
    (if p<0.05, variance is not homogenous)



Code Example(ANOVA test) : 
(if p<0.05, there is a difference between the groups)
When variance is homogenous : 
    owa.result <- aov(score ~ group, 
                    data=owa.df) 
    summary(owa.result)

When variance is not homogenous : 
    oneway.test(owa.df$score ~ owa.df$group, 
                data=owa.df, 
                var.equal = FALSE)
    library(nparcomp)
    result = mctp(owa.df$score ~ owa.df$group, 
                data=owa.df)
summary(result)

Code Example(Post-hoc test) : 
Method1 : Bonferroni(Heteroscedasticity of variance)
    pairwise.t.test(owa.df$score, 
                    owa.df$group, 
                    data=owa.df, 
                    p.adj="bonf")

Method2 : TukeyHSD(Homogeneity of variance)
    tukeyPlot <- TukeyHSD(owa.result) 
    plot(tukeyPlot)


</pre>
<br/>

<b>2. Two-Way ANOVA</b><br/>
Example <br/>
Cooking in two different temperatures(200,300) with two differnt methods(Oven, Fried)

1. First, you must check if there is an interaction effect bewteen the two factors.
<img src="\assets\images\StatisticalTest\Interaction.jpg" alt="Alt text"><br/>
Source : https://www.researchgate.net/figure/Patterns-of-two-way-interaction_fig1_324946870
2. If there are no interactions, the visual lines will appear parallel
3. Perform Anova Test
4. Perform Post-hoc test

<br/>
<pre>
Code Example(Anova Test) : 
    twa.result <- aov(taste ~ meth + temp + meth:temp, data=twa.df) 
    summary(twa.result)
    (meth:temp means there is an interaction effect)


Code Example(Post-hoc test) : 
Method 1 : t.test for each property
    tw1 <- twa.df[twa.df$meth=="Oven",]
    tw2 <- twa.df[twa.df$meth=="Oil",]

    t.test(taste ~ temp, 
        data=tw1, 
        alternative = c("two.sided"), 
        var.equal = TRUE, 
        conf.level = 0.95)
    t.test(taste ~ temp, 
        data=tw2, 
        alternative = c("two.sided"), 
        var.equal = TRUE, 
        conf.level = 0.95)

Method 2 : Tukey Test
    TukeyHSD(aov(taste ~ meth + temp + meth:temp, data=twa.df) )
    tukeyPlot <- TukeyHSD(aov(taste ~ meth + temp + meth:temp, data=twa.df) )
    plot(tukeyPlot)
</pre>
<br/>

<b>2. Repeated Measures ANOVA</b><br/>
Example <br/>
A comparison between an experimental group and a control group where one group is given aroma therapy.<br/>
A before and after is measured for each group

1. Sphercity test
2. Anova test
3. Post-hoc test

Dataset example image
<img src="\assets\images\StatisticalTest\Repeated.jpg" alt="Alt text"><br/>
<br/>
<pre>
Code Example(Sphercity test) : 
    twrma.matrix <- cbind(twrma.df$painscore[twrma.df$time=="Before"], 
                        twrma.df$painscore[twrma.df$time=="After"])

    twrma.model.lm <- lm(twrma.matrix ~ 1)
    time.f <- factor(c("Before","After"))
    options(contrasts=c("contr.sum", "contr.poly"))

    twrma.result.mt <- car::Anova(twrma.model.lm, 
                            idata=data.frame(time.f), #Anova 대문자
                            idesign=~time.f, 
                            type="III")
    summary(twrma.result.mt, multivariate=T)
(if p value under Mauchly Test for Sphercity is greater than 0.05, there is a sphercity)
(when there is sphercity, use Univariate Type III, if not, use Greenhouse-Geisser and Huynh-Feldt)

Code Example(ANOVA Test) : 
Method1 : 
    twrma.result <- aov(painscore ~ time*group + Error(id), data=twrma.df)
    summary(twrma.result)
Method2 : 
    twrma.result <- aov(painscore ~ time + group + time:group, data=twrma.df) 
    summary(twrma.result)


Code Example(Post-hoc test) : 
Method 1 : Group comparison based on before and after effect difference
    pre <- twrma.df[twrma.df$time=="Before",]
    post <- twrma.df[twrma.df$time=="After",]

    t.test(painscore ~ group, 
        data = pre, 
        alternative = c("two.sided"), 
        var.equal = TRUE,
        conf.level = 0.95)
    t.test(painscore ~ group, 
        data = post, 
        alternative = c("two.sided"), 
        var.equal = TRUE,
        conf.level = 0.95)


Method 2 : Before and after comparison absed on group difference
    controlG <- twrma.df[twrma.df$group=="Control",]
    treatG <- twrma.df[twrma.df$group=="Experiment",]

    t.test(painscore ~ time, 
        data = controlG, 
        alternative = c("two.sided"), 
        var.equal = TRUE,
        conf.level = 0.95)
    t.test(painscore ~ time, 
        data = treatG, 
        alternative = c("two.sided"), 
        var.equal = TRUE,
        conf.level = 0.95)
</pre>
<br/>





