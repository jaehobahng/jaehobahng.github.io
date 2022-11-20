---
layout: splash
title: "Apriori"
categories: R
tag: coding
---
# Apriori

#### Indicators
 - Support : How often do the two items occur together
     - P(A ∩ B)

 - Confidence : Indicator on how strong the connection is bewteen the two items<br/>
    Compared to when item A is purchased alone, how often is A and B purchased together
     - P(B|A) = P(A ∩ B) / P(A) = Support / P(A)
 
 - Lift : Whether the correlation is valid and to what direction the correlation is occurring
     - P(B|A) / P(B) = P(A ∩ B) / P(A) P(B) = Confidence / P(B)
     - Lift = 1 : Two items are independent
     - Lift > 1 : Two items have a positive correlation
     - Lift < 1 : Two items have a negative correlation

#### Apriori Steps
1. Decide on a minimum support number
2. Create a Candidate Itemset
3. Choose only from the candidate itemset transactions with a support indicator larger than the minimum support number

<pre>
Code Example : 
library(arules)
library(arulesViz)

    Import Data : 
        data(Groceries)    # Must be transactions data

    Visualize Transactions for frequent pattern : 
        itemFrequencyPlot(Groceries, topN = 10,type = "absolute")
        itemFrequencyPlot(Groceries, topN = 15)

    Create Model : 
        rules <- apriori(Groceries, 
                        parameter = list(supp = 0.001       #Minimum support
                                        , conf = 0.9        #Minimum confidence
                                        , maxlen = 4))      #Maximum number of items
        
    Inspect model : 
        rules <- sort(rules, by = "lift", decreasing = TRUE)
        inspect(rules[1:5])

        rules <- sort(rules, by = "confidence", decreasing = TRUE)
        inspect(rules[1:5])

    Delete Repetitive rules(Prune rules) : 
        subset.matrix = is.subset(rules, rules,sparse=FALSE)
        subset.matrix[lower.tri(subset.matrix, diag=TRUE)] = NA
        redundant <- colSums(subset.matrix, na.rm=TRUE) >= 1
        rules.pruned = rules[!redundant]
        rules.pruned

    Inspect with right hand side rules fixed : 
        beer.rules <- apriori(data = Groceries, 
                            parameter = list(support = 0.0015, confidence = 0.3),
                            appearance = list(default = "lhs", rhs = "bottled beer"))
        beer.rules <- sort(beer.rules, decreasing = TRUE, by = "lift")
        inspect(beer.rules)
        
    Inspect with left hand side rules fixed : 
        bottled.rules <- apriori(data=Groceries, 
                                parameter=list(supp=0.001,conf = 0.2),
                                appearance = list(default="rhs",lhs="bottled beer"),
                                control = list(verbose=FALSE))
        bottled.rules<-sort(bottled.rules, decreasing=TRUE,by="confidence")
        inspect(bottled.rules)
</pre>


#### How to change different data forms to transaction forms

##### matrix → transaction
<pre>
mtx <- matrix(c(1,1,1,0,0,
                1,1,1,1,0,
                0,0,1,1,0,
                0,1,0,1,1,
                0,0,0,1,0), ncol=5, byrow=T)

rownames(mtx) <- paste0("ti",1:5)
colnames(mtx) <- letters[1:5]

mtx.trans <- as(mtx, 'transactions')
inspect(mtx.trans)
</pre>
##### dataframe → transaction
<pre>
df <- as.data.frame(mtx)
df <- as.data.frame(sapply(df,as.logical))

df.trans <- as(df,'transactions')
</pre>
##### list → transaction
<pre>
list <- list(tr1=c("a","b","c"),
             tr2=c("a","d"),
             tr3=c("b","e"),
             tr4=c("a","d","e"),
             tr5=c("b","c","d"))


list.trans <- as(list,'transactions') 
list.trans
summary(list.trans)
</pre>