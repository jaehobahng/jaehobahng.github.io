---
layout: splash
title: "XGBoost"
categories: Kaggle R
tag: coding
---
<pre>
library(dplyr)
library(ggplot2)
library(string)
library(caret)
library(mice)
library(RandomForest)
library(Boruta)

train <- read.csv("train.csv")
test <- read.csv("test.csv")

str(train)
str(test)

###TEST TRAIN SET BIND###
test$Transported <- NA
full <- bind_rows(train,test,.id='id')
full$id <- ifelse(full$id=='1','train','test')

str(full)




###CHANGE BLANK CELLS TO NA###
colSums(is.na(full))
full[full=='' | full==' '] <- NA
colSums(is.na(full))





###COLUMN FORMATING###
str(full)

#1. CryoSleep : True,False to 1,0
full$CryoSleep <- ifelse(full$CryoSleep=='True',1,ifelse(full$CryoSleep=='False',0,NA))

#2. VIP : True,False to 1,0
full$VIP <- ifelse(full$VIP=='True',1,ifelse(full$VIP=='False',0,NA))


#3. Transported to factors
# full$Transported <- ifelse(full$Transported=='False',FALSE,ifelse(full$Transported=='True',TRUE,NA))
full$Transported <- factor(full$Transported,levels=c('True','False'))
levels(full$Transported)
str(full)







###Feature Engineering###
#Separate First/Last name and drop Name Column
library(stringr)
full$Lastname <- str_split(full$Name,' ',simplify=TRUE)[,2]
unique(full$Lastname)
full$Name <- NULL

#Cabin Separation(Deck/Num/Side) and drop original Cabin Column
full$Cabin
full$Deck_Cabin <- str_split(full$Cabin,'/',simplify=TRUE)[,1]
full$Num_Cabin <- str_split(full$Cabin,'/',simplify=TRUE)[,2]
full$Side_Cabin <- str_split(full$Cabin,'/',simplify=TRUE)[,3]
full$Cabin <- NULL
str(full)






###Replace Missing Values###
#Check Missing Values
colSums(is.na(full))

#1. Substitude missing fee values to 0
str(full)
fee <- c('RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck')
for(i in fee){
  full[is.na(full[,i]),i] <- 0
}

#2. Create column for sum of all fees used by customer
full$fee <- rowSums(full[,fee])



#3. People in CryoSleep(TRUE) do not use any money
full[full$fee>0&full$CryoSleep=='True'&!is.na(full$CryoSleep),c('CryoSleep','fee')]
full[(full$fee>0&is.na(full$CryoSleep)),c('CryoSleep')] <- 0
sum((is.na(full$CryoSleep)))
str(full)





#4. Use Mice to replace columns with integers(CryoSleep, Age, VIP)
library(mice)
colSums(is.na(full))
mice_mod <- mice(full[,names(full) %in% c('Age','fee','VIP','CryoSleep','Destination','HomePlanet','Deck_Cabin','CryoSleep')],
                 method = 'rf')

#Check if there is an abnormal difference in replaced and original Values
par(mfrow=c(3,2))
hist(complete(mice_mod)$Age,main = 'Mice Age')
hist(full$Age, main= 'Original Age')
hist(complete(mice_mod)$CryoSleep,main = 'Mice Age')
hist(full$CryoSleep, main= 'Original Age')
hist(complete(mice_mod)$VIP,main = 'Mice Age')
hist(full$VIP, main= 'Original Age')


full$Age <- complete(mice_mod)$Age
full$CryoSleep <- complete(mice_mod)$CryoSleep
full$VIP <- complete(mice_mod)$VIP

colSums(is.na(full))




#5. Replace HomePlanet missing values based on LastName
  #Assume that people with the same last name should be from the same HomePlanet
  #Replace missing HomePlanet values with the HomePlanet values of other idential LastName passengers
str(full)
#Check that there are passengers who have the same Lastname but one is missing a HomePlanet
print(full %>% group_by(Lastname,HomePlanet) %>% summarise(count = n()) %>% arrange(Lastname,desc(count)),n=100)

#Table to find what unique values Last Name Members have
vlookup <- full %>% filter(!is.na(Lastname)&!is.na(HomePlanet)) %>% dplyr::select(Lastname,HomePlanet) %>% distinct(Lastname,HomePlanet) %>% arrange(Lastname)
#(Check if the same Lastname can have multiple HomePlanets)(Blank has all the values so choose the most common value(earth))
vlookup %>% group_by(Lastname) %>% summarise(count=n()) %>% arrange(desc(count))
vlookup <- vlookup %>% 
  filter(!row_number() %in% c(1,3))


#Passengers who do not have a HomePlanet value
na.HomePlanet <- full %>% filter(is.na(HomePlanet)) %>% dplyr::select(HomePlanet,Lastname)

#Join NA values with HomePlanet of identical Lastname Passengers
na.vlookup <- na.HomePlanet %>% left_join(vlookup,by='Lastname')

#Replace Table value with values found above
na.vlookup$HomePlanet.y
full[is.na(full$HomePlanet),c('HomePlanet','Lastname')]$HomePlanet <- na.vlookup$HomePlanet.y

print(full %>% group_by(Lastname,HomePlanet) %>% summarise(count=n()),n=100)

#Check remaining missing Values for Homeplanet and replace them based on average fee used by passengers from each HomePlanet
full[is.na(full$HomePlanet),c('fee','HomePlanet')]
full %>% group_by(HomePlanet) %>% summarize(mean(fee))
#REPLACEMENT
full[is.na(full$HomePlanet),c('fee','HomePlanet')]$HomePlanet <- c('Earth','Earth','Mars','Europa','Europa','Earth','Earth')

colSums(is.na(full))






#Could not find valid correlation between columns to replace Destination and Cabin
#replace all NA values to 0
colSums(is.na(full))
str(full)

full[is.na(full$Destination),'Destination'] <- '0'
full[is.na(full$Deck_Cabin),'Deck_Cabin'] <- '0'
colSums(is.na(full))




###EDA###
str(full)
summary(full)

#Yes Relation(factor)
ggplot(na.omit(full),aes(x=CryoSleep,fill=Transported)) + geom_bar(position='fill')
ggplot(na.omit(full),aes(x=Deck_Cabin,fill=Transported)) + geom_bar(position='fill')
ggplot(na.omit(full),aes(x=HomePlanet,fill=Transported)) + geom_bar(position='fill')

#Negative Relation(continuous variable)
ggplot(na.omit(full),aes(x=fee,fill=Transported)) + geom_histogram(position='dodge')
ggplot(na.omit(full),aes(x=VRDeck,fill=Transported)) + geom_histogram(position='fill')
ggplot(na.omit(full),aes(x=Spa,fill=Transported)) + geom_histogram(position='fill')
ggplot(na.omit(full),aes(x=RoomService,fill=Transported)) + geom_histogram(position='fill')
ggplot(na.omit(full),aes(x=Age,fill=Transported)) + geom_histogram(position='fill')

#Positive Relation(continuous variable)
ggplot(na.omit(full),aes(x=FoodCourt,fill=Transported)) + geom_histogram(position='fill')

#No relation
ggplot(na.omit(full),aes(x=VIP,fill=Transported)) + geom_bar(,position='fill')
ggplot(na.omit(full),aes(x=Destination,fill=Transported)) + geom_bar(position='fill')
ggplot(na.omit(full),aes(x=Side_Cabin,fill=Transported)) + geom_bar(position='fill')



###Construct dataframe for RandomForest modeling###
str(full)

#Create train, test partition and remvoe PassengerID, Last name, id
full.model.train <- full[full$id=='train',!colnames(full) %in% c('id','PassengerId','Lastname')]
full.model.test <- full[full$id=='test',!colnames(full) %in% c('id','PassengerId','Lastname')]




#####RandomForest####

#Use Boruta to find which Columns can be removed for the model
library(Boruta)
set.seed(123)
feature.selection <- Boruta(Transported~., data = full.model.train, doTrace = 1)
table(feature.selection$finalDecision)
names(feature.selection$finalDecision)
fNames <- getSelectedAttributes(feature.selection) #withTentative = TRUE

#Create train/test data set with only necessary features and add Transported as dependent variable
full.model.train.features <- full.model.train[,fNames]
full.model.train.features$Transported <-full.model.train$Transported

full.model.test.features <- full.model.test[,fNames]
full.model.test.features$Transported <-full.model.test$Transported

str(full.model.train.features)


#Creat model
library(randomForest)
set.seed(123)
rf <- randomForest(Transported~., data = full.model.train.features, ntree = 200)

plot(rf)
legend('top',colnames(rf$err.rate),fill=1:3)

#Tree numer with minimum error rate
which.min(rf$err.rate[,1])

#Tune ntree parameter and evaluate model with train data set
set.seed(321)
rf2 <- randomForest(Transported ~., data = full.model.train.features, ntree = 171)
rf2.train.predict <- predict(rf2, newdata = full.model.train.features,type='response')
table(rf2.train.predict,full.model.train.features$Transported)
caret::confusionMatrix(rf2.train.predict,full.model.train.features$Transported)
varImpPlot(rf2)


#Predict test data set
rf.predict <- predict(rf2, newdata = full.model.test.features,type='response')
rf.predict <- as.data.frame(rf.predict)
rf.predict$PassengerId <- full[full$id=='test','PassengerId']

rf.predict <- rf.predict %>% dplyr::select(PassengerId,Transported=rf.predict)
rf.predict


# Submit File
write.csv(rf.predict, file = "RandomForest_Prediction.csv",row.names=FALSE,quote=FALSE)

</pre>