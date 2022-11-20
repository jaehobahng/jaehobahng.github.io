```R
library(dplyr)
library(ggplot2)
library(gridExtra)
library(stringr)
library(caret)
library(mice)
library(randomForest)
library(Boruta)
```

# Bind train/test data sets


```R
train <- read.csv("../input/spaceship-titanic/train.csv")
test <- read.csv("../input/spaceship-titanic/test.csv")
str(train)
str(test)

###TEST TRAIN SET BIND###
test$Transported <- NA
full <- bind_rows(train,test,.id='id')
full$id <- ifelse(full$id=='1','train','test')

str(full)
```

    'data.frame':	8693 obs. of  14 variables:
     $ PassengerId : chr  "0001_01" "0002_01" "0003_01" "0003_02" ...
     $ HomePlanet  : chr  "Europa" "Earth" "Europa" "Europa" ...
     $ CryoSleep   : chr  "False" "False" "False" "False" ...
     $ Cabin       : chr  "B/0/P" "F/0/S" "A/0/S" "A/0/S" ...
     $ Destination : chr  "TRAPPIST-1e" "TRAPPIST-1e" "TRAPPIST-1e" "TRAPPIST-1e" ...
     $ Age         : num  39 24 58 33 16 44 26 28 35 14 ...
     $ VIP         : chr  "False" "False" "True" "False" ...
     $ RoomService : num  0 109 43 0 303 0 42 0 0 0 ...
     $ FoodCourt   : num  0 9 3576 1283 70 ...
     $ ShoppingMall: num  0 25 0 371 151 0 3 0 17 0 ...
     $ Spa         : num  0 549 6715 3329 565 ...
     $ VRDeck      : num  0 44 49 193 2 0 0 NA 0 0 ...
     $ Name        : chr  "Maham Ofracculy" "Juanna Vines" "Altark Susent" "Solam Susent" ...
     $ Transported : chr  "False" "True" "False" "False" ...
    'data.frame':	4277 obs. of  13 variables:
     $ PassengerId : chr  "0013_01" "0018_01" "0019_01" "0021_01" ...
     $ HomePlanet  : chr  "Earth" "Earth" "Europa" "Europa" ...
     $ CryoSleep   : chr  "True" "False" "True" "False" ...
     $ Cabin       : chr  "G/3/S" "F/4/S" "C/0/S" "C/1/S" ...
     $ Destination : chr  "TRAPPIST-1e" "TRAPPIST-1e" "55 Cancri e" "TRAPPIST-1e" ...
     $ Age         : num  27 19 31 38 20 31 21 20 23 24 ...
     $ VIP         : chr  "False" "False" "False" "False" ...
     $ RoomService : num  0 0 0 0 10 0 0 0 0 0 ...
     $ FoodCourt   : num  0 9 0 6652 0 ...
     $ ShoppingMall: num  0 0 0 0 635 263 0 0 0 0 ...
     $ Spa         : num  0 2823 0 181 0 ...
     $ VRDeck      : num  0 0 0 585 0 60 0 0 0 0 ...
     $ Name        : chr  "Nelly Carsoning" "Lerome Peckers" "Sabih Unhearfus" "Meratz Caltilter" ...
    'data.frame':	12970 obs. of  15 variables:
     $ id          : chr  "train" "train" "train" "train" ...
     $ PassengerId : chr  "0001_01" "0002_01" "0003_01" "0003_02" ...
     $ HomePlanet  : chr  "Europa" "Earth" "Europa" "Europa" ...
     $ CryoSleep   : chr  "False" "False" "False" "False" ...
     $ Cabin       : chr  "B/0/P" "F/0/S" "A/0/S" "A/0/S" ...
     $ Destination : chr  "TRAPPIST-1e" "TRAPPIST-1e" "TRAPPIST-1e" "TRAPPIST-1e" ...
     $ Age         : num  39 24 58 33 16 44 26 28 35 14 ...
     $ VIP         : chr  "False" "False" "True" "False" ...
     $ RoomService : num  0 109 43 0 303 0 42 0 0 0 ...
     $ FoodCourt   : num  0 9 3576 1283 70 ...
     $ ShoppingMall: num  0 25 0 371 151 0 3 0 17 0 ...
     $ Spa         : num  0 549 6715 3329 565 ...
     $ VRDeck      : num  0 44 49 193 2 0 0 NA 0 0 ...
     $ Name        : chr  "Maham Ofracculy" "Juanna Vines" "Altark Susent" "Solam Susent" ...
     $ Transported : chr  "False" "True" "False" "False" ...
    

# Column Formating
##### 1. Chnage Blank cells to NA value
##### 2. CryoSleep : True,False to 1,0
##### 3. VIP : True,False to 1,0
##### 4. Transported to factors



```R
# 1. Blank cells to NA value
colSums(is.na(full))
full[full=='' | full==' '] <- NA
colSums(is.na(full))

# 2. CryoSleep : True,False to 1,0
full$CryoSleep <- ifelse(full$CryoSleep=='True',1,ifelse(full$CryoSleep=='False',0,NA))

# 3. VIP : True,False to 1,0
full$VIP <- ifelse(full$VIP=='True',1,ifelse(full$VIP=='False',0,NA))


# 4. Transported to factors
full$Transported <- factor(full$Transported,levels=c('True','False'))
```


<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>id</dt><dd>0</dd><dt>PassengerId</dt><dd>0</dd><dt>HomePlanet</dt><dd>0</dd><dt>CryoSleep</dt><dd>0</dd><dt>Cabin</dt><dd>0</dd><dt>Destination</dt><dd>0</dd><dt>Age</dt><dd>270</dd><dt>VIP</dt><dd>0</dd><dt>RoomService</dt><dd>263</dd><dt>FoodCourt</dt><dd>289</dd><dt>ShoppingMall</dt><dd>306</dd><dt>Spa</dt><dd>284</dd><dt>VRDeck</dt><dd>268</dd><dt>Name</dt><dd>0</dd><dt>Transported</dt><dd>4277</dd></dl>




<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>id</dt><dd>0</dd><dt>PassengerId</dt><dd>0</dd><dt>HomePlanet</dt><dd>288</dd><dt>CryoSleep</dt><dd>310</dd><dt>Cabin</dt><dd>299</dd><dt>Destination</dt><dd>274</dd><dt>Age</dt><dd>270</dd><dt>VIP</dt><dd>296</dd><dt>RoomService</dt><dd>263</dd><dt>FoodCourt</dt><dd>289</dd><dt>ShoppingMall</dt><dd>306</dd><dt>Spa</dt><dd>284</dd><dt>VRDeck</dt><dd>268</dd><dt>Name</dt><dd>294</dd><dt>Transported</dt><dd>4277</dd></dl>



# Feature engineering
#### 1. Split Name(First/Last)
#### 2. Split Cabin(Deck/Num/Side)
#### 3. Create Total fee column



```R
# 1. Separate First/Last name and drop Name Column
full$Lastname <- str_split(full$Name,' ',simplify=TRUE)[,2]
full$Name <- NULL
str(full)

# 2. Cabin Separation(Deck/Num/Side) and drop original Cabin Column
full$Cabin
full$Deck_Cabin <- str_split(full$Cabin,'/',simplify=TRUE)[,1]
full$Num_Cabin <- str_split(full$Cabin,'/',simplify=TRUE)[,2]
full$Side_Cabin <- str_split(full$Cabin,'/',simplify=TRUE)[,3]
full$Cabin <- NULL

# 3. Create column for sum of all fees used by customer
#    - Substitude missing fee values to 0
str(full)
fee <- c('RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck')
for(i in fee){
  full[is.na(full[,i]),i] <- 0
}
 
full$fee <- rowSums(full[,fee])

str(full)
```

    'data.frame':	12970 obs. of  15 variables:
     $ id          : chr  "train" "train" "train" "train" ...
     $ PassengerId : chr  "0001_01" "0002_01" "0003_01" "0003_02" ...
     $ HomePlanet  : chr  "Europa" "Earth" "Europa" "Europa" ...
     $ CryoSleep   : num  0 0 0 0 0 0 0 1 0 1 ...
     $ Cabin       : chr  "B/0/P" "F/0/S" "A/0/S" "A/0/S" ...
     $ Destination : chr  "TRAPPIST-1e" "TRAPPIST-1e" "TRAPPIST-1e" "TRAPPIST-1e" ...
     $ Age         : num  39 24 58 33 16 44 26 28 35 14 ...
     $ VIP         : num  0 0 1 0 0 0 0 0 0 0 ...
     $ RoomService : num  0 109 43 0 303 0 42 0 0 0 ...
     $ FoodCourt   : num  0 9 3576 1283 70 ...
     $ ShoppingMall: num  0 25 0 371 151 0 3 0 17 0 ...
     $ Spa         : num  0 549 6715 3329 565 ...
     $ VRDeck      : num  0 44 49 193 2 0 0 NA 0 0 ...
     $ Transported : Factor w/ 2 levels "True","False": 2 1 2 2 1 1 1 1 1 1 ...
     $ Lastname    : chr  "Ofracculy" "Vines" "Susent" "Susent" ...
    


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'B/0/P'</li><li>'F/0/S'</li><li>'A/0/S'</li><li>'A/0/S'</li><li>'F/1/S'</li><li>'F/0/P'</li><li>'F/2/S'</li><li>'G/0/S'</li><li>'F/3/S'</li><li>'B/1/P'</li><li>'B/1/P'</li><li>'B/1/P'</li><li>'F/1/P'</li><li>'G/1/S'</li><li>'F/2/P'</li><li>NA</li><li>'F/3/P'</li><li>'F/4/P'</li><li>'F/5/P'</li><li>'G/0/P'</li><li>'F/6/P'</li><li>'E/0/S'</li><li>'E/0/S'</li><li>'E/0/S'</li><li>'E/0/S'</li><li>'E/0/S'</li><li>'E/0/S'</li><li>'D/0/P'</li><li>'C/2/S'</li><li>'F/6/S'</li><li>'C/0/P'</li><li>'F/8/P'</li><li>'G/4/S'</li><li>'F/9/P'</li><li>'F/9/P'</li><li>'F/9/P'</li><li>'D/1/S'</li><li>'D/1/P'</li><li>'F/8/S'</li><li>'F/10/S'</li><li>'G/1/P'</li><li>'G/2/P'</li><li>'B/3/P'</li><li>'G/3/P'</li><li>'G/3/P'</li><li>'G/3/P'</li><li>'F/10/P'</li><li>'F/10/P'</li><li>'E/1/S'</li><li>'E/2/S'</li><li>'G/6/S'</li><li>'F/11/S'</li><li>'A/1/S'</li><li>'A/1/S'</li><li>'A/1/S'</li><li>'G/7/S'</li><li>'F/12/S'</li><li>'F/13/S'</li><li>'F/14/S'</li><li>'E/3/S'</li><li>'G/6/P'</li><li>'G/10/S'</li><li>'G/10/S'</li><li>'F/15/S'</li><li>'E/4/S'</li><li>'F/16/S'</li><li>'F/13/P'</li><li>'F/14/P'</li><li>'F/17/S'</li><li>'D/3/P'</li><li>'C/3/S'</li><li>'F/18/S'</li><li>'F/15/P'</li><li>'C/4/S'</li><li>'G/13/S'</li><li>'F/16/P'</li><li>'F/16/P'</li><li>'F/16/P'</li><li>'G/14/S'</li><li>'C/5/S'</li><li>'F/17/P'</li><li>'E/5/S'</li><li>'G/15/S'</li><li>'G/16/S'</li><li>'F/20/S'</li><li>'G/9/P'</li><li>'G/9/P'</li><li>'G/9/P'</li><li>'A/2/S'</li><li>'G/11/P'</li><li>'G/11/P'</li><li>'F/19/P'</li><li>'G/12/P'</li><li>NA</li><li>'F/23/S'</li><li>'F/24/S'</li><li>'G/18/S'</li><li>'G/18/S'</li><li>'F/21/P'</li><li>'D/2/S'</li><li>'G/19/S'</li><li>'G/19/S'</li><li>'G/19/S'</li><li>NA</li><li>'B/5/P'</li><li>'B/5/P'</li><li>'B/5/P'</li><li>'E/6/S'</li><li>'B/1/S'</li><li>'F/23/P'</li><li>'G/20/S'</li><li>'F/24/P'</li><li>'D/4/P'</li><li>'A/0/P'</li><li>'A/0/P'</li><li>'F/25/P'</li><li>'G/21/S'</li><li>'F/27/P'</li><li>'F/27/S'</li><li>'E/7/S'</li><li>'D/3/S'</li><li>'E/8/S'</li><li>'G/22/S'</li><li>'F/29/S'</li><li>'D/5/S'</li><li>'G/17/P'</li><li>'G/23/S'</li><li>'G/18/P'</li><li>'E/5/P'</li><li>'C/6/S'</li><li>'G/19/P'</li><li>'F/29/P'</li><li>'F/30/P'</li><li>'F/31/S'</li><li>'G/25/S'</li><li>'G/26/S'</li><li>'F/31/P'</li><li>'G/27/S'</li><li>'G/20/P'</li><li>'F/32/P'</li><li>'G/22/P'</li><li>'B/8/P'</li><li>'B/8/P'</li><li>'G/28/S'</li><li>'G/28/S'</li><li>'F/37/P'</li><li>'F/35/S'</li><li>'F/35/S'</li><li>'G/23/P'</li><li>'E/10/S'</li><li>'G/30/S'</li><li>'G/24/P'</li><li>'E/11/S'</li><li>'F/38/P'</li><li>'B/2/S'</li><li>'D/6/S'</li><li>'G/26/P'</li><li>'G/26/P'</li><li>'G/26/P'</li><li>'G/27/P'</li><li>'C/3/P'</li><li>'F/36/S'</li><li>'G/28/P'</li><li>'G/31/S'</li><li>'E/8/P'</li><li>'G/32/S'</li><li>'G/29/P'</li><li>'G/29/P'</li><li>'G/29/P'</li><li>'F/41/P'</li><li>'F/41/P'</li><li>'F/41/P'</li><li>'F/42/P'</li><li>'D/5/P'</li><li>'C/5/P'</li><li>'G/30/P'</li><li>'G/33/S'</li><li>'G/31/P'</li><li>'B/3/S'</li><li>'B/3/S'</li><li>'B/9/P'</li><li>'A/2/P'</li><li>'C/8/S'</li><li>'G/34/S'</li><li>'C/9/S'</li><li>'G/32/P'</li><li>'D/6/P'</li><li>'F/44/P'</li><li>'F/44/P'</li><li>'F/44/P'</li><li>'E/9/P'</li><li>'F/45/P'</li><li>'F/46/P'</li><li>'F/40/S'</li><li>'F/47/P'</li><li>'G/36/P'</li><li>'G/37/P'</li><li>'G/37/P'</li><li>'G/37/P'</li><li>'E/10/P'</li><li>⋯</li><li>'C/330/S'</li><li>'C/330/S'</li><li>'C/292/P'</li><li>'C/292/P'</li><li>'G/1437/P'</li><li>'E/583/S'</li><li>'E/569/P'</li><li>'F/1720/S'</li><li>'C/293/P'</li><li>'C/293/P'</li><li>'C/294/P'</li><li>'C/294/P'</li><li>'E/584/S'</li><li>'G/1439/S'</li><li>'F/1833/P'</li><li>'D/279/P'</li><li>'G/1443/P'</li><li>'G/1444/P'</li><li>'G/1444/P'</li><li>'G/1444/P'</li><li>NA</li><li>'G/1446/P'</li><li>'G/1447/P'</li><li>'G/1440/S'</li><li>'B/288/P'</li><li>'C/332/S'</li><li>'C/295/P'</li><li>'C/295/P'</li><li>'C/295/P'</li><li>'C/295/P'</li><li>'C/295/P'</li><li>'C/295/P'</li><li>'C/296/P'</li><li>'F/1724/S'</li><li>'F/1726/S'</li><li>'E/572/P'</li><li>'E/573/P'</li><li>'F/1728/S'</li><li>'E/586/S'</li><li>'F/1730/S'</li><li>'C/334/S'</li><li>'F/1835/P'</li><li>'G/1451/P'</li><li>'B/289/P'</li><li>'F/1836/P'</li><li>'F/1732/S'</li><li>'G/1443/S'</li><li>'G/1454/P'</li><li>'G/1444/S'</li><li>'G/1444/S'</li><li>'G/1444/S'</li><li>'F/1733/S'</li><li>NA</li><li>'D/271/S'</li><li>'B/290/P'</li><li>'C/301/P'</li><li>'F/1738/S'</li><li>'D/273/S'</li><li>'D/273/S'</li><li>'D/273/S'</li><li>'D/273/S'</li><li>'D/273/S'</li><li>'D/273/S'</li><li>'D/273/S'</li><li>NA</li><li>'G/1457/P'</li><li>'F/1843/P'</li><li>'F/1845/P'</li><li>'F/1845/P'</li><li>'D/284/P'</li><li>'G/1450/S'</li><li>'G/1451/S'</li><li>'G/1452/S'</li><li>'B/345/S'</li><li>'B/345/S'</li><li>'B/345/S'</li><li>'B/345/S'</li><li>'B/345/S'</li><li>'G/1453/S'</li><li>'F/1748/S'</li><li>'F/1748/S'</li><li>'F/1748/S'</li><li>'G/1454/S'</li><li>'G/1454/S'</li><li>'F/1750/S'</li><li>'G/1456/S'</li><li>'G/1468/P'</li><li>'G/1469/P'</li><li>'F/1849/P'</li><li>'F/1850/P'</li><li>'E/579/P'</li><li>'F/1851/P'</li><li>'G/1472/P'</li><li>'G/1473/P'</li><li>'B/295/P'</li><li>'B/295/P'</li><li>'G/1458/S'</li><li>'F/1852/P'</li><li>'D/289/P'</li><li>NA</li><li>'F/1854/P'</li><li>NA</li><li>'B/346/S'</li><li>'F/1755/S'</li><li>'G/1459/S'</li><li>'G/1459/S'</li><li>NA</li><li>'F/1757/S'</li><li>'G/1477/P'</li><li>'F/1859/P'</li><li>'E/581/P'</li><li>'G/1462/S'</li><li>'F/1759/S'</li><li>'F/1860/P'</li><li>'E/596/S'</li><li>'G/1464/S'</li><li>'F/1861/P'</li><li>'F/1862/P'</li><li>'A/108/S'</li><li>'G/1478/P'</li><li>'G/1466/S'</li><li>'G/1466/S'</li><li>'E/583/P'</li><li>'F/1760/S'</li><li>'G/1480/P'</li><li>'G/1481/P'</li><li>'F/1762/S'</li><li>'E/584/P'</li><li>'F/1869/P'</li><li>'F/1764/S'</li><li>'G/1486/P'</li><li>'G/1487/P'</li><li>NA</li><li>'G/1472/S'</li><li>'F/1767/S'</li><li>'G/1472/S'</li><li>'F/1874/P'</li><li>'D/293/P'</li><li>'F/1874/P'</li><li>'G/1490/P'</li><li>'G/1492/P'</li><li>'G/1475/S'</li><li>'C/304/P'</li><li>'C/304/P'</li><li>'C/304/P'</li><li>'C/304/P'</li><li>'D/294/P'</li><li>'D/294/P'</li><li>'D/294/P'</li><li>'A/109/S'</li><li>'C/305/P'</li><li>'C/306/P'</li><li>'C/306/P'</li><li>'F/1776/S'</li><li>'E/589/P'</li><li>'F/1879/P'</li><li>'G/1482/S'</li><li>'D/277/S'</li><li>'E/591/P'</li><li>'F/1781/S'</li><li>'B/352/S'</li><li>'B/352/S'</li><li>'B/352/S'</li><li>'C/340/S'</li><li>'F/1881/P'</li><li>'F/1882/P'</li><li>'F/1883/P'</li><li>'G/1495/P'</li><li>'C/341/S'</li><li>'A/96/P'</li><li>'F/1885/P'</li><li>NA</li><li>NA</li><li>'F/1887/P'</li><li>'E/605/S'</li><li>'G/1499/P'</li><li>'F/1790/S'</li><li>'G/1501/P'</li><li>'G/1501/P'</li><li>'G/1501/P'</li><li>'G/1501/P'</li><li>NA</li><li>'G/1501/P'</li><li>'F/1890/P'</li><li>'E/594/P'</li><li>'E/596/P'</li><li>'F/1791/S'</li><li>'G/1492/S'</li><li>'F/1794/S'</li><li>'E/598/P'</li><li>'G/1503/P'</li><li>'F/1795/S'</li><li>'G/1495/S'</li><li>'D/278/S'</li><li>'F/1796/S'</li><li>'G/1496/S'</li><li>NA</li><li>'D/296/P'</li><li>'D/297/P'</li><li>'G/1498/S'</li></ol>



    'data.frame':	12970 obs. of  17 variables:
     $ id          : chr  "train" "train" "train" "train" ...
     $ PassengerId : chr  "0001_01" "0002_01" "0003_01" "0003_02" ...
     $ HomePlanet  : chr  "Europa" "Earth" "Europa" "Europa" ...
     $ CryoSleep   : num  0 0 0 0 0 0 0 1 0 1 ...
     $ Destination : chr  "TRAPPIST-1e" "TRAPPIST-1e" "TRAPPIST-1e" "TRAPPIST-1e" ...
     $ Age         : num  39 24 58 33 16 44 26 28 35 14 ...
     $ VIP         : num  0 0 1 0 0 0 0 0 0 0 ...
     $ RoomService : num  0 109 43 0 303 0 42 0 0 0 ...
     $ FoodCourt   : num  0 9 3576 1283 70 ...
     $ ShoppingMall: num  0 25 0 371 151 0 3 0 17 0 ...
     $ Spa         : num  0 549 6715 3329 565 ...
     $ VRDeck      : num  0 44 49 193 2 0 0 NA 0 0 ...
     $ Transported : Factor w/ 2 levels "True","False": 2 1 2 2 1 1 1 1 1 1 ...
     $ Lastname    : chr  "Ofracculy" "Vines" "Susent" "Susent" ...
     $ Deck_Cabin  : chr  "B" "F" "A" "A" ...
     $ Num_Cabin   : chr  "0" "0" "0" "0" ...
     $ Side_Cabin  : chr  "P" "S" "S" "S" ...
    'data.frame':	12970 obs. of  18 variables:
     $ id          : chr  "train" "train" "train" "train" ...
     $ PassengerId : chr  "0001_01" "0002_01" "0003_01" "0003_02" ...
     $ HomePlanet  : chr  "Europa" "Earth" "Europa" "Europa" ...
     $ CryoSleep   : num  0 0 0 0 0 0 0 1 0 1 ...
     $ Destination : chr  "TRAPPIST-1e" "TRAPPIST-1e" "TRAPPIST-1e" "TRAPPIST-1e" ...
     $ Age         : num  39 24 58 33 16 44 26 28 35 14 ...
     $ VIP         : num  0 0 1 0 0 0 0 0 0 0 ...
     $ RoomService : num  0 109 43 0 303 0 42 0 0 0 ...
     $ FoodCourt   : num  0 9 3576 1283 70 ...
     $ ShoppingMall: num  0 25 0 371 151 0 3 0 17 0 ...
     $ Spa         : num  0 549 6715 3329 565 ...
     $ VRDeck      : num  0 44 49 193 2 0 0 0 0 0 ...
     $ Transported : Factor w/ 2 levels "True","False": 2 1 2 2 1 1 1 1 1 1 ...
     $ Lastname    : chr  "Ofracculy" "Vines" "Susent" "Susent" ...
     $ Deck_Cabin  : chr  "B" "F" "A" "A" ...
     $ Num_Cabin   : chr  "0" "0" "0" "0" ...
     $ Side_Cabin  : chr  "P" "S" "S" "S" ...
     $ fee         : num  0 736 10383 5176 1091 ...
    

# Replace Missing Values
#### 1. CryoSleep(based on fee)
#### 2. Mice to replace CryoSleep/Age/VIP
#### 3. HomePlanet based on Lastname


```R
#Check Missing Values
colSums(is.na(full))
```


<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>id</dt><dd>0</dd><dt>PassengerId</dt><dd>0</dd><dt>HomePlanet</dt><dd>288</dd><dt>CryoSleep</dt><dd>310</dd><dt>Destination</dt><dd>274</dd><dt>Age</dt><dd>270</dd><dt>VIP</dt><dd>296</dd><dt>RoomService</dt><dd>0</dd><dt>FoodCourt</dt><dd>0</dd><dt>ShoppingMall</dt><dd>0</dd><dt>Spa</dt><dd>0</dd><dt>VRDeck</dt><dd>0</dd><dt>Transported</dt><dd>4277</dd><dt>Lastname</dt><dd>0</dd><dt>Deck_Cabin</dt><dd>299</dd><dt>Num_Cabin</dt><dd>0</dd><dt>Side_Cabin</dt><dd>0</dd><dt>fee</dt><dd>0</dd></dl>




```R
# 1. People in CryoSleep(TRUE) do not use any money
full[full$fee>0&full$CryoSleep=='True'&!is.na(full$CryoSleep),c('CryoSleep','fee')]
full[(full$fee>0&is.na(full$CryoSleep)),c('CryoSleep')] <- 0
sum((is.na(full$CryoSleep)))
```


<table class="dataframe">
<caption>A data.frame: 0 × 2</caption>
<thead>
	<tr><th scope=col>CryoSleep</th><th scope=col>fee</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
</tbody>
</table>




136



```R
# 2. Use Mice to replace columns with integers(CryoSleep, Age, VIP)
library(mice)
mice_mod <- mice(full[,names(full) %in% c('Age','fee','VIP','CryoSleep','Destination','HomePlanet','Deck_Cabin','CryoSleep')],
                 method = 'rf')
```

    
     iter imp variable
      1   1  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      1   2  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      1   3  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      1   4  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      1   5  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      2   1  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      2   2  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      2   3  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      2   4  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      2   5  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      3   1  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      3   2  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      3   3  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      3   4  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      3   5  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      4   1  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      4   2  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      4   3  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      4   4  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      4   5  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      5   1  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      5   2  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      5   3  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      5   4  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
      5   5  CryoSleep

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

      Age  VIP

    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    Warning message in randomForest.default(x = xobs, y = yobs, ntree = 1, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”
    

    
    

    Warning message:
    “Number of logged events: 3”
    


```R
# 2.1 Check if there is an abnormal difference in replaced and original Values
par(mfrow=c(3,2))
hist(complete(mice_mod)$Age,main = 'Mice Age')
hist(full$Age, main= 'Original Age')
hist(complete(mice_mod)$CryoSleep,main = 'Mice CryoSleep')
hist(full$CryoSleep, main= 'Original CryoSleep')
hist(complete(mice_mod)$VIP,main = 'Mice VIP')
hist(full$VIP, main= 'Original VIP')


full$Age <- complete(mice_mod)$Age
full$CryoSleep <- complete(mice_mod)$CryoSleep
full$VIP <- complete(mice_mod)$VIP
```


    
![png](output_11_0.png)
    



```R
#3. Replace HomePlanet missing values based on LastName
#      Assume that people with the same last name should be from the same HomePlanet
#      Replace missing HomePlanet values with the HomePlanet values of other idential LastName passengers

#Check that there are passengers who have the same Lastname but one is missing a HomePlanet
print(full %>% group_by(Lastname,HomePlanet) %>% summarise(count = n()) %>% arrange(Lastname,desc(count)),n=100)
```

    `summarise()` has grouped output by 'Lastname'. You can override using the `.groups` argument.
    
    

    [90m# A tibble: 2,668 × 3[39m
    [90m# Groups:   Lastname [2,407][39m
        Lastname      HomePlanet count
        [3m[90m<chr>[39m[23m         [3m[90m<chr>[39m[23m      [3m[90m<int>[39m[23m
    [90m  1[39m [90m"[39m[90m"[39m            Earth        159
    [90m  2[39m [90m"[39m[90m"[39m            Europa        65
    [90m  3[39m [90m"[39m[90m"[39m            Mars          60
    [90m  4[39m [90m"[39m[90m"[39m            [31mNA[39m            10
    [90m  5[39m [90m"[39mAcobson[90m"[39m     Earth          5
    [90m  6[39m [90m"[39mAcobsond[90m"[39m    Earth          8
    [90m  7[39m [90m"[39mAdavisons[90m"[39m   Earth         10
    [90m  8[39m [90m"[39mAdkinson[90m"[39m    Earth          4
    [90m  9[39m [90m"[39mAdmingried[90m"[39m  Europa         4
    [90m 10[39m [90m"[39mAgeurante[90m"[39m   [31mNA[39m             1
    [90m 11[39m [90m"[39mAginge[90m"[39m      Europa         4
    [90m 12[39m [90m"[39mAginoid[90m"[39m     Europa         2
    [90m 13[39m [90m"[39mAilled[90m"[39m      Europa         2
    [90m 14[39m [90m"[39mAillyber[90m"[39m    Europa         7
    [90m 15[39m [90m"[39mAiming[90m"[39m      Europa         2
    [90m 16[39m [90m"[39mAinatint[90m"[39m    Europa         3
    [90m 17[39m [90m"[39mAindlylid[90m"[39m   Europa         4
    [90m 18[39m [90m"[39mAinserfle[90m"[39m   Europa        11
    [90m 19[39m [90m"[39mAinserfle[90m"[39m   [31mNA[39m             1
    [90m 20[39m [90m"[39mAirdring[90m"[39m    Europa         4
    [90m 21[39m [90m"[39mAivering[90m"[39m    Europa         2
    [90m 22[39m [90m"[39mAlaring[90m"[39m     Europa        10
    [90m 23[39m [90m"[39mAlaxed[90m"[39m      Europa         2
    [90m 24[39m [90m"[39mAlberts[90m"[39m     Earth          7
    [90m 25[39m [90m"[39mAlcemblery[90m"[39m  Europa         5
    [90m 26[39m [90m"[39mAlenat[90m"[39m      Europa         3
    [90m 27[39m [90m"[39mAlenter[90m"[39m     Europa         6
    [90m 28[39m [90m"[39mAlentonway[90m"[39m  Earth          6
    [90m 29[39m [90m"[39mAlest[90m"[39m       Earth          6
    [90m 30[39m [90m"[39mAlfordonard[90m"[39m Earth          7
    [90m 31[39m [90m"[39mAlfordonard[90m"[39m [31mNA[39m             1
    [90m 32[39m [90m"[39mAlindiveng[90m"[39m  Europa         4
    [90m 33[39m [90m"[39mAlldson[90m"[39m     Earth         14
    [90m 34[39m [90m"[39mAloquinght[90m"[39m  Europa         3
    [90m 35[39m [90m"[39mAloubtled[90m"[39m   Europa         7
    [90m 36[39m [90m"[39mAlshipson[90m"[39m   Earth          7
    [90m 37[39m [90m"[39mAlshipson[90m"[39m   [31mNA[39m             1
    [90m 38[39m [90m"[39mAlutorody[90m"[39m   Europa         1
    [90m 39[39m [90m"[39mAlvasquez[90m"[39m   Earth          8
    [90m 40[39m [90m"[39mAlvercal[90m"[39m    Europa         2
    [90m 41[39m [90m"[39mAlvesssidy[90m"[39m  Europa         2
    [90m 42[39m [90m"[39mAmbleetive[90m"[39m  Europa         7
    [90m 43[39m [90m"[39mAmbleetive[90m"[39m  [31mNA[39m             1
    [90m 44[39m [90m"[39mAmbleeve[90m"[39m    Europa         5
    [90m 45[39m [90m"[39mAmblereld[90m"[39m   Europa         2
    [90m 46[39m [90m"[39mAmetic[90m"[39m      Europa        12
    [90m 47[39m [90m"[39mAmincrerus[90m"[39m  Europa         5
    [90m 48[39m [90m"[39mAmonsmane[90m"[39m   Europa         2
    [90m 49[39m [90m"[39mAmonysidle[90m"[39m  Europa         1
    [90m 50[39m [90m"[39mAmoutake[90m"[39m    Europa         4
    [90m 51[39m [90m"[39mAmsive[90m"[39m      [31mNA[39m             1
    [90m 52[39m [90m"[39mAmspring[90m"[39m    Europa         2
    [90m 53[39m [90m"[39mAnake[90m"[39m       Mars           2
    [90m 54[39m [90m"[39mAnate[90m"[39m       Mars           8
    [90m 55[39m [90m"[39mAnche[90m"[39m       Mars           3
    [90m 56[39m [90m"[39mAncontaked[90m"[39m  Europa         1
    [90m 57[39m [90m"[39mAncy[90m"[39m        Mars           6
    [90m 58[39m [90m"[39mAndackson[90m"[39m   Earth          8
    [90m 59[39m [90m"[39mAnderking[90m"[39m   Europa         2
    [90m 60[39m [90m"[39mAndley[90m"[39m      Earth          6
    [90m 61[39m [90m"[39mAne[90m"[39m         Mars           4
    [90m 62[39m [90m"[39mAneetle[90m"[39m     Europa         5
    [90m 63[39m [90m"[39mAneter[90m"[39m      Europa         3
    [90m 64[39m [90m"[39mAnindery[90m"[39m    Europa         4
    [90m 65[39m [90m"[39mAnpie[90m"[39m       Mars           6
    [90m 66[39m [90m"[39mAntcal[90m"[39m      Europa         7
    [90m 67[39m [90m"[39mAnthompson[90m"[39m  Earth          7
    [90m 68[39m [90m"[39mAnthompson[90m"[39m  [31mNA[39m             2
    [90m 69[39m [90m"[39mAnthon[90m"[39m      Earth          2
    [90m 70[39m [90m"[39mAntoshipson[90m"[39m Earth          7
    [90m 71[39m [90m"[39mApeau[90m"[39m       Mars           5
    [90m 72[39m [90m"[39mApedishaft[90m"[39m  Europa         2
    [90m 73[39m [90m"[39mApenelexy[90m"[39m   Europa         3
    [90m 74[39m [90m"[39mApenelexy[90m"[39m   [31mNA[39m             1
    [90m 75[39m [90m"[39mApie[90m"[39m        Mars          11
    [90m 76[39m [90m"[39mApity[90m"[39m       Mars           2
    [90m 77[39m [90m"[39mAppie[90m"[39m       Mars           2
    [90m 78[39m [90m"[39mApple[90m"[39m       Mars           8
    [90m 79[39m [90m"[39mApple[90m"[39m       [31mNA[39m             1
    [90m 80[39m [90m"[39mArible[90m"[39m      Europa         7
    [90m 81[39m [90m"[39mArmstromez[90m"[39m  Earth          7
    [90m 82[39m [90m"[39mArner[90m"[39m       Earth          6
    [90m 83[39m [90m"[39mArneras[90m"[39m     Earth          9
    [90m 84[39m [90m"[39mAroodint[90m"[39m    Europa         6
    [90m 85[39m [90m"[39mArterate[90m"[39m    Europa         1
    [90m 86[39m [90m"[39mAsharing[90m"[39m    Europa         1
    [90m 87[39m [90m"[39mAshipeck[90m"[39m    Earth          4
    [90m 88[39m [90m"[39mAshipson[90m"[39m    Earth          7
    [90m 89[39m [90m"[39mAsivetfuel[90m"[39m  Europa         1
    [90m 90[39m [90m"[39mAsolipery[90m"[39m   Europa         4
    [90m 91[39m [90m"[39mAsoppor[90m"[39m     Europa         3
    [90m 92[39m [90m"[39mAssefle[90m"[39m     Europa         4
    [90m 93[39m [90m"[39mAsseple[90m"[39m     Europa         3
    [90m 94[39m [90m"[39mAssibler[90m"[39m    Europa         4
    [90m 95[39m [90m"[39mAsticit[90m"[39m     Europa         3
    [90m 96[39m [90m"[39mAtiveezy[90m"[39m    Europa         3
    [90m 97[39m [90m"[39mAtkinney[90m"[39m    Earth          5
    [90m 98[39m [90m"[39mAusivetpul[90m"[39m  Europa         3
    [90m 99[39m [90m"[39mAvidson[90m"[39m     Earth          5
    [90m100[39m [90m"[39mAvisley[90m"[39m     Earth          2
    [90m# … with 2,568 more rows[39m
    


```R
#Table to find what unique HomePlanet values Last Name Members have
vlookup <- full %>% filter(!is.na(Lastname)&!is.na(HomePlanet)) %>% dplyr::select(Lastname,HomePlanet) %>% distinct(Lastname,HomePlanet) %>% arrange(Lastname)

#(Check if the same Lastname can have multiple HomePlanets)
#   Blank has all the values so the most common value(earth) was chosen to replace NA values
head(vlookup %>% group_by(Lastname) %>% summarise(count=n()) %>% arrange(desc(count)),n=3)
vlookup <- vlookup %>% 
  filter(!row_number() %in% c(1,3))
```


<table class="dataframe">
<caption>A tibble: 3 × 2</caption>
<thead>
	<tr><th scope=col>Lastname</th><th scope=col>count</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>        </td><td>3</td></tr>
	<tr><td>Acobson </td><td>1</td></tr>
	<tr><td>Acobsond</td><td>1</td></tr>
</tbody>
</table>




```R
#Passengers who do not have a HomePlanet value
na.HomePlanet <- full %>% filter(is.na(HomePlanet)) %>% dplyr::select(HomePlanet,Lastname)

#Join NA values with HomePlanet of identical Lastname Passengers
na.vlookup <- na.HomePlanet %>% left_join(vlookup,by='Lastname')

#Replace Table value with values found above
full[is.na(full$HomePlanet),c('HomePlanet','Lastname')]$HomePlanet <- na.vlookup$HomePlanet.y

print(full %>% group_by(Lastname,HomePlanet) %>% summarise(count=n()),n=100)
```

    `summarise()` has grouped output by 'Lastname'. You can override using the `.groups` argument.
    
    

    [90m# A tibble: 2,409 × 3[39m
    [90m# Groups:   Lastname [2,407][39m
        Lastname      HomePlanet count
        [3m[90m<chr>[39m[23m         [3m[90m<chr>[39m[23m      [3m[90m<int>[39m[23m
    [90m  1[39m [90m"[39m[90m"[39m            Earth        169
    [90m  2[39m [90m"[39m[90m"[39m            Europa        65
    [90m  3[39m [90m"[39m[90m"[39m            Mars          60
    [90m  4[39m [90m"[39mAcobson[90m"[39m     Earth          5
    [90m  5[39m [90m"[39mAcobsond[90m"[39m    Earth          8
    [90m  6[39m [90m"[39mAdavisons[90m"[39m   Earth         10
    [90m  7[39m [90m"[39mAdkinson[90m"[39m    Earth          4
    [90m  8[39m [90m"[39mAdmingried[90m"[39m  Europa         4
    [90m  9[39m [90m"[39mAgeurante[90m"[39m   [31mNA[39m             1
    [90m 10[39m [90m"[39mAginge[90m"[39m      Europa         4
    [90m 11[39m [90m"[39mAginoid[90m"[39m     Europa         2
    [90m 12[39m [90m"[39mAilled[90m"[39m      Europa         2
    [90m 13[39m [90m"[39mAillyber[90m"[39m    Europa         7
    [90m 14[39m [90m"[39mAiming[90m"[39m      Europa         2
    [90m 15[39m [90m"[39mAinatint[90m"[39m    Europa         3
    [90m 16[39m [90m"[39mAindlylid[90m"[39m   Europa         4
    [90m 17[39m [90m"[39mAinserfle[90m"[39m   Europa        12
    [90m 18[39m [90m"[39mAirdring[90m"[39m    Europa         4
    [90m 19[39m [90m"[39mAivering[90m"[39m    Europa         2
    [90m 20[39m [90m"[39mAlaring[90m"[39m     Europa        10
    [90m 21[39m [90m"[39mAlaxed[90m"[39m      Europa         2
    [90m 22[39m [90m"[39mAlberts[90m"[39m     Earth          7
    [90m 23[39m [90m"[39mAlcemblery[90m"[39m  Europa         5
    [90m 24[39m [90m"[39mAlenat[90m"[39m      Europa         3
    [90m 25[39m [90m"[39mAlenter[90m"[39m     Europa         6
    [90m 26[39m [90m"[39mAlentonway[90m"[39m  Earth          6
    [90m 27[39m [90m"[39mAlest[90m"[39m       Earth          6
    [90m 28[39m [90m"[39mAlfordonard[90m"[39m Earth          8
    [90m 29[39m [90m"[39mAlindiveng[90m"[39m  Europa         4
    [90m 30[39m [90m"[39mAlldson[90m"[39m     Earth         14
    [90m 31[39m [90m"[39mAloquinght[90m"[39m  Europa         3
    [90m 32[39m [90m"[39mAloubtled[90m"[39m   Europa         7
    [90m 33[39m [90m"[39mAlshipson[90m"[39m   Earth          8
    [90m 34[39m [90m"[39mAlutorody[90m"[39m   Europa         1
    [90m 35[39m [90m"[39mAlvasquez[90m"[39m   Earth          8
    [90m 36[39m [90m"[39mAlvercal[90m"[39m    Europa         2
    [90m 37[39m [90m"[39mAlvesssidy[90m"[39m  Europa         2
    [90m 38[39m [90m"[39mAmbleetive[90m"[39m  Europa         8
    [90m 39[39m [90m"[39mAmbleeve[90m"[39m    Europa         5
    [90m 40[39m [90m"[39mAmblereld[90m"[39m   Europa         2
    [90m 41[39m [90m"[39mAmetic[90m"[39m      Europa        12
    [90m 42[39m [90m"[39mAmincrerus[90m"[39m  Europa         5
    [90m 43[39m [90m"[39mAmonsmane[90m"[39m   Europa         2
    [90m 44[39m [90m"[39mAmonysidle[90m"[39m  Europa         1
    [90m 45[39m [90m"[39mAmoutake[90m"[39m    Europa         4
    [90m 46[39m [90m"[39mAmsive[90m"[39m      [31mNA[39m             1
    [90m 47[39m [90m"[39mAmspring[90m"[39m    Europa         2
    [90m 48[39m [90m"[39mAnake[90m"[39m       Mars           2
    [90m 49[39m [90m"[39mAnate[90m"[39m       Mars           8
    [90m 50[39m [90m"[39mAnche[90m"[39m       Mars           3
    [90m 51[39m [90m"[39mAncontaked[90m"[39m  Europa         1
    [90m 52[39m [90m"[39mAncy[90m"[39m        Mars           6
    [90m 53[39m [90m"[39mAndackson[90m"[39m   Earth          8
    [90m 54[39m [90m"[39mAnderking[90m"[39m   Europa         2
    [90m 55[39m [90m"[39mAndley[90m"[39m      Earth          6
    [90m 56[39m [90m"[39mAne[90m"[39m         Mars           4
    [90m 57[39m [90m"[39mAneetle[90m"[39m     Europa         5
    [90m 58[39m [90m"[39mAneter[90m"[39m      Europa         3
    [90m 59[39m [90m"[39mAnindery[90m"[39m    Europa         4
    [90m 60[39m [90m"[39mAnpie[90m"[39m       Mars           6
    [90m 61[39m [90m"[39mAntcal[90m"[39m      Europa         7
    [90m 62[39m [90m"[39mAnthompson[90m"[39m  Earth          9
    [90m 63[39m [90m"[39mAnthon[90m"[39m      Earth          2
    [90m 64[39m [90m"[39mAntoshipson[90m"[39m Earth          7
    [90m 65[39m [90m"[39mApeau[90m"[39m       Mars           5
    [90m 66[39m [90m"[39mApedishaft[90m"[39m  Europa         2
    [90m 67[39m [90m"[39mApenelexy[90m"[39m   Europa         4
    [90m 68[39m [90m"[39mApie[90m"[39m        Mars          11
    [90m 69[39m [90m"[39mApity[90m"[39m       Mars           2
    [90m 70[39m [90m"[39mAppie[90m"[39m       Mars           2
    [90m 71[39m [90m"[39mApple[90m"[39m       Mars           9
    [90m 72[39m [90m"[39mArible[90m"[39m      Europa         7
    [90m 73[39m [90m"[39mArmstromez[90m"[39m  Earth          7
    [90m 74[39m [90m"[39mArner[90m"[39m       Earth          6
    [90m 75[39m [90m"[39mArneras[90m"[39m     Earth          9
    [90m 76[39m [90m"[39mAroodint[90m"[39m    Europa         6
    [90m 77[39m [90m"[39mArterate[90m"[39m    Europa         1
    [90m 78[39m [90m"[39mAsharing[90m"[39m    Europa         1
    [90m 79[39m [90m"[39mAshipeck[90m"[39m    Earth          4
    [90m 80[39m [90m"[39mAshipson[90m"[39m    Earth          7
    [90m 81[39m [90m"[39mAsivetfuel[90m"[39m  Europa         1
    [90m 82[39m [90m"[39mAsolipery[90m"[39m   Europa         4
    [90m 83[39m [90m"[39mAsoppor[90m"[39m     Europa         3
    [90m 84[39m [90m"[39mAssefle[90m"[39m     Europa         4
    [90m 85[39m [90m"[39mAsseple[90m"[39m     Europa         3
    [90m 86[39m [90m"[39mAssibler[90m"[39m    Europa         4
    [90m 87[39m [90m"[39mAsticit[90m"[39m     Europa         3
    [90m 88[39m [90m"[39mAtiveezy[90m"[39m    Europa         3
    [90m 89[39m [90m"[39mAtkinney[90m"[39m    Earth          5
    [90m 90[39m [90m"[39mAusivetpul[90m"[39m  Europa         3
    [90m 91[39m [90m"[39mAvidson[90m"[39m     Earth          5
    [90m 92[39m [90m"[39mAvisley[90m"[39m     Earth          2
    [90m 93[39m [90m"[39mAvisnydes[90m"[39m   Earth          7
    [90m 94[39m [90m"[39mAxlentindy[90m"[39m  Europa         3
    [90m 95[39m [90m"[39mAyalazquez[90m"[39m  Earth          6
    [90m 96[39m [90m"[39mBabre[90m"[39m       Mars           2
    [90m 97[39m [90m"[39mBache[90m"[39m       Mars           5
    [90m 98[39m [90m"[39mBaciffhaut[90m"[39m  Europa         3
    [90m 99[39m [90m"[39mBacistion[90m"[39m   Europa         6
    [90m100[39m [90m"[39mBacke[90m"[39m       Mars           5
    [90m# … with 2,309 more rows[39m
    


```R
#Check remaining missing Values for Homeplanet and replace them based on average fee used by passengers from each HomePlanet
full[is.na(full$HomePlanet),c('fee','HomePlanet')]
full %>% group_by(HomePlanet) %>% summarize(mean(fee))
```


<table class="dataframe">
<caption>A data.frame: 7 × 2</caption>
<thead>
	<tr><th></th><th scope=col>fee</th><th scope=col>HomePlanet</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>235</th><td>   0</td><td>NA</td></tr>
	<tr><th scope=row>808</th><td>   0</td><td>NA</td></tr>
	<tr><th scope=row>2632</th><td>1159</td><td>NA</td></tr>
	<tr><th scope=row>8970</th><td>2607</td><td>NA</td></tr>
	<tr><th scope=row>10584</th><td>6221</td><td>NA</td></tr>
	<tr><th scope=row>11914</th><td>   0</td><td>NA</td></tr>
	<tr><th scope=row>12726</th><td>   0</td><td>NA</td></tr>
</tbody>
</table>




<table class="dataframe">
<caption>A tibble: 4 × 2</caption>
<thead>
	<tr><th scope=col>HomePlanet</th><th scope=col>mean(fee)</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Earth </td><td> 678.9645</td></tr>
	<tr><td>Europa</td><td>3417.1096</td></tr>
	<tr><td>Mars  </td><td>1054.0418</td></tr>
	<tr><td>NA    </td><td>1426.7143</td></tr>
</tbody>
</table>




```R
#REPLACEMENT
full[is.na(full$HomePlanet),c('fee','HomePlanet')]$HomePlanet <- c('Earth','Earth','Mars','Europa','Europa','Earth','Earth')
```


```R
#Could not find valid correlation between columns to replace Destination and Cabin
#replace all NA values to 0
full[is.na(full$Destination),'Destination'] <- '0'
full[is.na(full$Deck_Cabin),'Deck_Cabin'] <- '0'
```


```R
#There should be no remaining NA values otherthan Transported(our dependent variable)
colSums(is.na(full))
```


<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>id</dt><dd>0</dd><dt>PassengerId</dt><dd>0</dd><dt>HomePlanet</dt><dd>0</dd><dt>CryoSleep</dt><dd>0</dd><dt>Destination</dt><dd>0</dd><dt>Age</dt><dd>0</dd><dt>VIP</dt><dd>0</dd><dt>RoomService</dt><dd>0</dd><dt>FoodCourt</dt><dd>0</dd><dt>ShoppingMall</dt><dd>0</dd><dt>Spa</dt><dd>0</dd><dt>VRDeck</dt><dd>0</dd><dt>Transported</dt><dd>4277</dd><dt>Lastname</dt><dd>0</dd><dt>Deck_Cabin</dt><dd>0</dd><dt>Num_Cabin</dt><dd>0</dd><dt>Side_Cabin</dt><dd>0</dd><dt>fee</dt><dd>0</dd></dl>



# EDA
#### Find factors that have a relationship with Transported


```R
#Yes Relation(factor)
grid.arrange(
    ggplot(na.omit(full),aes(x=CryoSleep,fill=Transported)) + geom_bar(position='fill'),
    ggplot(na.omit(full),aes(x=Deck_Cabin,fill=Transported)) + geom_bar(position='fill'),
    ggplot(na.omit(full),aes(x=HomePlanet,fill=Transported)) + geom_bar(position='fill')
    )
```


    
![png](output_20_0.png)
    



```R
#Negative Relation(continuous variable)
grid.arrange(ggplot(na.omit(full),aes(x=fee,fill=Transported)) + geom_histogram(position='dodge'),
             ggplot(na.omit(full),aes(x=VRDeck,fill=Transported)) + geom_histogram(position='fill'),
             ggplot(na.omit(full),aes(x=Spa,fill=Transported)) + geom_histogram(position='fill'),
             ggplot(na.omit(full),aes(x=RoomService,fill=Transported)) + geom_histogram(position='fill'),
             ggplot(na.omit(full),aes(x=Age,fill=Transported)) + geom_histogram(position='fill')
             )
```

    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    
    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    
    Warning message:
    “Removed 18 rows containing missing values (geom_bar).”
    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    
    Warning message:
    “Removed 14 rows containing missing values (geom_bar).”
    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    
    Warning message:
    “Removed 20 rows containing missing values (geom_bar).”
    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    
    


    
![png](output_21_1.png)
    



```R
#Positive Relation(continuous variable)
ggplot(na.omit(full),aes(x=FoodCourt,fill=Transported)) + geom_histogram(position='fill')
```

    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    
    Warning message:
    “Removed 14 rows containing missing values (geom_bar).”
    


    
![png](output_22_1.png)
    



```R
#No relation
grid.arrange(ggplot(na.omit(full),aes(x=VIP,fill=Transported)) + geom_bar(,position='fill'),
             ggplot(na.omit(full),aes(x=Destination,fill=Transported)) + geom_bar(position='fill'),
             ggplot(na.omit(full),aes(x=Side_Cabin,fill=Transported)) + geom_bar(position='fill')
             )
```


    
![png](output_23_0.png)
    


# Dataframe for Modeling


```R
#Create train, test partition and remvoe PassengerID, Last name, id
full.model.train <- full[full$id=='train',!colnames(full) %in% c('id','PassengerId','Lastname')]
full.model.test <- full[full$id=='test',!colnames(full) %in% c('id','PassengerId','Lastname')]
```

# Modeling


```R
#Use Boruta to find which Columns can be removed for the model
set.seed(123)
feature.selection <- Boruta(Transported~., data = full.model.train, doTrace = 1)
fNames <- getSelectedAttributes(feature.selection) #withTentative = TRUE
```

    After 11 iterations, +1.3 mins: 
    
     confirmed 13 attributes: Age, CryoSleep, Deck_Cabin, Destination, fee and 8 more;
    
     rejected 1 attribute: VIP;
    
     no more attributes left.
    
    
    


```R
#Create train/test data set with only necessary features and add Transported as dependent variable
full.model.train.features <- full.model.train[,fNames]
full.model.train.features$Transported <-full.model.train$Transported

full.model.test.features <- full.model.test[,fNames]
full.model.test.features$Transported <-full.model.test$Transported

str(full.model.train.features)
```

    'data.frame':	8693 obs. of  14 variables:
     $ HomePlanet  : chr  "Europa" "Earth" "Europa" "Europa" ...
     $ CryoSleep   : num  0 0 0 0 0 0 0 1 0 1 ...
     $ Destination : chr  "TRAPPIST-1e" "TRAPPIST-1e" "TRAPPIST-1e" "TRAPPIST-1e" ...
     $ Age         : num  39 24 58 33 16 44 26 28 35 14 ...
     $ RoomService : num  0 109 43 0 303 0 42 0 0 0 ...
     $ FoodCourt   : num  0 9 3576 1283 70 ...
     $ ShoppingMall: num  0 25 0 371 151 0 3 0 17 0 ...
     $ Spa         : num  0 549 6715 3329 565 ...
     $ VRDeck      : num  0 44 49 193 2 0 0 0 0 0 ...
     $ Deck_Cabin  : chr  "B" "F" "A" "A" ...
     $ Num_Cabin   : chr  "0" "0" "0" "0" ...
     $ Side_Cabin  : chr  "P" "S" "S" "S" ...
     $ fee         : num  0 736 10383 5176 1091 ...
     $ Transported : Factor w/ 2 levels "True","False": 2 1 2 2 1 1 1 1 1 1 ...
    


```R
#Create model
set.seed(123)
rf <- randomForest(Transported~., data = full.model.train.features, ntree = 200)
plot(rf)
legend('top',colnames(rf$err.rate),fill=1:3)
```


    
![png](output_29_0.png)
    



```R
#Tree numer with minimum error rate
which.min(rf$err.rate[,1])
```


142



```R
#Tune ntree parameter and evaluate model with train data set
set.seed(321)
rf2 <- randomForest(Transported ~., data = full.model.train.features, ntree = 171)
rf2.train.predict <- predict(rf2, newdata = full.model.train.features,type='response')
table(rf2.train.predict,full.model.train.features$Transported)
caret::confusionMatrix(rf2.train.predict,full.model.train.features$Transported)
```


                     
    rf2.train.predict True False
                True  4369   687
                False    9  3628



    Confusion Matrix and Statistics
    
              Reference
    Prediction True False
         True  4369   687
         False    9  3628
                                             
                   Accuracy : 0.9199         
                     95% CI : (0.914, 0.9256)
        No Information Rate : 0.5036         
        P-Value [Acc > NIR] : < 2.2e-16      
                                             
                      Kappa : 0.8397         
                                             
     Mcnemar's Test P-Value : < 2.2e-16      
                                             
                Sensitivity : 0.9979         
                Specificity : 0.8408         
             Pos Pred Value : 0.8641         
             Neg Pred Value : 0.9975         
                 Prevalence : 0.5036         
             Detection Rate : 0.5026         
       Detection Prevalence : 0.5816         
          Balanced Accuracy : 0.9194         
                                             
           'Positive' Class : True           
                                             



```R
#Check Important Factors
varImpPlot(rf2)
```


    
![png](output_32_0.png)
    



```R
#Predict test data set
rf.predict <- predict(rf2, newdata = full.model.test.features,type='response')
rf.predict <- as.data.frame(rf.predict)
rf.predict$PassengerId <- full[full$id=='test','PassengerId']

rf.predict <- rf.predict %>% dplyr::select(PassengerId,Transported=rf.predict)
rf.predict
```


<table class="dataframe">
<caption>A data.frame: 4277 × 2</caption>
<thead>
	<tr><th></th><th scope=col>PassengerId</th><th scope=col>Transported</th></tr>
	<tr><th></th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;fct&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>8694</th><td>0013_01</td><td>True </td></tr>
	<tr><th scope=row>8695</th><td>0018_01</td><td>False</td></tr>
	<tr><th scope=row>8696</th><td>0019_01</td><td>True </td></tr>
	<tr><th scope=row>8697</th><td>0021_01</td><td>True </td></tr>
	<tr><th scope=row>8698</th><td>0023_01</td><td>False</td></tr>
	<tr><th scope=row>8699</th><td>0027_01</td><td>True </td></tr>
	<tr><th scope=row>8700</th><td>0029_01</td><td>True </td></tr>
	<tr><th scope=row>8701</th><td>0032_01</td><td>True </td></tr>
	<tr><th scope=row>8702</th><td>0032_02</td><td>True </td></tr>
	<tr><th scope=row>8703</th><td>0033_01</td><td>True </td></tr>
	<tr><th scope=row>8704</th><td>0037_01</td><td>False</td></tr>
	<tr><th scope=row>8705</th><td>0040_01</td><td>False</td></tr>
	<tr><th scope=row>8706</th><td>0040_02</td><td>False</td></tr>
	<tr><th scope=row>8707</th><td>0042_01</td><td>True </td></tr>
	<tr><th scope=row>8708</th><td>0046_01</td><td>False</td></tr>
	<tr><th scope=row>8709</th><td>0046_02</td><td>False</td></tr>
	<tr><th scope=row>8710</th><td>0046_03</td><td>False</td></tr>
	<tr><th scope=row>8711</th><td>0047_01</td><td>True </td></tr>
	<tr><th scope=row>8712</th><td>0047_02</td><td>True </td></tr>
	<tr><th scope=row>8713</th><td>0047_03</td><td>False</td></tr>
	<tr><th scope=row>8714</th><td>0048_01</td><td>True </td></tr>
	<tr><th scope=row>8715</th><td>0049_01</td><td>False</td></tr>
	<tr><th scope=row>8716</th><td>0054_01</td><td>True </td></tr>
	<tr><th scope=row>8717</th><td>0054_02</td><td>True </td></tr>
	<tr><th scope=row>8718</th><td>0054_03</td><td>False</td></tr>
	<tr><th scope=row>8719</th><td>0055_01</td><td>False</td></tr>
	<tr><th scope=row>8720</th><td>0057_01</td><td>True </td></tr>
	<tr><th scope=row>8721</th><td>0059_01</td><td>True </td></tr>
	<tr><th scope=row>8722</th><td>0060_01</td><td>False</td></tr>
	<tr><th scope=row>8723</th><td>0063_01</td><td>True </td></tr>
	<tr><th scope=row>⋮</th><td>⋮</td><td>⋮</td></tr>
	<tr><th scope=row>12941</th><td>9216_01</td><td>False</td></tr>
	<tr><th scope=row>12942</th><td>9223_01</td><td>True </td></tr>
	<tr><th scope=row>12943</th><td>9223_02</td><td>True </td></tr>
	<tr><th scope=row>12944</th><td>9228_01</td><td>False</td></tr>
	<tr><th scope=row>12945</th><td>9229_01</td><td>False</td></tr>
	<tr><th scope=row>12946</th><td>9232_01</td><td>False</td></tr>
	<tr><th scope=row>12947</th><td>9236_01</td><td>True </td></tr>
	<tr><th scope=row>12948</th><td>9238_01</td><td>False</td></tr>
	<tr><th scope=row>12949</th><td>9238_02</td><td>True </td></tr>
	<tr><th scope=row>12950</th><td>9238_03</td><td>True </td></tr>
	<tr><th scope=row>12951</th><td>9238_04</td><td>False</td></tr>
	<tr><th scope=row>12952</th><td>9238_05</td><td>True </td></tr>
	<tr><th scope=row>12953</th><td>9238_06</td><td>True </td></tr>
	<tr><th scope=row>12954</th><td>9238_07</td><td>False</td></tr>
	<tr><th scope=row>12955</th><td>9240_01</td><td>False</td></tr>
	<tr><th scope=row>12956</th><td>9243_01</td><td>True </td></tr>
	<tr><th scope=row>12957</th><td>9245_01</td><td>False</td></tr>
	<tr><th scope=row>12958</th><td>9249_01</td><td>False</td></tr>
	<tr><th scope=row>12959</th><td>9255_01</td><td>True </td></tr>
	<tr><th scope=row>12960</th><td>9258_01</td><td>True </td></tr>
	<tr><th scope=row>12961</th><td>9260_01</td><td>True </td></tr>
	<tr><th scope=row>12962</th><td>9262_01</td><td>True </td></tr>
	<tr><th scope=row>12963</th><td>9263_01</td><td>True </td></tr>
	<tr><th scope=row>12964</th><td>9265_01</td><td>True </td></tr>
	<tr><th scope=row>12965</th><td>9266_01</td><td>True </td></tr>
	<tr><th scope=row>12966</th><td>9266_02</td><td>True </td></tr>
	<tr><th scope=row>12967</th><td>9269_01</td><td>False</td></tr>
	<tr><th scope=row>12968</th><td>9271_01</td><td>True </td></tr>
	<tr><th scope=row>12969</th><td>9273_01</td><td>True </td></tr>
	<tr><th scope=row>12970</th><td>9277_01</td><td>True </td></tr>
</tbody>
</table>



# Submit Result


```R
#PREDICTION OUTPUT
write.csv(rf.predict, file = "submission.csv",row.names=FALSE,quote=FALSE)
```
