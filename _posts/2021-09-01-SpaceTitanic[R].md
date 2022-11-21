```R
library(dplyr)
library(ggplot2)
library(stringr)
library(caret)
library(mice)
library(xgboost)
library(gridExtra)
library(InformationValue)
```

    
    Attaching package: ‘dplyr’
    
    
    The following objects are masked from ‘package:stats’:
    
        filter, lag
    
    
    The following objects are masked from ‘package:base’:
    
        intersect, setdiff, setequal, union
    
    
    Loading required package: lattice
    
    
    Attaching package: ‘caret’
    
    
    The following object is masked from ‘package:httr’:
    
        progress
    
    
    
    Attaching package: ‘mice’
    
    
    The following object is masked from ‘package:stats’:
    
        filter
    
    
    The following objects are masked from ‘package:base’:
    
        cbind, rbind
    
    
    
    Attaching package: ‘xgboost’
    
    
    The following object is masked from ‘package:dplyr’:
    
        slice
    
    
    
    Attaching package: ‘gridExtra’
    
    
    The following object is masked from ‘package:dplyr’:
    
        combine
    
    
    
    Attaching package: ‘InformationValue’
    
    
    The following objects are masked from ‘package:caret’:
    
        confusionMatrix, precision, sensitivity, specificity
    
    
    

# Load and bind datasets


```R
# Load datasets
train <- read.csv("../input/spaceship-titanic/train.csv")
test <- read.csv("../input/spaceship-titanic/test.csv")

names(train)
names(test)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'PassengerId'</li><li>'HomePlanet'</li><li>'CryoSleep'</li><li>'Cabin'</li><li>'Destination'</li><li>'Age'</li><li>'VIP'</li><li>'RoomService'</li><li>'FoodCourt'</li><li>'ShoppingMall'</li><li>'Spa'</li><li>'VRDeck'</li><li>'Name'</li><li>'Transported'</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'PassengerId'</li><li>'HomePlanet'</li><li>'CryoSleep'</li><li>'Cabin'</li><li>'Destination'</li><li>'Age'</li><li>'VIP'</li><li>'RoomService'</li><li>'FoodCourt'</li><li>'ShoppingMall'</li><li>'Spa'</li><li>'VRDeck'</li><li>'Name'</li></ol>




```R
# Bind train/test datasets
test$Transported <- NA
full <- bind_rows(train,test,.id='id')
full$id <- ifelse(full$id=='1','train','test')

str(full)
```

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
#### 1. Change Blank cells to NA value
#### 2. CryoSleep : True,False to 1,0
#### 3. VIP : True,False to 1,0
#### 4. Transported to factors


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

# 2. Cabin Separation(Deck/Num/Side) and drop original Cabin Column
full$Deck_Cabin <- str_split(full$Cabin,'/',simplify=TRUE)[,1]
full$Num_Cabin <- str_split(full$Cabin,'/',simplify=TRUE)[,2]
full$Side_Cabin <- str_split(full$Cabin,'/',simplify=TRUE)[,3]
full$Cabin <- NULL

# 3. Create column for sum of all fees used by customer
#    - Substitude missing fee values to 0
fee <- c('RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck')
for(i in fee){
  full[is.na(full[,i]),i] <- 0
}

full$fee <- rowSums(full[,fee])

# Check Dataset
str(full)
```

# Replace Missing Values
#### 1. CryoSleep(based on fee)
#### 2. Mice to replace CryoSleep/Age/VIP
#### 3. HomePlanet based on Lastname
#### 4. Remaining NA values


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




```R
# Left NA values
sum((is.na(full$CryoSleep)))
```


136



```R
# 2. Use Mice to replace columns with integers(CryoSleep, Age, VIP)
library(mice)
mice_mod <- mice(full[,names(full) %in% c('Age','fee','VIP','CryoSleep','Destination','HomePlanet','Deck_Cabin','CryoSleep')],
                 method = 'rf')
```


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


    
![png](output_13_0.png)
    



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
<caption>A data.frame: 0 × 2</caption>
<thead>
	<tr><th scope=col>fee</th><th scope=col>HomePlanet</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
</tbody>
</table>




<table class="dataframe">
<caption>A tibble: 3 × 2</caption>
<thead>
	<tr><th scope=col>HomePlanet</th><th scope=col>mean(fee)</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Earth </td><td> 678.5778</td></tr>
	<tr><td>Europa</td><td>3417.7334</td></tr>
	<tr><td>Mars  </td><td>1054.0799</td></tr>
</tbody>
</table>




```R
#Replace NA values manually
full[is.na(full$HomePlanet),c('fee','HomePlanet')]$HomePlanet <- c('Earth','Earth','Mars','Europa','Europa','Earth','Earth')
```


    Error in `$<-.data.frame`(`*tmp*`, HomePlanet, value = c("Earth", "Earth", : replacement has 7 rows, data has 0
    Traceback:
    

    1. `$<-`(`*tmp*`, HomePlanet, value = c("Earth", "Earth", "Mars", 
     . "Europa", "Europa", "Earth", "Earth"))

    2. `$<-.data.frame`(`*tmp*`, HomePlanet, value = c("Earth", "Earth", 
     . "Mars", "Europa", "Europa", "Earth", "Earth"))

    3. stop(sprintf(ngettext(N, "replacement has %d row, data has %d", 
     .     "replacement has %d rows, data has %d"), N, nrows), domain = NA)



```R
# Remaining NA values
#    Could not find valid correlation between columns to replace Destination and Cabin
#    replace all NA values to 0
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
#### Find factors that have a correlation with Transported


```R
#Yes Relation(factor)
grid.arrange(
  ggplot(na.omit(full),aes(x=CryoSleep,fill=Transported)) + geom_bar(position='fill'),
  ggplot(na.omit(full),aes(x=Deck_Cabin,fill=Transported)) + geom_bar(position='fill'),
  ggplot(na.omit(full),aes(x=HomePlanet,fill=Transported)) + geom_bar(position='fill')
)
```


    
![png](output_22_0.png)
    



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
    
    


    
![png](output_23_1.png)
    



```R
#Positive Relation(continuous variable)
ggplot(na.omit(full),aes(x=FoodCourt,fill=Transported)) + geom_histogram(position='fill')
```

    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    
    Warning message:
    “Removed 14 rows containing missing values (geom_bar).”
    


    
![png](output_24_1.png)
    



```R
#No relation
grid.arrange(ggplot(na.omit(full),aes(x=VIP,fill=Transported)) + geom_bar(,position='fill'),
             ggplot(na.omit(full),aes(x=Destination,fill=Transported)) + geom_bar(position='fill'),
             ggplot(na.omit(full),aes(x=Side_Cabin,fill=Transported)) + geom_bar(position='fill')
)
```


    
![png](output_25_0.png)
    


# Dataframe for Modeling
#### 1. Create dummy variables
#### 2. Drop unnecessary columns
#### 3. Sclae continuous variables
#### 4. Divide train/test data sets


```R
# 1. Create dummy variables

# 1.1) Create dummy variables data sets
encoder <- dummyVars(~HomePlanet + Destination + Deck_Cabin + Side_Cabin, 
                    data = full, sep = '.')
dummy <- predict(encoder, newdata = full)

# 1.2) Drop dummy variable columns from original data set
dummy.names <- c('HomePlanet','CryoSleep','Destination','Deck_Cabin','Side_Cabin','Num_Cabin')

full.dummy <- full[!colnames(full) %in% dummy.names]

# 1.3) Join original data set + dummy variable dataset
full.dummy <- cbind(full.dummy,dummy)
```


```R
# 2. Drop unnecessary columns
full.dummy.model <- full.dummy[,!colnames(full.dummy) %in% c('PassengerId','Lastname')]


# 3. Scale continuous variables
scale <- c('Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','fee')
full.dummy.model[,scale] <- sapply(full.dummy.model[,scale],scale)


# 4. Divide train/test data sets
train.full.dummy.model <- full.dummy.model[full.dummy.model$id=='train',-1]
test.full.dummy.model <- full.dummy.model[full.dummy.model$id=='test',-1]
```

# Modeling


```R
# Set hyperparameter range to train xgboost model
grid = expand.grid(
  nrounds = c(75, 100),
  colsample_bytree = 1,    #기본값 1
  min_child_weight = 1,    #기본값 1
  eta = c(0.01, 0.1, 0.3), #기본값 0.3
  gamma = c(0.5, 0.25),    
  subsample = 0.5,         #기본값 1
  max_depth = c(2, 3)      
)

cntrl = caret::trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "final"                                                        
)
```


```R
# Train XGB model to find optimal hyperparameters
set.seed(1)
train.xgb = caret::train(
  x = train.full.dummy.model[,-8],
  y = train.full.dummy.model[, 8],
  trControl = cntrl,
  tuneGrid = grid,
  method = "xgbTree"
)
```


```R
train.xgb
```


    eXtreme Gradient Boosting 
    
    No pre-processing
    Resampling: Cross-Validated (5 fold) 
    Summary of sample sizes: 6955, 6955, 6954, 6954, 6954 
    Resampling results across tuning parameters:
    
      eta   max_depth  gamma  nrounds  Accuracy   Kappa    
      0.01  2          0.25    75      0.7399061  0.4803699
      0.01  2          0.25   100      0.7428978  0.4863249
      0.01  2          0.50    75      0.7423227  0.4851630
      0.01  2          0.50   100      0.7407113  0.4819726
      0.01  3          0.25    75      0.7751067  0.5502633
      0.01  3          0.25   100      0.7763717  0.5528111
      0.01  3          0.50    75      0.7710798  0.5422630
      0.01  3          0.50   100      0.7723449  0.5448150
      0.10  2          0.25    75      0.7940885  0.5879656
      0.10  2          0.25   100      0.7946636  0.5890437
      0.10  2          0.50    75      0.7910969  0.5819518
      0.10  2          0.50   100      0.7922474  0.5842038
      0.10  3          0.25    75      0.7959296  0.5915594
      0.10  3          0.25   100      0.7989213  0.5975715
      0.10  3          0.50    75      0.7951234  0.5899062
      0.10  3          0.50   100      0.7989203  0.5975729
      0.30  2          0.25    75      0.7980008  0.5957331
      0.30  2          0.25   100      0.7998409  0.5994654
      0.30  2          0.50    75      0.7930540  0.5858306
      0.30  2          0.50   100      0.7981154  0.5959808
      0.30  3          0.25    75      0.7989195  0.5977279
      0.30  3          0.25   100      0.8011044  0.6021094
      0.30  3          0.50    75      0.7989208  0.5977067
      0.30  3          0.50   100      0.8006452  0.6011541
    
    Tuning parameter 'colsample_bytree' was held constant at a value of 1
    
    Tuning parameter 'min_child_weight' was held constant at a value of 1
    
    Tuning parameter 'subsample' was held constant at a value of 0.5
    Accuracy was used to select the optimal model using the largest value.
    The final values used for the model were nrounds = 100, max_depth = 3, eta
     = 0.3, gamma = 0.25, colsample_bytree = 1, min_child_weight = 1 and
     subsample = 0.5.



```R
# Set optimal hyperparameters
param <- list(  objective           = "binary:logistic",     #회귀면 reg:linear
                booster             = "gbtree",
                eval_metric         = "error",                   #회귀 예시에는 없었음
                eta                 = 0.3, 
                max_depth           = 3, 
                subsample           = 0.5,
                colsample_bytree    = 1,
                gamma               = 0.25,
                min_child_weight = 1
)
```


```R
# Matrix for XGBoost function
x <- as.matrix(train.full.dummy.model[, -8])
y <- ifelse(train.full.dummy.model$Transported == 'True', 1, 0)
train.mat <- xgboost::xgb.DMatrix(data = x, 
                                  label = y)
```


```R
# Create model with optimal hyperparameters
set.seed(1)
xgb.fit <- xgb.train(params = param, data = train.mat, nrounds = 75)
xgb.fit
```


    ##### xgb.Booster
    raw: 93.6 Kb 
    call:
      xgb.train(params = param, data = train.mat, nrounds = 75)
    params (as set within xgb.train):
      objective = "binary:logistic", booster = "gbtree", eval_metric = "error", eta = "0.3", max_depth = "3", subsample = "0.5", colsample_bytree = "1", gamma = "0.25", min_child_weight = "1", validate_parameters = "TRUE"
    xgb.attributes:
      niter
    callbacks:
      cb.print.evaluation(period = print_every_n)
    # of features: 27 
    niter: 75
    nfeatures : 27 



```R
# Predict/evaluate  using train dataset
pred <- predict(xgb.fit, x)

Cutoff <- optimalCutoff(y,pred)
caret::confusionMatrix(as.factor(y),as.factor(ifelse(pred>=Cutoff,1,0)))
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction    0    1
             0 3566  749
             1  748 3630
                                              
                   Accuracy : 0.8278          
                     95% CI : (0.8197, 0.8357)
        No Information Rate : 0.5037          
        P-Value [Acc > NIR] : <2e-16          
                                              
                      Kappa : 0.6556          
                                              
     Mcnemar's Test P-Value : 1               
                                              
                Sensitivity : 0.8266          
                Specificity : 0.8290          
             Pos Pred Value : 0.8264          
             Neg Pred Value : 0.8291          
                 Prevalence : 0.4963          
             Detection Rate : 0.4102          
       Detection Prevalence : 0.4964          
          Balanced Accuracy : 0.8278          
                                              
           'Positive' Class : 0               
                                              



```R
# Predict for test data set
testx <- as.matrix(test.full.dummy.model[, -8])
xgb.test <- predict(xgb.fit, testx)
Transported <- factor(ifelse(xgb.test>=Cutoff,'True','False'),levels=c('True','False'))
Transported <- as.data.frame(Transported)
Transported$PassengerId <- full[full$id=='test','PassengerId']

xgb.test.result <- Transported %>% dplyr::select(PassengerId,Transported)
```


```R
# Submit File
write.csv(xgb.test.result, file = "submission.csv",row.names=FALSE,quote=FALSE)
```
