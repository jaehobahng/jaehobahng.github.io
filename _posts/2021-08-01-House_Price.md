```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingClassifier
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Misc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
```

## 1. Import Files(데이터 불러오기)


```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')
```

## 2. Check basic frame of dataframe
## 2. 데이터프레임 기본 틀 확인

#### 행과 칼럼 수 확인


```python
train.shape, test.shape
```




    ((1460, 81), (1459, 80))



#### Print first 5 rows for better understanding of dataset
#### 데이터의 이해를 돕기 위해 첫 5개 행 출력


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   int64  
     1   MSSubClass     1460 non-null   int64  
     2   MSZoning       1460 non-null   object 
     3   LotFrontage    1201 non-null   float64
     4   LotArea        1460 non-null   int64  
     5   Street         1460 non-null   object 
     6   Alley          91 non-null     object 
     7   LotShape       1460 non-null   object 
     8   LandContour    1460 non-null   object 
     9   Utilities      1460 non-null   object 
     10  LotConfig      1460 non-null   object 
     11  LandSlope      1460 non-null   object 
     12  Neighborhood   1460 non-null   object 
     13  Condition1     1460 non-null   object 
     14  Condition2     1460 non-null   object 
     15  BldgType       1460 non-null   object 
     16  HouseStyle     1460 non-null   object 
     17  OverallQual    1460 non-null   int64  
     18  OverallCond    1460 non-null   int64  
     19  YearBuilt      1460 non-null   int64  
     20  YearRemodAdd   1460 non-null   int64  
     21  RoofStyle      1460 non-null   object 
     22  RoofMatl       1460 non-null   object 
     23  Exterior1st    1460 non-null   object 
     24  Exterior2nd    1460 non-null   object 
     25  MasVnrType     1452 non-null   object 
     26  MasVnrArea     1452 non-null   float64
     27  ExterQual      1460 non-null   object 
     28  ExterCond      1460 non-null   object 
     29  Foundation     1460 non-null   object 
     30  BsmtQual       1423 non-null   object 
     31  BsmtCond       1423 non-null   object 
     32  BsmtExposure   1422 non-null   object 
     33  BsmtFinType1   1423 non-null   object 
     34  BsmtFinSF1     1460 non-null   int64  
     35  BsmtFinType2   1422 non-null   object 
     36  BsmtFinSF2     1460 non-null   int64  
     37  BsmtUnfSF      1460 non-null   int64  
     38  TotalBsmtSF    1460 non-null   int64  
     39  Heating        1460 non-null   object 
     40  HeatingQC      1460 non-null   object 
     41  CentralAir     1460 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1460 non-null   int64  
     44  2ndFlrSF       1460 non-null   int64  
     45  LowQualFinSF   1460 non-null   int64  
     46  GrLivArea      1460 non-null   int64  
     47  BsmtFullBath   1460 non-null   int64  
     48  BsmtHalfBath   1460 non-null   int64  
     49  FullBath       1460 non-null   int64  
     50  HalfBath       1460 non-null   int64  
     51  BedroomAbvGr   1460 non-null   int64  
     52  KitchenAbvGr   1460 non-null   int64  
     53  KitchenQual    1460 non-null   object 
     54  TotRmsAbvGrd   1460 non-null   int64  
     55  Functional     1460 non-null   object 
     56  Fireplaces     1460 non-null   int64  
     57  FireplaceQu    770 non-null    object 
     58  GarageType     1379 non-null   object 
     59  GarageYrBlt    1379 non-null   float64
     60  GarageFinish   1379 non-null   object 
     61  GarageCars     1460 non-null   int64  
     62  GarageArea     1460 non-null   int64  
     63  GarageQual     1379 non-null   object 
     64  GarageCond     1379 non-null   object 
     65  PavedDrive     1460 non-null   object 
     66  WoodDeckSF     1460 non-null   int64  
     67  OpenPorchSF    1460 non-null   int64  
     68  EnclosedPorch  1460 non-null   int64  
     69  3SsnPorch      1460 non-null   int64  
     70  ScreenPorch    1460 non-null   int64  
     71  PoolArea       1460 non-null   int64  
     72  PoolQC         7 non-null      object 
     73  Fence          281 non-null    object 
     74  MiscFeature    54 non-null     object 
     75  MiscVal        1460 non-null   int64  
     76  MoSold         1460 non-null   int64  
     77  YrSold         1460 non-null   int64  
     78  SaleType       1460 non-null   object 
     79  SaleCondition  1460 non-null   object 
     80  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    


```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1452.000000</td>
      <td>1460.000000</td>
      <td>...</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>730.500000</td>
      <td>56.897260</td>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
      <td>103.685262</td>
      <td>443.639726</td>
      <td>...</td>
      <td>94.244521</td>
      <td>46.660274</td>
      <td>21.954110</td>
      <td>3.409589</td>
      <td>15.060959</td>
      <td>2.758904</td>
      <td>43.489041</td>
      <td>6.321918</td>
      <td>2007.815753</td>
      <td>180921.195890</td>
    </tr>
    <tr>
      <th>std</th>
      <td>421.610009</td>
      <td>42.300571</td>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>20.645407</td>
      <td>181.066207</td>
      <td>456.098091</td>
      <td>...</td>
      <td>125.338794</td>
      <td>66.256028</td>
      <td>61.119149</td>
      <td>29.317331</td>
      <td>55.757415</td>
      <td>40.177307</td>
      <td>496.123024</td>
      <td>2.703626</td>
      <td>1.328095</td>
      <td>79442.502883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>365.750000</td>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>129975.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>730.500000</td>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>383.500000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>163000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1095.250000</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>166.000000</td>
      <td>712.250000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1460.000000</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>...</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>15500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 38 columns</p>
</div>



### Check for null values
#### The imputation for null values will be performaed later, but an early check for null values provides for a better understanding of the dataset

### Null 값 확인
#### Null 값에 대한 대치는 추후에 진행 될 예정이나 전반적인 데이터셋에 대한 이해를 돕기 위해 간단히 확인 차원에서 진행하였다.


```python
pd.set_option('display.max_rows',None)

train.isnull().sum().sort_values(ascending=False).head(20)
```




    PoolQC          1453
    MiscFeature     1406
    Alley           1369
    Fence           1179
    FireplaceQu      690
    LotFrontage      259
    GarageYrBlt       81
    GarageCond        81
    GarageType        81
    GarageFinish      81
    GarageQual        81
    BsmtFinType2      38
    BsmtExposure      38
    BsmtQual          37
    BsmtCond          37
    BsmtFinType1      37
    MasVnrArea         8
    MasVnrType         8
    Electrical         1
    Id                 0
    dtype: int64



## 3. EDA

#### 3.1Check distribution of dependent variable
#### The distribution is right skewed which makes us think the dpendent variable may have to be normalized for better performance (performed later on)

#### 3.1 종속변수의 분포도 확인
#### 종속변수의 분포도는 오른쪽 고리가 긴 형태를 띄고 있으며 모델링시 성능 개선을 위해 정규화를 진행할 예정입니다(추후에 정규화 실행)


```python
f, ax = plt.subplots(figsize = (10,10))
sns.distplot(train['SalePrice'])
ax.set(title = 'SalePrice Distribution')
sns.despine(left=True)
```

    C:\Users\Jae-Ho Bahng\anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


    
<img src="\assets\images\HousePrice\output_16_1.png" alt="Alt text">
    


#### Use the describe function to pick out only the nominal values.
#### Seperate nominal values from ordinal values for visualization

#### 
#### 수치형과 범주형 변수를 나누기 위해 describe 함수를 사용하여 수치형 변수를 선정한다.
#### 범주형과 연속형 변수를 분리하여 다른방법으로 시각화를 진행한다


```python
num_column = train.describe().columns.to_list()
num_column.remove('SalePrice')
num_column.remove('Id')
# num_column = num_column.remove('SalePrice')
# num_column = num_column.remove('Id')
```

#### 3.2 Visualization for numerical columns(scatter plot)
#### - It seems most nominal values have a positive correlation with the dependent variable "SalePrice"
#### - Variables like "OverQuall" have a high positive correlation while "OpenPorchSF" seems to have little or no correlation
#### 
#### 3.2 수치형 변수의 시각화(산점도 사용)
#### - 연속형 데이터는 보통 종속형 변수와 양의 상관관계가 있다는 것을 알 수 있다.
#### - OverQuall은 거의 정비례 하는 양상을 보이는 반면 OpenPorchSF 처럼 상관관계가 보이지 않는 변수도 다수 있다.


```python
fig, axs = plt.subplots(ncols=3, nrows=int(len(num_column)/3), figsize=(16, 60))

for i,feature in enumerate(list(train[num_column]),1) : 
    plt.subplot(int(len(num_column)/3),3,i)
    sns.scatterplot(x = feature, y = 'SalePrice', data = train)
    plt.tick_params(axis='x', labelsize = 8)
    plt.tick_params(axis='y', labelsize = 8)

plt.show()
```


    
<img src="\assets\images\HousePrice\output_21_0.png" alt="Alt text">
    



```python
train_str = train[train.columns[~train.columns.isin(num_column)]]
train_str = train_str.drop(columns = ['Id'])
```

### 3.3 Visualizaiton for factoral columns
### 3.3 범주형 변수 시각화


```python
str_column = train_str.columns.to_list()
str_column.remove('SalePrice')
#sns.barplot(x='Street',y='SalePrice',data=train_str)
```


```python
order = train.groupby(["MSZoning"])["SalePrice"].mean().sort_values().index


fig, axs = plt.subplots(ncols=3, nrows=round((len(str_column)/3)+0.5), figsize=(16, 100))

for i,feature in enumerate(list(train[str_column]),1) : 
    plt.subplot(round((len(str_column)/3)+0.5),3,i)
    order = train.groupby([feature])["SalePrice"].mean().sort_values(ascending=False).index
    sns.barplot(x = feature, y = 'SalePrice', data = train, order=order)
    plt.tick_params(axis='x', labelsize = 10)
    plt.tick_params(axis='y', labelsize = 8)
    plt.xticks(rotation=90)

plt.show()
```


    
<img src="\assets\images\HousePrice\output_25_0.png" alt="Alt text">


# 4. Feature Engineering

### 4.1 Normalize Dependent variable
#### Since I found out that our dependent variable was right skewed from our EDA, the log1p function will be used to normalize the data
#### 
### 4.1 종속형 변수 정규화
#### EDA 단계에서 종속변수가 우측 꼬리가 긴 양수의 왜도가 있음을 확인하였기에, log1p함수를 사용하여 좌우 대칭을 만들어줄 것이다


```python
f, ax = plt.subplots(figsize = (5,5))
sns.distplot(train['SalePrice'])
ax.set(title = 'SalePrice Distribution')
sns.despine(left=True)
```

    C:\Users\Jae-Ho Bahng\anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


    
<img src="\assets\images\HousePrice\output_28_1.png" alt="Alt text">
    



```python
f, ax = plt.subplots(figsize = (5,5))
sns.distplot(np.log1p(train['SalePrice']))
ax.set(title = 'SalePrice Distribution')
sns.despine(left=True)
```

    C:\Users\Jae-Ho Bahng\anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


    
<img src="\assets\images\HousePrice\output_29_1.png" alt="Alt text">



```python
train["SalePrice"] = np.log1p(train["SalePrice"])
```


```python
# Split features and labels
train_labels = train['SalePrice'].reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test

# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset
all = pd.concat([train_features, test_features]).reset_index(drop=True)
all.shape
```




    (2919, 80)



### 4.2 Imputate missing data
### 4.2 결측치 대치


```python
# Print how many percent of each column is missing
# 칼럼별로 몇 퍼센트의 데이터가 결측치인지 출력

def percent_missing(df) : 
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data))   #All column names as list
    dict_x = {}
    for i in range(0, len(df_cols)) : 
        dict_x.update({df_cols[i] : round(data[df_cols[i]].isnull().mean()*100,2)}) #True/False(0과1의 숫자임으로 평균을 구하면 값이 없는 percentage 구할 수 있음)
        
    return dict_x
```


```python
missing_percentage = percent_missing(all)
missing_percentage_df = sorted(missing_percentage.items(), key = lambda x:x[1], reverse = True)
missing_percentage_df[0:35]
```




    [('PoolQC', 99.66),
     ('MiscFeature', 96.4),
     ('Alley', 93.22),
     ('Fence', 80.44),
     ('FireplaceQu', 48.65),
     ('LotFrontage', 16.65),
     ('GarageYrBlt', 5.45),
     ('GarageFinish', 5.45),
     ('GarageQual', 5.45),
     ('GarageCond', 5.45),
     ('GarageType', 5.38),
     ('BsmtCond', 2.81),
     ('BsmtExposure', 2.81),
     ('BsmtQual', 2.77),
     ('BsmtFinType2', 2.74),
     ('BsmtFinType1', 2.71),
     ('MasVnrType', 0.82),
     ('MasVnrArea', 0.79),
     ('MSZoning', 0.14),
     ('Utilities', 0.07),
     ('BsmtFullBath', 0.07),
     ('BsmtHalfBath', 0.07),
     ('Functional', 0.07),
     ('Exterior1st', 0.03),
     ('Exterior2nd', 0.03),
     ('BsmtFinSF1', 0.03),
     ('BsmtFinSF2', 0.03),
     ('BsmtUnfSF', 0.03),
     ('TotalBsmtSF', 0.03),
     ('Electrical', 0.03),
     ('KitchenQual', 0.03),
     ('GarageCars', 0.03),
     ('GarageArea', 0.03),
     ('SaleType', 0.03),
     ('Id', 0.0)]




```python
len(str_column) + len(num_column)
```




    79




```python
len(all.columns)
```




    80



#### Imputate median for nominal values and mode for factoral values
#### (A more sophisticated imputation could be possible based on the characteristics of each column, but we will skip this part for the simplicity of the code)
#### 수치형 변수는 중앙값으로 / 범주형 변수는 최빈값으로 대체
#### (더 자세한 대치는 칼럼마다 특성을 분석하며 대치해야지만 코드의 간소함을 위해 단순하게 대치)


```python
def missing_fill(data) : 
    for i in num_column : 
        data[i] = data[i].fillna(data[i].median())
    for i in str_column : 
        data[i] = data[i].fillna(data[i].mode()[0])
```


```python
missing_fill(all)
```


```python
all.isnull().sum()[(all.isnull().sum() > 0) == True]
```




    Series([], dtype: int64)



### 4.3 Create Features

### 4.3 신규 변수 생성

We will plot histograms based on our numerical columns to see any paterns or insights we can get to create useful features

이번엔 바플롯이 아닌 연속형 변수에 대한 히스토그램을 생성하여 생성할 수 있는 변수들에 대한 인사이트를 얻어보고자합니다


```python
fig, axs = plt.subplots(ncols=3, nrows=round((len(num_column)/3)+0.5), figsize=(16, 60))

for i,feature in enumerate(list(train[num_column]),1) : 
    plt.subplot(round((len(num_column)/3)+0.5),3,i)
    sns.histplot(all[feature])
    plt.tick_params(axis='x', labelsize = 8)
    plt.tick_params(axis='y', labelsize = 8)

plt.show()
```


    
<img src="\assets\images\HousePrice\output_43_0.png" alt="Alt text">
    


Numerous columns have most values at 0.
For these values we will create a Yes/No column of whether the column is greater than 0 or not.

많은 변수들이 0에 분포가 몰려있는 것을 볼 수 있다.
이러한 변수들은 Yes/No 형식의 칼럼을 생성하여 0보다 크면 전부 1 아니면 0으로 치환할 것이다


```python
all["Pool_YN"] = all['PoolArea'].apply(lambda x : 1 if x > 0 else 0)
all["Garage_YN"] = all['GarageArea'].apply(lambda x : 1 if x > 0 else 0)
all["Misc_YN"] = all['MiscVal'].apply(lambda x : 1 if x > 0 else 0)
all["ScreenPorch_YN"] = all['ScreenPorch'].apply(lambda x : 1 if x > 0 else 0)
all["Bsmt_YN"] = all['TotalBsmtSF'].apply(lambda x : 1 if x > 0 else 0)
all["Fireplace_YN"] = all['Fireplaces'].apply(lambda x : 1 if x > 0 else 0)
```

In addition we will creat new values such as "Total bathrooms" adding all bathrooms in the hosue and "Home_Quality_Cond" by multiplying the two Overall variables in the dataset

그외에도 화장실 수를 합쳐서 총 화장실 수를 구하고 "OverQual" 과 "OverallCond"를 곱하여 집의 전체적인 상태를 확인할 수 있는 변수를 생성할 것이다.


```python
all['Total_Bathrooms'] = (all['FullBath'] + (0.5 * all['HalfBath']) + all['BsmtFullBath'] + (0.5 * all['BsmtHalfBath']))

all['Home_Quality_Cond'] = all['OverallQual'] * all['OverallCond']
```

# 5. Prepare data for modeling
# 5. 모델링을 위한 데이터 준비

### 5.1 Create Dummy Variable
### 5.1 가변수 생성


```python
all = pd.get_dummies(all).reset_index(drop=True)
```


```python
all.shape
```




    (2919, 297)



### 5.2 Redivide "all" dataframe to train/test dataset
### 5.2 train/test 데이터셋으로 "all" 데이터프레임 분할


```python
len(train)
```




    1460




```python
X_train = all.iloc[:len(train),:]
X_test = all.iloc[len(train):,:]

Y_train = train_labels
```


```python
X_train.shape, X_test.shape, Y_train.shape
```




    ((1460, 297), (1459, 297), (1460,))



# 6. Modeling
# 6. 모델링

Create Crossvalidation function and RMSE function

교차검증과 RMSE 함수 생성


```python
# Setup cross validation folds
kf = KFold(n_splits=12, random_state=42, shuffle=True)
```


```python
# Define error metrics
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X_train):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)
```

#### 6.1 Xgboost model with variable importance

#### 6.1 XGboost 모델과 변수별 중요도 출력


```python
xgboost = XGBRegressor(objective='reg:squarederror',random_state=42)

xgboost.fit(X_train,Y_train)

xgb_pred = xgboost.predict(X_train)

print('XGBoost Model RMSE : {}'.format(rmse(Y_train,xgb_pred)))
print('XGBoost Cross Validation RMSE : {}'.format(cv_rmse(xgboost).mean()))
```

    XGBoost Model RMSE : 0.010266704509935385
    XGBoost Cross Validation RMSE : 0.13800160649394327
    


```python
xgb_importance = pd.DataFrame(X_train.columns)
xgb_importance["Value"] = xgboost.feature_importances_

xgb_importance.sort_values(by = "Value", ascending=False).head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>257</th>
      <td>GarageFinish_Unf</td>
      <td>0.187682</td>
    </tr>
    <tr>
      <th>226</th>
      <td>CentralAir_N</td>
      <td>0.112039</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OverallQual</td>
      <td>0.098860</td>
    </tr>
    <tr>
      <th>26</th>
      <td>GarageCars</td>
      <td>0.046649</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Total_Bathrooms</td>
      <td>0.042071</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Fireplaces</td>
      <td>0.039055</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GrLivArea</td>
      <td>0.032751</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Home_Quality_Cond</td>
      <td>0.030084</td>
    </tr>
    <tr>
      <th>242</th>
      <td>Functional_Sev</td>
      <td>0.023773</td>
    </tr>
    <tr>
      <th>12</th>
      <td>TotalBsmtSF</td>
      <td>0.023771</td>
    </tr>
    <tr>
      <th>45</th>
      <td>MSZoning_C (all)</td>
      <td>0.022465</td>
    </tr>
    <tr>
      <th>191</th>
      <td>BsmtQual_Ex</td>
      <td>0.019858</td>
    </tr>
    <tr>
      <th>114</th>
      <td>BldgType_1Fam</td>
      <td>0.015155</td>
    </tr>
    <tr>
      <th>49</th>
      <td>MSZoning_RM</td>
      <td>0.013170</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Neighborhood_Crawfor</td>
      <td>0.013150</td>
    </tr>
    <tr>
      <th>238</th>
      <td>Functional_Maj2</td>
      <td>0.011840</td>
    </tr>
    <tr>
      <th>143</th>
      <td>Exterior1st_BrkComm</td>
      <td>0.010473</td>
    </tr>
    <tr>
      <th>33</th>
      <td>PoolArea</td>
      <td>0.007343</td>
    </tr>
    <tr>
      <th>241</th>
      <td>Functional_Mod</td>
      <td>0.006861</td>
    </tr>
    <tr>
      <th>100</th>
      <td>Condition1_PosA</td>
      <td>0.006477</td>
    </tr>
  </tbody>
</table>
</div>



#### 6.2 RandomForest model with variable importance

#### 6.2 RandomForest 모델 생성과 변수돌 중요도 출력


```python
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train,Y_train)

rf_pred = rf.predict(X_train)

print('RandomForest Model RMSE : {}'.format(rmse(Y_train,rf_pred)))
print('RandomForest Cross Validation RMSE : {}'.format(cv_rmse(rf).mean()))
```

    RandomForest Model RMSE : 0.05332766403383876
    RandomForest Cross Validation RMSE : 0.13833334267800787
    


```python
#Random Forest variable importance
rf_importance = pd.DataFrame(X_train.columns)
rf_importance["Value"] = rf.feature_importances_

rf_importance.sort_values(by = "Value", ascending=False).head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>OverallQual</td>
      <td>0.533961</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GrLivArea</td>
      <td>0.094318</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Total_Bathrooms</td>
      <td>0.042779</td>
    </tr>
    <tr>
      <th>12</th>
      <td>TotalBsmtSF</td>
      <td>0.040787</td>
    </tr>
    <tr>
      <th>26</th>
      <td>GarageCars</td>
      <td>0.037111</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Home_Quality_Cond</td>
      <td>0.036136</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1stFlrSF</td>
      <td>0.023689</td>
    </tr>
    <tr>
      <th>27</th>
      <td>GarageArea</td>
      <td>0.023446</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BsmtFinSF1</td>
      <td>0.016658</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LotArea</td>
      <td>0.011541</td>
    </tr>
    <tr>
      <th>6</th>
      <td>YearBuilt</td>
      <td>0.011078</td>
    </tr>
    <tr>
      <th>7</th>
      <td>YearRemodAdd</td>
      <td>0.006467</td>
    </tr>
    <tr>
      <th>227</th>
      <td>CentralAir_Y</td>
      <td>0.005973</td>
    </tr>
    <tr>
      <th>257</th>
      <td>GarageFinish_Unf</td>
      <td>0.005620</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2ndFlrSF</td>
      <td>0.005573</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LotFrontage</td>
      <td>0.005390</td>
    </tr>
    <tr>
      <th>11</th>
      <td>BsmtUnfSF</td>
      <td>0.004344</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Id</td>
      <td>0.004084</td>
    </tr>
    <tr>
      <th>25</th>
      <td>GarageYrBlt</td>
      <td>0.003872</td>
    </tr>
    <tr>
      <th>254</th>
      <td>GarageType_Detchd</td>
      <td>0.003711</td>
    </tr>
  </tbody>
</table>
</div>



### 6.3 SVR model


```python
# SVR
svr = SVR()
svr.fit(X_train,Y_train)

svr_pred = svr.predict(X_train)

print('SVR Model RMSE : {}'.format(rmse(Y_train,svr_pred)))
print('SVR Cross Validation RMSE : {}'.format(cv_rmse(svr).mean()))
```

    SVR Model RMSE : 0.19838753934609538
    SVR Cross Validation RMSE : 0.2029710849061702
    

### 6.4 Ensemble Models for better prediction
The XGBoost Model has the lowest model RMSE, but with cross validation, random forest and XGBoost have a similar rmse indicating that XGBoost may be slightly overfitted to the trian dataset.
To complement this problem, we will stack models, giving them a percentage out of 100% based on their CV RMSE.

### 6.4 더 좋은 모델링을 위한 앙상블 기법
XGBoost 모델이 모델 자체의 RMSE는 가장 낮게 측정 되었지만 교차검증에서의 RMSE결과는 RandomForest 와 비슷하게 측정되었다.
이것은 즉 XGBoost 모델 자체가 학습 데이터셋에 과적합이 되었을 가능성이 있으므로 CV RMSE 기준으로 모델의 출력값들을 앙상블하여 결과값을 출력할 것이다.


```python
train_pred = 0.5*rf_pred + 0.4*xgb_pred + 0.1*svr_pred
rmse(Y_train,train_pred)
```




    0.04341094212052227



Since the RMSE of the ensemble model is similar to the XGBoost model rmse that we thought may be over fitted, we can tell that the ensemble model is better than a single model output

3개의 모델을 합친 경우가 과적합 가능성이 있었던 XGBoost 모델의 최초 rmse 값과 흡사한 수준으로 나와 더 좋은 모델임을 알 수 있다


```python
#Function to ensemble test dataset

def test_predictions(X) : 
    return(0.5*rf.predict(X) + 0.4*xgboost.predict(X) + 0.1*svr.predict(X))
```

Since we used a log1p function to normalize the dependent variable in the training dataset, we will use expm1 to return the output of the test data set to the original values

테스트 데이터셋에 대한 예측값을 도출하고, log1p를 사용했기 때문에 expm1 함수로 원래 분포로 종속변수를 돌려놓고 분포도를 확인한다


```python
test_pred = test_predictions(X_test)
```


```python
#Undo log1p function
submission_pred = np.expm1(test_pred)
```


```python
sns.distplot(submission_pred)
```

    C:\Users\Jae-Ho Bahng\anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    




    <AxesSubplot:ylabel='Density'>




    
<img src="\assets\images\HousePrice\output_75_2.png" alt="Alt text">
    



```python
submission = pd.read_csv('sample_submission.csv')
```


```python
submission.shape
```




    (1459, 2)




```python
submission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>169277.052498</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>187758.393989</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>183583.683570</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>179317.477511</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>150730.079977</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission["SalePrice"] = submission_pred
```


```python
submission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>122111.928807</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>154136.830049</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>178468.004852</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>181603.697373</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>177678.596664</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission.to_csv("submission_house_price.csv", index=False)
```
