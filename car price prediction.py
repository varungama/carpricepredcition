#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from statistics import *
import warnings
warnings.filterwarnings('ignore')
d=pd.read_csv('E:\\cartrue.csv')


# In[2]:


d1 = d.drop(['Unnamed: 0', 'Description','ExteriorColor','InteriorColor','FuelType','CabType','BedLength','City'], axis = 1)


# In[3]:


d1['Price']=d1['Price'].str.replace("$","")
d1['Price']=d1['Price'].str.replace(",","")
d1['Miles']=d1['Miles'].str.replace(",","")


# In[4]:


accident =[]
owner=[]
usetype=[]
for q in range(0,9993):
    con = d['Condition'][q].split(",")
    if len(con)==5:
        can1 = con[0].replace("'","")
        acc = can1.replace("[","")
        accident.append(acc)
        own = con[2].replace(" '","")
        owner.append(own)
        can2 = con[4].replace(" '","")
        can3 = can2.replace("'","")
        ust = can3.replace("]","")
        usetype.append(ust)
    else:
        can1 = con[0].replace("'","")
        acc = can1.replace("[","")
        accident.append(acc)
        own = None
        owner.append(own)
        can2 = con[2].replace(" '","")
        can3 = can2.replace("'","")
        ust = can3.replace("]","")
        usetype.append(ust)


d1['Accidents']=accident
d1['NoOfOwners']=owner
d1['UseType']=usetype


# In[5]:


d1 = d1.drop(['Condition'], axis = 1)
d2=d1


# In[6]:


for v in range(0,9993):
    if d1['MPG'][v]=='6.2L V-8 Gas' or d1['MPG'][v]=='1.5L Inline-4 Plug-In Hybrid':
        d2['DriveType'][v]=d2['Engine'][v]
        d2['Engine'][v]=d2['MPG'][v]
        d2['MPG'][v]=None
    elif d1['MPG'][v]=='4.3L V-6 Gas' or d1['MPG'][v]=='6.7L V-8 Diesel Turbocharged':
        d2['DriveType'][v]=d2['Engine'][v]
        d2['Engine'][v]=d2['MPG'][v]
        d2['MPG'][v]=None
    elif d1['MPG'][v]=='6.0L V-8 Gas' or d1['MPG'][v]=='2.0L Inline-4 Hybrid Turbocharged':
        d2['DriveType'][v]=d2['Engine'][v]
        d2['Engine'][v]=d2['MPG'][v]
        d2['MPG'][v]=None
    elif d1['MPG'][v]=='2.0L Inline-4 Plug-In Hybrid' or d1['MPG'][v]=='6.4L V-8 Gas':
        d2['DriveType'][v]=d2['Engine'][v]
        d2['Engine'][v]=d2['MPG'][v]
        d2['MPG'][v]=None
    elif d1['MPG'][v]=='L - Hydrogen' or d1['MPG'][v]=='6.6L V-8 Diesel Turbocharged':
        d2['DriveType'][v]=d2['Engine'][v]
        d2['Engine'][v]=d2['MPG'][v]
        d2['MPG'][v]=None
    elif d1['MPG'][v]=='2.0L Inline-4 Plug-In Hybrid Turbocharged' or d1['MPG'][v]=='6.7L V-6 Diesel Turbocharged':
        d2['DriveType'][v]=d2['Engine'][v]
        d2['Engine'][v]=d2['MPG'][v]
        d2['MPG'][v]=None
    elif d1['MPG'][v]=='2.1L Inline-4 Diesel Turbocharged' or d1['MPG'][v]=='1.4L Inline-4 Plug-In Hybrid':
        d2['DriveType'][v]=d2['Engine'][v]
        d2['Engine'][v]=d2['MPG'][v]
        d2['MPG'][v]=None
    elif d1['MPG'][v]=='5.7L V-8 Gas' or d1['MPG'][v]=='3.0L V-6 Plug-In Hybrid Turbocharged':
        d2['DriveType'][v]=d2['Engine'][v]
        d2['Engine'][v]=d2['MPG'][v]
        d2['MPG'][v]=None
    elif d1['MPG'][v]=='3.0L V-6 Diesel Turbocharged' or d1['MPG'][v]=='6.8L V-10 Gas':
        d2['DriveType'][v]=d2['Engine'][v]
        d2['Engine'][v]=d2['MPG'][v]
        d2['MPG'][v]=None
    elif d1['MPG'][v]=='7.2L V-8 Gas':
        d2['DriveType'][v]=d2['Engine'][v]
        d2['Engine'][v]=d2['MPG'][v]
        d2['MPG'][v]=None


# In[7]:


mpg= d2['MPG'].str.split("/", expand=True)
d2['MPG_cty']=mpg[0]
d2['MPG_hwy']=mpg[1]
d2 = d2.drop(['MPG'], axis = 1)
d2['MPG_cty']=d1['MPG_cty'].str.replace(" cty","")
d2['MPG_hwy']=d1['MPG_hwy'].str.replace(" hwy","")


# In[8]:


eng = d2['Engine'].str.split("L", expand=True)
de = eng[1]
gas = de.str.split("Gas", expand=True)
for e in range(0,9993):
    if eng[0][e]=='':
        eng[0][e]='0.5'

d2['Engine_L']=eng[0]
d2['Engine_Gas']=gas[0]
d2=d2.drop(['Engine'], axis = 1)


# In[9]:


for t in range(0,9993):
    if d1['Transmission'][t]=='Crew Cab' or d1['Transmission'][t]=='Standard':
        d2['Transmission'][t]=d['FuelType'][t]
    elif d1['Transmission'][t]=='Extended Cab' or d1['Transmission'][t]=='Regular Cab':
        d2['Transmission'][t]=d['FuelType'][t]


# In[10]:


type(d2['MPG_cty'][0])


# In[11]:


d2['CarBrand']=d2['CarBrand'].str.lower()
#d2['City']=d2['City'].str.lower()
d2['State']=d2['State'].str.lower()
d2['ExteColor']=d2['ExteColor'].str.lower()
d2['InterColor']=d2['InterColor'].str.lower()
d2['style']=d2['style'].str.lower()
d2['Transmission']=d2['Transmission'].str.lower()
d2['UseType']=d2['UseType'].str.lower()
d2['Engine_Gas']=d2['Engine_Gas'].str.lower()
d2['Model']=d2['Model'].str.lower()
d3=d2


# In[12]:


d2.replace([None],np.nan,inplace=True)


# In[13]:


Transmission_map = {'automatic':1,
                    'manual':0,
}

NoOfOwners_map = {'1 Owner':8,
              '2 Owners':7,
              '3 Owners':6,
              '4 Owners':5,
              '5 Owners':4,
              '6 Owners':3,
              '7 Owners':2,
              '8 Owners':1,
              '9 Owners':0,
}

d3['Transmission']=d3.Transmission.map(Transmission_map)
d3['NoOfOwners']=d3.NoOfOwners.map(NoOfOwners_map)


# In[14]:


# KNN_imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)
d3_imputed = imputer.fit_transform(d3[['Transmission', 'NoOfOwners', 'MPG_cty', 'MPG_hwy']])
d3_imputed


# In[15]:


d3_imputed.tolist()
Tran_imp=[]
NoOf_imp=[]
MPG_imp=[]
MPGimp=[]
for i in range(0,9993):
    Tran_imp.append(round(d3_imputed[i][0]))
    NoOf_imp.append(round(d3_imputed[i][1]))
    MPG_imp.append(round(d3_imputed[i][2]))
    MPGimp.append(round(d3_imputed[i][3]))
d3['Transmission']=Tran_imp
d3['NoOfOwners']=NoOf_imp
d3['MPG_cty']=MPG_imp
d3['MPG_hwy']=MPGimp

d3['fueleconomy'] =round((0.55 * d3['MPG_cty']) + (0.45 * d3['MPG_hwy']),2)
d3 = d3.drop(['MPG_cty','MPG_hwy'], axis = 1)


# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.boxplot(d3['Miles'])
# plt.title('Boxplot')
# plt.show()
# 
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.boxplot(d3['fueleconomy'])
# plt.title('Boxplot')
# plt.show()

# In[16]:


d3['Price']=d3['Price'].astype('int64')
d3['Miles']=d3['Miles'].astype('int64')
d3['Year']=d3['Year'].astype('string')
d3['NoOfOwners']=d3['NoOfOwners'].astype('string')
#d3['MPG_cty']=d3['MPG_cty'].astype('int')
#d3['MPG_hwy']=d3['MPG_hwy'].astype('int')
d3['Engine_L']=d3['Engine_L'].astype('string')
d3['Transmission']=d3['Transmission'].astype('string')

d3['Year']=d3['Year'].astype('object')
d3['Engine_L']=d3['Engine_L'].astype('object')
d3['NoOfOwners']=d3['NoOfOwners'].astype('object')
d3['Transmission']=d3['Transmission'].astype('object')
#d3['MPG_cty']=d3['MPG_cty'].astype('object')
#d3['MPG_hwy']=d3['MPG_hwy'].astype('object')
#d3.isnull().sum()


# In[17]:


d3.columns


# In[ ]:





# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# 
# scaler = MinMaxScaler()
# num_vars = ['Miles']
# d3[num_vars] = scaler.fit_transform(d3[num_vars])

# In[18]:


type(d3.fueleconomy[0])


# In[19]:


from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train, df_test = train_test_split(d3, train_size = 0.8, test_size = 0.2, random_state = 100)


# In[20]:


y_train = df_train.pop('Price')
X_train = df_train
y_test = df_test.pop('Price')
X_test = df_test


# In[21]:


df_train.shape


# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.boxplot(X_train['Miles'])
# plt.title('Boxplot')
# plt.show()
# 
# 
# sns.boxplot(X_train['fueleconomy'])
# plt.title('Boxplot')
# plt.show()

# In[22]:


from feature_engine.outliers import Winsorizer
import seaborn as sns
import matplotlib.pyplot as plt
for i in X_train:
    if X_train[i].dtype=="object":
        continue
    else:
        windsoriser = Winsorizer(capping_method='gaussian',tail='both',fold=1.5,variables=i)
        X_train[i]= windsoriser.fit_transform(X_train[[i]])

        # we can inspect the minimum caps and maximum caps
        windsoriser.right_tail_caps_,windsoriser.left_tail_caps_

        # lets see boxplot
        sns.boxplot(X_train[i])
        plt.title('Boxplot')
        plt.show()


# In[23]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

scaler = FunctionTransformer(np.log2, validate = True)
num_vars = ['Miles']
X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
X_test[num_vars] = scaler.fit_transform(X_test[num_vars])


# In[24]:


X_train.Miles[0]


# In[42]:


catego=['CarBrand', 'Model','Year','State','ExteColor','InterColor','style', 'DriveType', 'Accidents', 'UseType', 'Engine_Gas']
from catboost import CatBoostRegressor

catboo = CatBoostRegressor(iterations=99,
                            random_state = 2021, od_type = 'Iter',
                            eval_metric="R2",learning_rate=0.085,depth=16,l2_leaf_reg=5,bagging_temperature=1
                            ,border_count=255,grow_policy='Lossguide',max_leaves=500)
catboo.fit(X_train, y_train,cat_features=catego,eval_set=(X_test, y_test),plot=True)


# In[43]:


from sklearn.metrics import r2_score
x_pred = catboo.predict(X_train)
r2_score(y_train,x_pred)


# In[44]:


y_pred = catboo.predict(X_test)
r2_score(y_test,y_pred)


# In[45]:


params = {'depth':[3,1,2,6,4,5,7,8,9,10],
          'iterations':[250,100,500],
          'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
          'l2_leaf_reg':[3,1,5,10,100],
          'border_count':[32,5,10,20,50],
          'thread_count':[4]}


# grid_cat = GridSearchCV(estimator = catboo, param_grid = params, scoring="neg_mean_squared_error", cv = 3, verbose = 2)
# grid_cat.fit(X_train, y_train, cat_features=catego,eval_set=(X_test, y_test),plot=True)

# grid_cat.best_estimator_

# In[46]:


import pickle
filename = 'prediction'
pickle.dump(catboo,open(filename,'wb'))


# In[47]:


(0.55*20)+(0.45*26)


# In[51]:


out = catboo.predict(np.array([['toyota', 'highlander', '2019', 10.3000, 'tx', 'blue',
       'black', 'suv', 'FWD', 1, 'No accidents',
       8, 'personal use', '3.0', 'inline-4', 22.7]]))


# In[52]:


out[0]


# In[33]:


pd.DataFrame()
d3['Miles'] = scaler.fit_transform(d3[['Miles']])


# In[34]:


d3['Miles'][0]


# In[35]:


plt.scatter(y_pred,y_test,color="blue")
plt.plot(x_pred,y_train,color="red")
x_pred = x_pred.reshape(-1,1)


# In[ ]:





# In[36]:


type('Miles')


# In[37]:


model = pickle.load(open('pricepred','rb'))


# In[ ]:


out1 = model.predict(np.array([['volvo', 'xc60', '2018', 40670, 'tx', 'white',
       'black', 'suv', 'AWD', '1', 'No accidents',
       '7', 'personal use', '2.0', 'inline-4', '23.700000']]))


# In[ ]:


out1


# In[ ]:




