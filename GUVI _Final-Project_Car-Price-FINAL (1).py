#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary library

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ## Loading the data and making a copy

# In[2]:


#Taking a copy so that the original dataset is not affected
df = pd.read_csv("I://SSSK//Data Science-D55//Project - GUVI//Final project//Car_price//cars_price(copy).csv")
raw_data = df.copy()


# In[3]:


#Many of the observations are filled with the symbol "?". We can replace it with blanks.
raw_data.replace("?","NaN",inplace=True)
raw_data


# In[4]:


#Checking the statistics of each column to identify any data types other than integers or floats. Please note regression models can only be constructed with numerical (int or float) values.
raw_data.describe(include="all")


# In[5]:


#checking data types of each column
raw_data.dtypes


# ### Handling null values

# In[6]:


#Prior to commencing the data type conversion process, it is imperative to verify the absence of any null values within the dataset.


# In[7]:


# .isna() method is not picking up the NaN values. Hence I have used .isin(["NaN"]) method
raw_data.isin(["NaN"])


# In[8]:


#Finding the number of null values present in the dataset.
raw_data.isin(["NaN"]).sum()


# ## Dropping missing values

# In[9]:


# The target variable in this dataset is 'Price,' and there are four observations with missing values. It is safe to remove these entire rows since they do not contribute to the prediction.
# Originally, the dataset comprised 206 rows. Following the removal of rows with null values, there are now 193 rows. This removal is deemed acceptable as long as it remains below 6% of the total data.
# raw_data = raw_data[~(raw_data[column_to_check] == 'specific_text_to_remove','specific_text_to_remove')]
#raw_data_NO_mv = raw_data.dropna(axis=0) - Not worked


raw_data = raw_data[~(raw_data["num-of-doors"] == 'NaN')]
raw_data = raw_data[~(raw_data["bore"] == 'NaN')]
raw_data = raw_data[~(raw_data["stroke"] == 'NaN')]
raw_data = raw_data[~(raw_data["horsepower"] == 'NaN')]
raw_data = raw_data[~(raw_data["peak-rpm"] == 'NaN')]
raw_data = raw_data[~(raw_data["price"] == 'NaN')]
raw_data


# In[10]:


raw_data.isin(["NaN"]).sum()


# In[11]:


#Dropping the column 'normalized-losses' which does not have close to 20% of the data. 
data_no_mv = raw_data.drop("normalized-losses", axis=1)
data_no_mv.describe(include="all")


# In[12]:


# Now we have do not have any blank data. the dataset is ready for preprocessing.
data_no_mv.isin(["NaN"]).sum()


# ## Handling inconsistent data types

# In[13]:


#Checking the data types of the all the variables
data_no_mv.dtypes


# ### Explaining the complex features
# 
# Symboling:
# 
# Symboling is a measure of the risk associated with the car. It's often used as a representation of the car's insurance risk rating. A higher symboling value indicates a higher risk.
# 
# Normalized-losses:
# 
# This feature represents the relative average loss payment per insured vehicle year. The values are normalized, meaning they are represented on a scale from 0 to 1. A higher normalized-losses value may indicate a higher likelihood of insurance losses.
# 
# Aspiration:
# 
# Aspiration refers to the type of air intake system the car's engine has. It can be either "std" (standard) or "turbo" (turbocharged). Turbocharged engines generally provide more power.
# 
# Bore:
# 
# Bore is the diameter of each cylinder in the engine. It is one of the parameters that determine the engine's displacement and, consequently, its performance characteristics.
# 
# Stroke:
# 
# Stroke is the distance the piston travels inside the cylinder. Together with the bore, it helps determine the engine's displacement and overall performance.
# 
# Compression-ratio:
# 
# The compression ratio is a measure of how much the air-fuel mixture is compressed inside the engine's cylinders before ignition. It is an important factor influencing engine efficiency and performance. Higher compression ratios often lead to more powerful engines.

# ### Building upon the preceding explanation, I've introduced a new column containing the actual data type details for each variable. This analysis was conducted using Excel.
# 
# ![image-5.png](attachment:image-5.png)

# #### Since we have multiple categorial variables, we to first assign dummies before we start changing the data types

# ## Assigining Dummy variables

# In[14]:


data_with_dummies = data_no_mv.copy()


# In[15]:


# Setting the default display columns to all, as we need to know the data types of each variables
pd.set_option('display.max_columns', None)
data_with_dummies


# In[16]:


data_with_dummies.dtypes


# In[17]:


# We are selecting only the variables with object as their datatypes
car_df_objects = data_with_dummies.select_dtypes(include=['object']).copy()


# In[18]:


car_df_objects.columns


# In[19]:


# Creating a new list with the selected features

dummy_labels = ['make', 'fuel-type', 'aspiration', 'num-of-doors',
       'body-style', 'drive-wheels', 'engine-location', 'engine-type',
       'num-of-cylinders', 'fuel-system']


# In[20]:


# Creating a new dataframe with the selected features
df_new = car_df_objects[dummy_labels]


# In[21]:


df_new.head() 


# In[22]:


#Utilizing sklearn.preprocessing.LabelEncoder, we are assigning dummy values to each variable. This process involves the deployment of a 'for-loop'.
# The process will generate a new column containing the dummy values corresponding to their respective source columns.
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for column in df_new.columns:
    if df_new[column].dtype == 'object':  # Apply label encoding only to object (string) columns
        df_new[column + '_encoded'] = label_encoder.fit_transform(df_new[column])


# In[23]:


#Displaying the dataset to confirm the successful assignment of dummy values for each variable, ensuring their presence in the respective new columns
df_new


# In[24]:


df_new.columns


# In[25]:


# Choosing only the labeled columns from the 'df_new' dataset to be appended to the 'data_with_dummies' dataset.

new_columns_to_added = ['make_encoded', 'fuel-type_encoded',
       'aspiration_encoded', 'num-of-doors_encoded', 'body-style_encoded',
       'drive-wheels_encoded', 'engine-location_encoded',
       'engine-type_encoded', 'num-of-cylinders_encoded',
       'fuel-system_encoded']


# In[26]:


data_with_dummies[new_columns_to_added] = df_new[new_columns_to_added]


# In[27]:


# Now the dataset contains the columns with dummy variables
data_with_dummies


# In[28]:


data_with_dummies.columns


# In[29]:


#As we have labeled columns with dummy variables, the inclusion of their corresponding source columns is redundant for the regression model. Consequently, we can remove those columns.

columns_to_remove = ['make', 'fuel-type', 'aspiration', 'num-of-doors',
       'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders',
        'fuel-system']
data_with_dummies = data_with_dummies.drop(columns=columns_to_remove)


# In[30]:


# Displaying the dataset to confirm the presence of only the labelled columns excluding their respective source columns
data_with_dummies


# ## Reordering the columns to ensure that it matches with the original dataset

# In[31]:


data_with_dummies.columns


# In[32]:


reorder_columns = ['symboling', 'make_encoded', 'fuel-type_encoded', 'aspiration_encoded', 'num-of-doors_encoded', 
                   'body-style_encoded', 'drive-wheels_encoded', 'engine-location_encoded','wheel-base', 'length', 
                   'width', 'height', 'curb-weight', 'engine-type_encoded', 'num-of-cylinders_encoded', 
                   'engine-size', 'fuel-system_encoded', 'bore', 'stroke', 'compression-ratio', 'horsepower', 
                   'peak-rpm', 'city-mpg', 'highway-mpg', 'price', ]


# In[33]:


data_w_dum_reordered = data_with_dummies[reorder_columns]


# In[34]:


# Displaying the dataset to correct order of the columns
data_w_dum_reordered.head()


# ### Now we can continue with the process of resolving the inconsistencies in the data types of the variables. As mentioned earlier, regression models can only handle variables with integer or float data types.

# In[35]:


#Checking the datatypes of all variables
data_w_dum_reordered.dtypes


# # The following 5 features currently have an 'object' data type and should be updated to their respective data types as mentioned below:
# # bore
# # stroke
# # horsepower
# # peak-rpm
# # price
# #df['Column1'] = df['Column1'].astype(float)
# ![image.png](attachment:image.png)

# In[36]:


# Standardizing all five data types to float for the regression model as the regression model will consdier all the values as float. This is the reason we are seeing the results in float.

columns_to_convert = ["bore", "stroke", "horsepower","peak-rpm","price"]

for column in columns_to_convert:
    data_w_dum_reordered[column] = data_w_dum_reordered[column].astype(float)
    print(f"{column}: {data_w_dum_reordered[column].dtypes}")


# In[37]:


# Verfying the datatypes
data_cleaned = data_w_dum_reordered.copy()
data_cleaned.dtypes


# #### All the datatypes have been standardised

# #### Now we have cleaned the dataset which is ready to be analysed for prediction modeling

# In[38]:


#Utilizing the 'describe' function to examine the statistics of each variable, ensuring that the dataset comprises 193 observations without any missing values
data_cleaned.describe()


# ### Finding the significant feature using OLS method

# In[39]:


data_cleaned.columns


# In[40]:


x1 = data_cleaned[['symboling', 'make_encoded', 'fuel-type_encoded', 'aspiration_encoded',
       'num-of-doors_encoded', 'body-style_encoded', 'drive-wheels_encoded',
       'engine-location_encoded', 'wheel-base', 'length', 'width', 'height',
       'curb-weight', 'engine-type_encoded', 'num-of-cylinders_encoded',
       'engine-size', 'fuel-system_encoded', 'bore', 'stroke',
       'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
       'highway-mpg',]]
y = data_cleaned["price"]


# In[41]:


# Fitting all the features to know its respective p-values. P-values will reveal the signficance of the feature. 
x = sm.add_constant(x1)
result = sm.OLS(y,x).fit()
result.summary()


# #### Results summary: I am removing all the features which has the P-value of more than 0.1,

# In[42]:


# I am retaining the feature which has the p-value less than 0.1, which the indicates the significance of the feature.
# though fuel-system_encoded had 0.102 p-value I am retaining the feature as removing it is giving the r-squared value of 0.897

x1 = data_cleaned[['make_encoded', 'aspiration_encoded', 'body-style_encoded', 
                   'drive-wheels_encoded', 'engine-location_encoded', 'width', 'height', 
                   'engine-size',  'fuel-system_encoded', 'peak-rpm', 'stroke', 'highway-mpg']]
y = data_cleaned["price"]


# In[43]:


x = sm.add_constant(x1)
result = sm.OLS(y,x).fit()
result.summary()


# ##### FINDINGS: The 'R2' has a very negligible reduction. With regards to 'Adjusted R2' the value does not change. Hence the model explains 90% of the variability of the target variable.

# # Linear regression using SK learn

# 
# # Train_Test split. 
# 
# #We are using SKlearn to split the data into training (80%) and testing (20%) sets.

# #### We are using the significant columns identified through the OLS method to perform an 80-20 split of the data.

# In[44]:


data_optimized = data_cleaned[['make_encoded', 'aspiration_encoded', 'body-style_encoded', 
                   'drive-wheels_encoded', 'engine-location_encoded', 'width', 'height', 
                   'engine-size',  'fuel-system_encoded', 'stroke', 'peak-rpm', 'highway-mpg', 'price']]


# In[45]:


data_optimized.head()


# In[46]:


data_optimized.describe()


# #### Findings: All the variables except engine-size and price is normally distributed. We will first create model with existing data and apply no scaling at this stage

# In[47]:


# Taking a copy of the data, so that the dataset with optimized features are not affected
data_final = data_optimized.copy()


# In[48]:


# Resetting the index, as the data will have the index of the original dataset
data_final = data_final.reset_index(drop=True)
data_final


# #### Train_Test split of the dataset (data_cinal) (80%-20%)

# In[49]:


# Importing train_test_split function from SKlearn

from sklearn.model_selection import train_test_split

# splitting dataset into features and target variables

X = data_final.drop('price', axis=1)
y = data_final["price"]


# In[50]:


# Splitting the dataset. Using random state "42" to ensure that the data is split with the same random style
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[51]:


# Importing LinearRegression function from SKlearn
from sklearn.linear_model import LinearRegression

# Assignging varible to regression model
model = LinearRegression()


# In[52]:


#Fitting the training data into the moddel
model.fit(X_train, y_train)


# In[53]:


# predcting the target variable using the model
y_hat = model.predict(X_train)


# In[54]:


y_hat


# #### R2-score (Model's prediction power on training set)

# In[55]:


model.score(X_train, y_train)


# #### R2 (Model's prediction power on test set)

# In[56]:


model.score(X_test, y_test)


# ##### Defining a function for calculating Adjusted R2

# In[57]:


def adj_r2(x,y):
    r2 = model.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# ##### Adjusted R2 on the training set

# In[58]:


adj_r2(X_train, y_train)


# ##### Adjusted R2 on the test set

# In[59]:


adj_r2(X_test, y_test)


# In[60]:


#plotting the y_train and y_hat(predicted_train data) to know the accuracy of the prediction
plt.scatter(y_train, y_hat)
plt.xlabel('y_train', size=18)
plt.ylabel('y_hat', size=18)
plt.show


# ##### Metrics for the train data

# In[61]:


from sklearn import metrics
mae = metrics.mean_absolute_error(y_train, y_hat)
mse = metrics.mean_squared_error(y_train, y_hat)
r2 = metrics.r2_score(y_train, y_hat)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')


# In[62]:


y_hat_test = model.predict(X_test)
y_hat_test


# In[63]:


#plotting the y_test and y_hat_test(predicted_train data) to know the accuracy of the prediction
plt.scatter(y_test, y_hat_test)
plt.xlabel('y_test', size=18)
plt.ylabel('y_hat_test', size=18)
plt.show


# In[64]:


from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_hat_test)
mse = metrics.mean_squared_error(y_test, y_hat_test)
r2 = metrics.r2_score(y_test, y_hat_test)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')


# ## MODEL_2 - With normalized price

# ### Now we will scale the data and create a new model on the scaled pice 
# 

# ##### Finding the distribution of the target variable

# In[65]:


sns.displot(data_optimized["price"])


# ##### FINDINGS: Data is not normally distributed. Hence, I am taking the log of all the values 

# In[66]:


log_price = np.log(data_optimized["price"])
data_optimized["log_price"] = log_price


# In[67]:


sns.displot(data_optimized["log_price"])


# ##### Now the price is normally distributed

# In[68]:


#I am removing the existing price columns as we have another column with standardised price
data_final1 = data_optimized.drop(["price"], axis=1)
data_final1


# In[69]:


#Resetting the index as the dataset is still following the index of uncleaned dataset
data_final1 = data_final1.reset_index(drop=True)
data_final1


# ##### Train_test split (80-20)

# In[70]:


from sklearn.model_selection import train_test_split
X = data_final1.drop('log_price', axis=1)
y = data_final1["log_price"]


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[72]:


model1 = LinearRegression()


# In[73]:


model1.fit(X_train, y_train)


# In[74]:


y_hat1 = model1.predict(X_train)


# In[75]:


# Since we took log price, the results are in log values. We can use exponential to convert them into actual price


# In[76]:


model1.score(X_train, y_train)


# In[77]:


model1.score(X_test, y_test)


# In[78]:


model1.coef_


# In[79]:


model1.intercept_


# In[80]:


def adj_r2(x,y):
    r2 = model1.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# In[81]:


adj_r2(X_train, y_train)


# In[82]:


from sklearn import metrics
mae = metrics.mean_absolute_error(y_train, y_hat1)
mse = metrics.mean_squared_error(y_train, y_hat1)
r2 = metrics.r2_score(y_train, y_hat1)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')


# In[83]:


y_hat1_test = model1.predict(X_test)


# In[84]:


y_hat1_test


# In[85]:


actual_price = np.exp(y_hat1_test)
actual_price


# In[86]:


mae = metrics.mean_absolute_error(y_test, y_hat1_test)
mse = metrics.mean_squared_error(y_test, y_hat1_test)
r2 = metrics.r2_score(y_test, y_hat1_test)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')


# ## MODEL_3 - Creating model with the scaled dataset

# In[87]:


from sklearn.preprocessing import StandardScaler
target = data_final1['log_price']
inputs = data_final1.drop('log_price', axis=1)


# In[88]:


scaler = StandardScaler()
scaler.fit(inputs)


# In[89]:


inputs_scaled = scaler.transform(inputs)


# In[90]:


X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, target, test_size=0.2, random_state=42)


# In[91]:


reg1 = LinearRegression()
reg1.fit(X_train,y_train)


# In[92]:


y_hat2 = reg1.predict(X_train)


# In[93]:


y_hat2_test = reg1.predict(X_test)


# In[94]:


reg1.score(X_train, y_train)


# In[95]:


def adj_r2(x,y):
    r2 = reg1.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
adj_r2(X_train, y_train)


# In[96]:


reg1.score(X_test, y_test)


# In[97]:


adj_r2(X_test, y_test)


# In[98]:


#from sklearn import metrics
# Metrics for Test Set

mae = metrics.mean_absolute_error(y_test, y_hat2_test)
mse = metrics.mean_squared_error(y_test, y_hat2_test)
r2 = metrics.r2_score(y_test, y_hat2_test)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')


# In[99]:


# Metrics for training Set

mae = metrics.mean_absolute_error(y_train, y_hat2)
mse = metrics.mean_squared_error(y_train, y_hat2)
r2 = metrics.r2_score(y_train, y_hat2)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')


# ## Final result: There is no signficant imporovement in the predictive power of the models after scaling the data. Hence the first model is suffcient for prediction.

# In[ ]:




