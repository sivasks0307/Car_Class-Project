#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary packages

# In[34]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# make_pipeline it automatically names each step in the pipeline with lowercase versions of the class names. This eliminates the need to explicitly provide names for each step in the pipeline.
from sklearn.pipeline import make_pipeline 
from sklearn.metrics import classification_report, confusion_matrix


# In[35]:


# Reading the data set
df = pd.read_csv("I://SSSK//Data Science-D55//Project - GUVI//Final project//Car_Class//cars_class(copy.csv")
df


# In[7]:


#Statistics of the dataset
df.describe()


# ##### Findings: Almost all variables exhibit a normal distribution, a conclusion drawn from examining their mean, standard deviation, minimum, and maximum values.

# In[36]:


#Displaying the names of all columns or features to identify any potential absence in the statistical output.
df.columns


# ##### All the columns or features listed above are included in the statistics, indicating that they exclusively consist of only numerical data types (integers or floats).

# In[37]:


# Examining the dataset to identify any null values and obtain the data types of each feature.
df.info()


# ##### Findings: Fortunately, all features have integer data types, and no null values were found.

# ## Creating the model (LogisticRegression)

# In[11]:


# Separating features (X) and the target variable (y) from the DataFrame
# The 'ID' feature holds no informative value as it solely serves as an index. Therefore, we can exclude it from the dataframe before fitting it to the model.

X = df.drop(['ID', 'Class'], axis=1)
y = df['Class']


# In[12]:


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


# Applying the optimal hyperparameters obtained through the 'RandomizedSearchCV' method. The complete code provided at the end of this page.
best_hyperparameters = {'penalty': 'l2', 'class_weight': None, 'C': 29.763514416313132}


# In[14]:


# Constructing a pipeline with scaling and logistic regression using the optimal hyperparameters obtained through the 'RandomizedSearchCV' method.
# make_pipeline automatically names each step in the pipeline with lowercase versions of the class names. This eliminates the need to explicitly provide names for each step in the pipeline.

model = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=5000, **best_hyperparameters))


# In[15]:


# Fitting the model to the scaled training dataset.
model.fit(X_train, y_train)


# ## Test set prediction results

# In[16]:


# Evaluating the model on the scaled test set

test_accuracy = model.score(X_test, y_test)
print("\nTest Accuracy:", test_accuracy)


# In[32]:


## Classification report for the test set

# Defining the actual class names
class_names = ['bus', 'Opel Manta', 'Saab', 'Van']

# Classification report for the test set
class_rep_test = classification_report(y_test, model.predict(X_test), target_names=class_names)

print("\nClassification Report (Test Set):")
print(class_rep_test)


# In[26]:


## Confusion matrix for the test set

conf_matrix_test = confusion_matrix(y_test, model.predict(X_test))

# Defining the actual class names
class_names = ['bus', 'Opel Manta', 'Saab', 'Van']

# Create a DataFrame for the confusion matrix with rows and column names
conf_matrix_df_test = pd.DataFrame(conf_matrix_test, 
                                   index=class_names, 
                                   columns=class_names)

print("\nConfusion Matrix (Test Set):")
print(conf_matrix_df_test)


# ## Training set prediction results

# In[17]:


# Evaluating the model on the scaled training set

train_accuracy = model.score(X_train, y_train)
print("Train Accuracy:", train_accuracy)


# In[33]:


## Classification report for the training set

# Defining the actual class names
class_names = ['bus', 'Opel Manta', 'Saab', 'Van']

# Classification report for the training set
class_rep_train = classification_report(y_train, model.predict(X_train), target_names=class_names)

print("\nClassification Report (Training Set):")
print(class_rep_train)


# In[29]:


# Confusion matrix for the training set

conf_matrix_train = confusion_matrix(y_train, model.predict(X_train))

# Defining the actual class names
class_names = ['bus', 'Opel Manta', 'Saab', 'Van']

# Create a DataFrame for the confusion matrix with rows and column names
conf_matrix_df_train = pd.DataFrame(conf_matrix_train, 
                                   index=class_names, 
                                   columns=class_names)

print("\nConfusion Matrix (Training Set):")
print(conf_matrix_df_train)


# In[ ]:





# ## Conclusion

# ##### I have experimented with various methods, including Decision Tree, Random Forest, XGBoost, and found that only Logistic Regression yielded the highest prediction accuracy.

# ##### I attempted feature engineering to identify the optimal features for the model and discovered that the best feature does not contribute to any improvement in the model's accuracy.

# In[ ]:





# ### Code with randomized search, in case we want to explore how the optimal parameters were determined.

# In[ ]:


# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import RandomizedSearchCV, train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np

# df = pd.read_csv("I://SSSK//Data Science-D55//Project - GUVI//Final project//Car_Class//cars_class(copy.csv")
# X = df.drop(['ID', 'Class'], axis=1)
# y = df['Class']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=5000))

# param_dist = {
#     'logisticregression__penalty': ['l2'],
#     'logisticregression__C': np.logspace(-4, 4, 20),
#     'logisticregression__class_weight': [None, 'balanced'],
# }

# random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
#                                    n_iter=10, cv=5, scoring='accuracy', random_state=42)

# random_search.fit(X_train, y_train)

# print("Best Hyperparameters:", random_search.best_params_)

# test_accuracy = random_search.best_estimator_.score(X_test, y_test)
# print("\nTest Accuracy:", test_accuracy)

# train_accuracy = random_search.best_estimator_.score(X_train, y_train)
# print("Train Accuracy:", train_accuracy)

# class_rep_test = classification_report(y_test, random_search.best_estimator_.predict(X_test))
# print("\nClassification Report (Test Set):")
# print(class_rep_test)

# class_rep_train = classification_report(y_train, random_search.best_estimator_.predict(X_train))
# print("\nClassification Report (Training Set):")
# print(class_rep_train)

# conf_matrix_test = confusion_matrix(y_test, random_search.best_estimator_.predict(X_test))
# print("\nConfusion Matrix (Test Set):")
# print(conf_matrix_test)

# conf_matrix_train = confusion_matrix(y_train, random_search.best_estimator_.predict(X_train))
# print("\nConfusion Matrix (Training Set):")
# print(conf_matrix_train)

