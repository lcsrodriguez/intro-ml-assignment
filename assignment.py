#!/usr/bin/env python
# coding: utf-8

# # Introduction to ML - Assignment 1 - April 9, 2022

# - Student : **Lucas RODRIGUEZ** &bull; [lucas.rodriguez3@studio.unibo.it](mailto:lucas.rodriguez3@studio.unibo.it)
# - Git repo : [github.com/lcsrodriguez/intro-ml-assignment](https://github.com/lcsrodriguez/intro-ml-assignment)

# ## General principles for a correct model development: Read carefully!!!

# **The pre-modeling phase aims to obtain a clean training and test database to feed the learning algorithms.**
# 
# It is a very (the most?) important phase of the model development process as the Garbage In-Garbage Out principle applies ... so take your time to get it right.
# 
# Once the raw data has been acquired (how to do that depends on the support and the characteristics of the raw data (structured or unstructured data)) in a suitable environment (pandas) the first thing is to do an **Exploratory Data Analysis - EDA**, in simple words ... **look the data**:
# - make graphs (histograms, scatter plots, box-plots ...)
# - analyze the marginal distributions (mean, variance, max, min, percentiles)
# - analyze the joint distribution of the variables (correlations)
# 
# Once you have an idea of what your data is like, you can start addressing any (but almost certain) problems you will encounter ...
# 
# The most common answer these questions (not necessarily in this order ...):
# 
# - <mark style="background:yellow">**are there any non-numeric formats (strings) in the data?**</mark> $\Longrightarrow$ <b style="color: green">OK</b>
#     - almost all ML algorithms can work only with numeric data (int or float) and therefore this data must be converted into numeric data.
#     - in general this process goes under the name of **'category encoding'** and the type of encoding to use depends on the characteristics of the variables ...
# 
# 
# - <mark style="background:yellow">**are there missing data for some variables**?</mark> 
# $\Longrightarrow$ <b style="color: green">OK</b>
#     - and if so what is the best strategy to manage them?
#         - delete the variables?
#         - delete observations with missing data?
#         - replace the missing data with an estimate of the missing value?
# 
# 
#   Which strategy to adopt depends on the number of observations and variables you have available...
# 
# - <mark style="background:yellow">**are there outliers in the variables**?</mark>
#     - how do i identify them?
#         - univariate or multivariate analysis...
#     - how do I manage them?
#         - delete observations with anomalous data? $\Longrightarrow$ <b style="color: red">Not a solution : can remove too many tuples</b>
#         - replace the outliers with an estimate? $\Longrightarrow$ <b style="color: green">Solution</b>
#     - are outliers really a problem? $\Longrightarrow$ <b style="color: red">WARNING : PROBLEM</b>
#         - there are algorithms that are robust in the presence of outliers
# 
# 
#   Again which strategy to adopt depends on the size of the sample you have available ...
# 
# <mark style="background:yellow">**Two points to remember**</mark>
# 
# - the principle less data = more variance always applies ...
# - **if you use estimates to replace missing or outliers values** these estimates must be computed
#     - **after** you have splitted the sample into training-test subsamples
#     - <mark style="background:yellow">**on training data only**</mark> and then applied to test data
# 
# **How to organize these activities?**
# 
# - try to avoid spaghetti code
# - try to write functions that group the operations that are related to the same transformation
# - try to use Pipelines to organize the flow of data between these functions
# 
# 
# **The paragraphs reported in each exercise are indicative only and serve to remind the student of the minimal set of analyzes that must be carried out. The student is free to add any other type of analysis he deems appropriate at his discretion**.

# ---

# ## Exercise 1 - Data Preprocessing

# In this exercise, you need to process a data file that contains many invalid lines. You will find some null data and others field with various values (eg 'Missing' or NA) which indicate that the corresponding data is not valid. Furthermore, even in some numeric columns there may be characters (eg '-') which indicate the absence of a data. Also in this case you have to understand how to deal with the data (eg replace with 0). When you have found a clean sub-set of data you will need to convert all the columns into numerical data by applying the techniques learned in the course to deal with categorical data. Finally, choose a data normalization method.
# 
# **For this exercise you need to use the file: exercise-1-1.csv**

# ### Import Libraries and Upload Data File

# In[54]:


# Attempt to remove old values for used variables
try:
    del df
    del X
except:
    pass

# Importing relevant libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

# Gathering the main file called : exercise-1-1.csv
df = pd.read_csv("exercise-1-1.csv", sep=";", index_col="id", na_values=np.nan)


# In[55]:


# Get some informations about the different columns composing the csv file
df.info()

# Get the general shape of the DataFrame
df.shape


# We have an initial Pandas DataFrame composed of 9 columns (plus 1 column for the unique id of each row) and 15139 rows.

# In[56]:


df


# ### Data pre-processing

# We can handle duplicate rows by using the following instruction

# In[57]:


df.drop_duplicates(inplace=True) # we keep the "keep" argument to its default value : "first"
df.shape


# We have removed more than 1000 lines which were duplicated in the original DataFrame

# **Extra Remark** : We don't modify the columns labels because these are in lowercase and with dash, so everything is understandable.

# **Check for Uniqueness of Data** - Avoid to use columns with a single constant value for all records ... 

# In[58]:


# We first determine for each column, the number of different values in this column by using the following Pandas method
df.nunique()


# We see that the column labeled as `status-prev` and `status-after` only contain 1 single value for each tuple in the DataFrame.
# 
# $\Longrightarrow$ They are <ins style='color:red'>**Zero-variance predictors**</ins> which refer to input features that contain a single value across the entire set of observations. 
# 
# 
# $\Longrightarrow$ Accordingly, they do not add any value to the prediction algorithm since the target variable is not affected by the input value, making them redundant.
# 
# $\Longrightarrow$ Thanks to Pandas, we will remove these two columns by using the following method :

# In[59]:


df.drop(columns = df.columns[df.nunique() == 1], inplace = True)
df.head()


# In[60]:


df.shape


# We now have only 7 columns.

# **Cleaning Data** - Converting date to datetime, replace '-' with appropriate value in the 'limit-balance' column, you should also pay attention to the number format of 'balance' and 'limit-balance' column, it does not seem the original format can be used as a numerical format ... 

# In[61]:


df.info()


# We see that :
# - The columns `balance` and `limit-balance` have an `object` type. We have to convert it into 2 `float64` fields by first handling carefuly the way they are originally set (with a point and not a comma).
# - The column `reference-date` is an object column. However, Pandas offers the opportunity to deal with `datetime64`-typed column.
# - The column `economic-sector-code` is an integer-populated column and we have some `missing` observation which have to be replaced by NaN $\Longrightarrow$ we have to convert the column into a `int64` column.
# 
# **References** :
# - https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes
# - https://pbpython.com/pandas_dtypes.html
# 
# ---

# $\longrightarrow$ **Column `economic-sector-code`**

# In[62]:


df["economic-sector-code"].replace("missing", np.nan, inplace=True)


# $\longrightarrow$ **Column `geographic-area`**

# In[63]:


df["geographic-area"].replace("missing", np.nan, inplace=True)


# $\longrightarrow$ **Column `reference-date`**

# In[64]:


# Dealing with reference-date : convert the column from date to datetime
df["reference-date"] =  pd.to_datetime(df["reference-date"])


# $\longrightarrow$ **Column `limit-balance`** (only the dash issue)
# 
# We have to replace the "-" by NaN values. However, we see that we cannot just perform a `replace("-", "<sth>")` because we have some spaces surrounding the "-". The right solution here is to perform a regex-driven replace operation.

# In[65]:


df["balance"].replace(to_replace="\s*-\s*", value=np.NaN, inplace=True, regex=True)
df["limit-balance"].replace(to_replace="\s*-\s*", value=np.NaN, inplace=True, regex=True)


# $\longrightarrow$ **Columns `balance` and `limit-balance`**
# 
# We have pointed out the format of `balance` and `limit-balance` data aren't really compatible with a direct `float64` conversion. 
# 
# Indeed, we see several times that :
# - We can have a correct float (set as a string) with 1 point
# - But we can also find some tuples with 2 points. My hypothesis is the first point (at the left) represents a separator for the $10^3$ gap
# 
# 
# <b style='color:red'>Solution</b> : We have to introduce and implement a function which will perform replacement and conversion : if we find 2 points, we remove the first one and concatenate the 2 parts of the results; otherwise, we do nothing.
# 
# <b style='color:green'>Remark</b> : We can use a $\lambda$-function to exploit the conciseness of Python.
# 
# 
# At final step, we can perform the cast of the whole column as a `float64` type.

# In[66]:


# Dealing with the point/comma issue in balance and limit-balance columns
# We have to cast the balance column to a float64 type

# Removing useless spaces
df["balance"].replace(" ", "")

def filter_numbers(x):
    """
    Function filtering and cleaning properly the balance and limit-balance columns
    :param x: String representing a float
    """
    if not pd.isna(x):
        if x.count(".") == 2:
            x_tmp = x.split(".")
            return x_tmp[0] + x_tmp[1] + "." + x_tmp[2]
        return x
    else:
        return np.nan

df["balance"] = df["balance"].apply(filter_numbers)
df["limit-balance"] = df["limit-balance"].apply(filter_numbers)

df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
df["limit-balance"] = pd.to_numeric(df["limit-balance"], errors="coerce")

# We have an issue concerning the double point into the balance/limit-balance columns.


# In[67]:


df.info()
df.head()


# ### Categorical Data Handling

# We have 2 features variables which are set as "categorical" : `counterparty-type` and `geographic-area`.
# 
# We can see that these 2 variables only take few values :

# In[68]:


df.nunique()[["counterparty-type", "geographic-area"]]


# In[69]:


df["counterparty-type"].unique()


# In[70]:


df["geographic-area"].unique()


# <b style="color:green">Remark</b> : We have 2 features with nominal (not ordinal) values. Indeed,
# - For `geographic-area` : there are only 6 zones. We don't have any natural order to apply on this set.
# - For `counterparty-type` : it seems there isn't any order in the type alphabet
# 
# $\Longrightarrow$ We have to apply an algorithm dealing with nominal features.

# We have to adopt a clear strategy to handle these categorical data.
# 
# We will use the algorithms from `sklearn.preprocessing`. Since we don't have ordinal features, we cannot use the `LabelEncoder` object.
# 
# $\Longrightarrow$ We will use  `One-Hot encoding` $\Longrightarrow$ It will add $9 + 6 = 15$ new "composite features".
# 
# 
# <p style='color:blue'>
# Forcing an ordinal relationship via an ordinal encoding and allowing the model to assume a natural ordering between categories may result in poor performance or unexpected results (predictions halfway between categories).
# </p>
# 

# $\longrightarrow$ Encoding of `counterparty-type`

# In[71]:


OHE = pd.get_dummies(df[['counterparty-type', 'geographic-area']])
X = df.copy()
X.drop(['counterparty-type', 'geographic-area'], axis=1, inplace=True)
X[OHE.columns] = OHE


# In[72]:


X


# ### NaN data (missing values) handling
# 
# Indeed, the final step of this exercise is to remove the high-correlated data.
# However, we cannot compute a simple correlation matrix with a dataframe containing NaN elements
# 
# $\Longrightarrow$ We have to handle the NaN data cleaning

# In[73]:


msno.matrix(X)


# In[74]:


msno.bar(X, color="red")


# We are observing 3 things :
# - The variable `limit-balance` has only almost 2000 values different from `NaN` over 14 000 observations
# - The variable `economic-sector-code` has only almost 2000 values different from `NaN` over 14 000 observations
# - The variable `balance` has 46 NaN tuples over 14 000. We can easily fill the other values with for instance
#     - the mean (or median) of the column
#     - the most frequent value of the column

# $\longrightarrow$ **Handling `limit-balance` and `economic-sector-code`**

# In[75]:


X.drop("limit-balance", axis=1, inplace=True)
X.drop("economic-sector-code", axis=1, inplace=True)


# $\longrightarrow$ **Handling `balance`**

# In[76]:


X["balance"].fillna(X["balance"].mean(), inplace=True)


# $\longrightarrow$ **Final summary of missing values**

# In[77]:


percent_missing = X.isnull().sum() * 100 / len(df)
percent_missing


# $\Longrightarrow$ Once we get a 0% for each column of missing values rate, we can consider the next step !

# ### Remove High-Correlated Data

# In[78]:


correlation_matrix = X.corr().abs()
sns.heatmap(correlation_matrix, annot=False, cmap="winter")
plt.title("Correlation matrix")
plt.show()


# In[79]:


# Select upper triangle of correlation matrix (because it's a symmetric matrix)
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
#print(upper)

# Find features with correlation greater than a fixed threshold
threshold = 0.9
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
#print(to_drop)

# Drop selected features
try:
    if len(to_drop) != 0:
        X.drop(to_drop, axis=1, inplace=True)
        print("Removing high-correlated columns")
except:
    pass


# Finnaly, our dataset at this stage is composed as follows :

# In[80]:


X.describe()


# ---

# ## Exercise 2 - Classification with Support Vector Machines (SVM)

# In this exercise you will have to use the data reported in the file **exercise-1-2.csv** which contains a series of data related to diagnostic images. The data relate to a number of characteristics found during breast cancer analyzes. You must use the SVM method to correctly classify the data. Remember to divide the data into a training set and a test set, then measure the effectiveness of your method. Finally, produce the confusion matrix related to your analysis.

# ### Loading data and import libraries

# In[1]:


# Attempt to remove old values for used variables
try:
    del df
    del X
    del X_train, X_test, Y_train, Y_test
except:
    pass

# Importing relevant libraries
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats.mstats import winsorize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Settings
plt.rcParams.update({'font.size': 15, 'figure.figsize': (10, 8)}) # set font and plot size to be larger
warnings.filterwarnings("ignore")

# Reading & Declaration the main dataset file
filename = "exercise-1-2.csv"
df = pd.read_csv(filename, index_col="id", na_values=np.nan)


# We recover some statistical indicators from each column :

# In[2]:


df.describe().transpose().round(2)


# ### Data pre-processing & Data cleaning

# $\longrightarrow$ **Dropping useless column** <b style="color:red">$\Longrightarrow$ DANGER : we have NaN values</b>

# In[3]:


df.drop("Unnamed: 32", axis = 1, inplace = True)


# $\longrightarrow$ **Checking for NaN values**

# In[4]:


msno.matrix(df)


# In[5]:


msno.bar(df, color="red")


# In[6]:


df.dropna(inplace=True) # just in case, to be sure


# In[7]:


df


# We can see that we don't have any missing values in the dataset ! $\Longrightarrow$ No pre-processing for NaN values needed !

# $\longrightarrow$ **Binarization of the target column**

# In[8]:


df["diagnosis"] = (df["diagnosis"] == "M").astype(int)


# $\longrightarrow$ **Feature normalization** 

# In[9]:


scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])


# In[10]:


df


# $\longrightarrow$ **Splitting dataset into $X$ and $Y$ dataframes**

# In[11]:


X = df.copy()
Y = X.pop("diagnosis")
Y.astype("int")


# $\longrightarrow$ **Handling outliers** <b style="color:red">$\Longrightarrow$ AFTER train/test split AND ONLY on train set</b>

# ### Visual Analysis of Data

# In this case you have a very large number of features and clearly you cannot make an n-dimension plotter! Try to select pairs of variables that can be informative...

# We have $31$ variables : 
# - 1 target variable : `diagnosis` 
# - 30 explanatory variables
# 
# In reality, we have 10 real variables, with which we have created 3 columns representing the `mean`, `se` (standard error) and `worst`.
# 
# The variable `diagnosis` is constructed as binary one.
# 
# $\Longrightarrow$ According to this dataset structure, we expect to "explain" the `diagnosis` variable thanks to the data provided by the other columns.

# In[12]:


df.plot(kind="scatter",
       x="radius_mean",
       y="smoothness_mean",
       title="smoothness as function of radius",
       color="green")


# In[13]:


df.plot(kind="box",
       title="boxplot of radius_mean",
       color="red", xlabel=None)
plt.xticks(rotation=90)


# As you can see, we have a lot of outliers

# In[14]:


sns.boxplot(x = df["radius_mean"], color="red")


# <b style="color: red">Important remark</b> : SVM is not <ins>very robust to outliers</ins>. Presence of a few outliers can lead to very bad
# global misclassification.
# 
# 
# **Reference** : https://lstat.kuleuven.be/research/lsd/lsd2008/presentationslsd3/Debruyne.pdf
# 
# We can easily remove the outliers **but this will narrow our dataset**. Indeed, if we drop the value, we have to remove all the corresponding row.
# 
# $\Longrightarrow$ Instead, we will use the **Winsorize method** to replace the outliers by "normal values".

# In[15]:


# Computation of IQR : Interquartile Range
Q1 = df["radius_mean"].quantile(0.25)
Q3 = df["radius_mean"].quantile(0.75)
IQR = Q3 - Q1

# Kind of "scale" parameter
alpha = 1.5
lower_boundary = Q1 - alpha*IQR
upper_boundary = Q3 + alpha*IQR
print(lower_boundary, upper_boundary)

# Apply the winsorized method on the first feature column
new_df_rm = winsorize(df["radius_mean"], (0.01, 0.03))

# Plot the new bloxplot and the new distplot
sns.boxplot(x = new_df_rm, color="green")
sns.displot(x = new_df_rm, bins = 40, kde = False, color="green")


# There is no outliers anymore. 
# 
# 
# #### Extension of this method to each column
# $\Longrightarrow$ We can extend this method for the other columns. 
# 
# $\Longrightarrow$ We will see that on the current dataframe <b style="color:red">AFTER</b> the train/test split method.

# In[16]:


# Plotting distribution plots for each column
fig, ax = plt.subplots(5, 3, figsize=(20, 20))
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        g = sns.distplot(X[X.columns[i*2+j]], ax=ax[i][j], color="blue")
        g.set(xlabel=None)
        g.set_title(X[X.columns[i*2+j]].name, fontsize = 15)


# $\Longrightarrow$ As we can see, these plots confirm we'll have to **perform the same operation** on the whole dataset.

# ### Dealing with correlated variables

# In[17]:


correlation_matrix = X.corr().abs()
sns.heatmap(correlation_matrix, annot=False, cmap="winter")
plt.title("Correlation matrix")


# We can drop the most correlated variables as below :

# In[18]:


# Select upper triangle of correlation matrix (because it's a symmetric matrix)
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
#print(upper)

# Find features with correlation greater than a fixed threshold
threshold = 0.9
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
#print(to_drop)

# Drop selected features
try:
    if len(to_drop) != 0:
        X.drop(to_drop, axis=1, inplace=True)
        print("Removing high-correlated columns")
except:
    pass


# ### Create Training and Test Dataset

# $\longrightarrow$ **Train/Test dataset split**

# In[25]:


# Size setting
size_sample_test = 0.2

# Creation of training and testing dataset + Shuffling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = size_sample_test)
X_train.sample(frac=1)
X_test.sample(frac=1)


# $\longrightarrow$ **Detection and replacing of outliers in the dataset**

# In[26]:


def apply_outliers_detect_col(col):
    """
    Function returning an outliers-corrected column from the initial column : col
    :param col: Initial column
    """
    return winsorize(col, limits=[0.05, 0.05])

def apply_outliers_detect(df):
    """
    Function returnin an outliers-corrected dataframe from the initial dataframe : df
    :param df: Initial dataframe
    """
    return df.apply(apply_outliers_detect_col, axis = 0)

# Applying the filtering of outliers on the train dataset ONLY
X_train = apply_outliers_detect(X_train)
X_train


# ### Apply SVM Method

# The regularization parameter $C$ serves as a degree of importance that is given to misclassifications. SVM pose a quadratic optimization problem that looks for maximizing the margin between both classes and minimizing the amount of misclassifications. However, for non-separable problems, in order to find a solution, the miclassification constraint must be relaxed, and this is done by setting the mentioned "regularization".
# 
# So, intuitively, as lambda grows larger the less the wrongly classified examples are allowed (or the highest the price the pay in the loss function). Then when lambda tends to infinite the solution tends to the hard-margin (allow no miss-classification). When lambda tends to 0 (without being 0) the more the miss-classifications are allowed.
# 
# There is definitely a tradeoff between these two and normally smaller lambdas, but not too small, generalize well. Below are three examples for linear SVM classification (binary).

# In[27]:


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 10.0 # SVM regularization parameter

# Create the SVM model
classifier = SVC(kernel='linear', C = C, gamma='auto', random_state=0)

# Fit the model for the data
classifier.fit(X_train, Y_train)


# ### Predictions

# In[28]:


# Make predictions
Y_pred = classifier.predict(X_test)


# ### Analyze Accuracy and Confusion Matrix

# In[29]:


# Compute the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)

# Show confusion matrix in a separate window
plt.matshow(cm, cmap="spring")
plt.title('Confusion matrix\n Total number of observations from $y_{{test}} = 131$\n')
plt.colorbar()
for (i, j), z in np.ndenumerate(cm):
    plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color='black')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# Cross Validation : $K$-Fold

# In[30]:


# Setting up K to 10
K = 10
accuracy = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv= K)

acc = {
    "Accuracy": accuracy.mean()*100,
    "Standard-Error": accuracy.std()*100
}
acc


# We obtain :
# - a model accuracy of $97\%$
# - a standard deviation of $2$

# In[115]:


from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(estimator=classifier,
    X=X_train,
    y=Y_train,
    #train_sizes=np.linspace(0.1, 1.0, 10),
    cv=10,
    n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
    color='blue', marker='o',
    markersize=5, label='Training accuracy')

plt.fill_between(train_sizes,
    train_mean + train_std,
    train_mean - train_std,
    alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
    color='green', linestyle='--',
    marker='s', markersize=5,
    label='Validation accuracy')

plt.fill_between(train_sizes,
    test_mean + test_std,
    test_mean - test_std,
    alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5, 1.2))


# ### Appendix : Confusion matrix composition
# <img src="https://i1.wp.com/miro.medium.com/max/2102/1*fxiTNIgOyvAombPJx5KGeA.png?zoom=2&w=578&ssl=1" width="400px"/>
