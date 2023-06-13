#!/usr/bin/env python
# coding: utf-8

# **Vedant Modak**
#   | BE(IT) undergrad @ PES Modern College of Engineering,Pune.

# **Health-Care Analysis**

# In[292]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# **Importing the dataset**

# In[293]:


df=pd.read_csv('F:\Data Analytics\Portfolio\Projects\Project - 3 (HealthCare analysis)\BRCA.csv')


# In[294]:


df.head()


# In[295]:


df.tail()


# In[296]:


df.shape


# In[297]:


df.info()


# **Cleaning the dataset**

# In[298]:


df = df.drop_duplicates()


# In[299]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[300]:


df.isnull().sum()


# In[301]:


df.describe()


# In[302]:


df.dropna(inplace=True)


# In[303]:


df.isnull().sum()


# In[304]:


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)


# **Removing Outliers**

# In[305]:


sns.boxplot(df['Age'])


# In[306]:


sns.boxplot(df['Protein1'])


# In[307]:


sns.boxplot(df['Protein2'])


# In[308]:


sns.boxplot(df['Protein3'])


# In[309]:


sns.boxplot(df['Protein4'])


# In[310]:


def remove_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    threshold = 1.5 * IQR
    outlier_mask = (column < Q1 - threshold) | (column > Q3 + threshold)
    return column[~outlier_mask]


# In[311]:


col_name = ['Age', 'Protein1', 'Protein2', 'Protein3', 'Protein4']
for col in col_name:
    df[col] = remove_outliers(df[col])


# In[312]:


plt.figure(figsize=(10, 6)) 

for col in col_name:
    sns.boxplot(data=df[col])
    plt.title(col)
    plt.show()


# **Converting categorical variables into numerical format using encoding**

# In[313]:


df.head(5)


# In[314]:


del df['Patient_ID']


# In[315]:


df.head(2)


# **Changing data types and converting categorical values into numerical values using Label encoding**

# In[316]:


df["Gender"] = df["Gender"].astype("str")
df["Tumour_Stage"] = df["Tumour_Stage"].astype("str")
df["Histology"] = df["Histology"].astype("str")
df["ER status"] = df["ER status"].astype("str")
df["PR status"] = df["PR status"].astype("str")
df["HER2 status"] = df["HER2 status"].astype("str")
df["Surgery_type"] = df["Surgery_type"].astype("str")
df["Patient_Status"] = df["Patient_Status"].astype("str")


# In[317]:


from sklearn.preprocessing import LabelEncoder


# In[318]:


encode = LabelEncoder()
label1=encode.fit_transform(df["Gender"])
label2=encode.fit_transform(df["Tumour_Stage"])
label3=encode.fit_transform(df["Histology"])
label4=encode.fit_transform(df["ER status"])
label5=encode.fit_transform(df["PR status"])
label6=encode.fit_transform(df["HER2 status"])
label7=encode.fit_transform(df["Surgery_type"])
label8=encode.fit_transform(df["Patient_Status"])


# In[319]:


df=df.drop("Gender",axis='columns')
df=df.drop("Tumour_Stage",axis='columns')
df=df.drop("Histology",axis='columns')
df=df.drop("ER status",axis='columns')
df=df.drop("PR status",axis='columns')
df=df.drop("HER2 status",axis='columns')
df=df.drop("Surgery_type",axis='columns')
df=df.drop("Patient_Status",axis='columns')


# In[320]:


df.head(2)


# In[321]:


df["Gender"]=label1
df["Tumour_Stage"]=label2
df["Histology"]=label3
df["ER status"]=label4
df["PR status"]=label5
df["HER2 status"]=label6
df["Surgery_type"]=label7
df["Patient_Status"]=label8


# In[322]:


df.head(10)


# In[323]:


del df['Date_of_Surgery']


# In[324]:


del df['Date_of_Last_Visit']


# In[325]:


df.head(10)


# In[326]:


df["Age"] = df["Age"].astype("int64")
df["Protein1"] = df["Protein1"].astype("float32")
df["Protein2"] = df["Protein2"].astype("float32")
df["Protein3"] = df["Protein3"].astype("float32")
df["Protein4"] = df["Protein4"].astype("float32")


# In[327]:


df.info()


# In[328]:


df.corr()


# In[329]:


plt.figure(figsize=(16, 8))
sns.heatmap(df.corr())


# **General Analysis**

# In[330]:


sns.relplot(x = "Tumour_Stage", y = "Age", hue= "Patient_Status", data =df);


# This tells us as the age increases, the chances of death also increase irrespective of the stage of the tumor.

# Patients are more likely to be alive if the tumore stage is below 1.

# **On the basis of this data, lets predict whether patient will be Alive or Dead**

# In[331]:


feature=['Age', 'Protein1', 'Protein2', 'Protein3', 'Protein4', 'Gender', 'Tumour_Stage', 'Histology', 'ER status', 'PR status', 'HER2 status', 'Surgery_type']


# In[332]:


X=df[feature]
X.columns


# In[333]:


y=df[['Patient_Status']]
y.columns


# In[334]:


from sklearn.model_selection import train_test_split


# In[335]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[336]:


X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
y_train = y_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
y_test = y_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)


# In[337]:


X_train.shape


# In[338]:


X_test.shape


# In[339]:


y_train.shape


# In[340]:


y_test.shape


# **Prediction using Decision Tree Classifier**

# In[341]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


# In[342]:


Cancer_1=DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
Cancer_1.fit(X_train,y_train)


# In[343]:


y_predicted=Cancer_1.predict(X_test)
y_predicted


# In[344]:


accuracy_score(y_test, y_predicted)*100


# This model is 84% accurate

# **Prediction using Random Forest Classifier**

# In[345]:


from sklearn.ensemble import RandomForestClassifier
Cancer_2=RandomForestClassifier()
Cancer_2.fit(X_train,y_train)


# In[346]:


y_predicted_2=Cancer_2.predict(X_test)
y_predicted_2


# In[347]:


accuracy_score(y_test, y_predicted_2)*100


# This model is 89% accurate

# **Comparing the results of the models**

# In[348]:


def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}


# In[349]:


Cancer_1_eval = evaluate_model(Cancer_1, X_test, y_test)

# Print result
print('Accuracy:', Cancer_1_eval['acc'])
print('Precision:', Cancer_1_eval['prec'])
print('Recall:', Cancer_1_eval['rec'])
print('F1 Score:', Cancer_1_eval['f1'])
print('Cohens Kappa Score:', Cancer_1_eval['kappa'])
print('Area Under Curve:', Cancer_1_eval['auc'])
print('Confusion Matrix:\n', Cancer_1_eval['cm'])


# In[350]:


Cancer_2_eval = evaluate_model(Cancer_2, X_test, y_test)

# Print result
print('Accuracy:', Cancer_2_eval['acc'])
print('Precision:', Cancer_2_eval['prec'])
print('Recall:', Cancer_2_eval['rec'])
print('F1 Score:', Cancer_2_eval['f1'])
print('Cohens Kappa Score:', Cancer_2_eval['kappa'])
print('Area Under Curve:', Cancer_2_eval['auc'])
print('Confusion Matrix:\n', Cancer_2_eval['cm'])


# In[351]:


Cancer_1_eval = evaluate_model(Cancer_1, X_test, y_test)
Cancer_2_eval = evaluate_model(Cancer_2, X_test, y_test)




# Intitialize figure with two plots
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor('white')

# First plot
## set bar size
barWidth = 0.2
Cancer_1_score = [Cancer_1_eval['acc'], Cancer_1_eval['prec'], Cancer_1_eval['rec'], Cancer_1_eval['f1'], Cancer_1_eval['kappa']]
Cancer_2_score = [Cancer_2_eval['acc'], Cancer_2_eval['prec'], Cancer_2_eval['rec'], Cancer_2_eval['f1'], Cancer_2_eval['kappa']]


## Set position of bar on X axis
r1 = np.arange(len(Cancer_1_score))
r2 = [x + barWidth for x in r1]

## Make the plot
ax1.bar(r1, Cancer_1_score, width=barWidth, edgecolor='white', label='Decision Tree Classifier')
ax1.bar(r2, Cancer_2_score, width=barWidth, edgecolor='white', label='Random Forest Classifier')

## Configure x and y axis
ax1.set_xlabel('Metrics', fontweight='bold')
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
ax1.set_xticks([r + (barWidth * 1.5) for r in range(len(Cancer_1_score))], )
ax1.set_xticklabels(labels)
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_ylim(0, 1)

## Create legend & title
ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
ax1.legend()

# Second plot
## Comparing ROC Curve
ax2.plot(Cancer_1_eval['fpr'], Cancer_1_eval['tpr'], label='Decision Tree Classifier, auc = {:0.5f}'.format(Cancer_1_eval['auc']))
ax2.plot(Cancer_2_eval['fpr'], Cancer_2_eval['tpr'], label='Random Forest Classifier, auc = {:0.5f}'.format(Cancer_2_eval['auc']))

## Configure x and y axis
ax2.set_xlabel('False Positive Rate', fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontweight='bold')

## Create legend & title
ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend(loc=4)

plt.show()

