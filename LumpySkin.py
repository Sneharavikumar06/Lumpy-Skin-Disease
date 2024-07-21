# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats


# In[2]:


data=pd.read_csv(r"D:\DigitalValleyIndia\2023-2024\B.E\Karunya\NewRegistered\kumar Krishna\Lumpyskin_diseasedata.csv")
data.head()


# In[3]:


data=data.drop(['region','country','reportingDate'],axis=1)
data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.duplicated()
data=data.drop_duplicates()
data.info()


# In[7]:


z = np.abs(stats.zscore(data['cld']))
print(z)


# In[8]:


out=np.where(z>3)[0]
print('cld'+ " "+str(out))
data['cld']=data['cld'].drop(out)


# In[9]:


z = np.abs(stats.zscore(data['dtr']))
out=np.where(z>3)[0]
print('dtr'+ " "+str(out))
data['dtr']=data['dtr'].drop(out)


# In[10]:


z = np.abs(stats.zscore(data['frs']))
out=np.where(z>3)[0]
print('frs'+ " "+str(out))
data['frs']=data['frs'].drop(out)


# In[11]:


z = np.abs(stats.zscore(data['tmn']))
out=np.where(z>3)[0]
print('tmn'+ " "+str(out))
data['tmn']=data['tmn'].drop(out)


# In[12]:


z = np.abs(stats.zscore(data['tmp']))
out=np.where(z>3)[0]
print('tmp'+ " "+str(out))
data['tmp']=data['tmp'].drop(out)


# In[13]:


z = np.abs(stats.zscore(data['tmx']))
out=np.where(z>3)[0]
print('tmx'+ " "+str(out))
data['tmx']=data['tmx'].drop(out)


# In[14]:


z = np.abs(stats.zscore(data['wet']))
out=np.where(z>3)[0]
print('wet'+ " "+str(out))
data['wet']=data['wet'].drop(out)


# In[15]:


z = np.abs(stats.zscore(data['dominant_land_cover']))
out=np.where(z>3)[0]
print('dominant_land_cover'+ " "+str(out))
data['dominant_land_cover']=data['dominant_land_cover'].drop(out)


# In[16]:


data.info()


# In[17]:


data=data.fillna(method='bfill')
data.isnull().sum()


# In[18]:


data.info()


# In[19]:


data.describe()


# In[20]:


data.head()


# In[21]:




for column in data.columns:
    data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min()) 
data.head()



# In[22]:


data.info()


# In[23]:


df = pd.DataFrame(data)

# Export the Pandas DataFrame to a CSV file.
df.to_csv('dataset.csv', index=False)


# In[25]:


import matplotlib.pyplot as plt
import pandas as pd


features_to_plot = [col for col in df.columns if col not in ['x', 'y', 'lumpy']]


plt.figure(figsize=(15, 15))

# Create separate scatterplots for each feature against 'lumpy'
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(4, 4, i)  
    plt.scatter(df[feature], df['lumpy'], alpha=0.3)
    plt.xlabel(feature)
    plt.ylabel('lumpy')

plt.tight_layout()  
plt.show()


# In[26]:


# Create a subplot grid with 4 rows and 2 columns
fig, axes = plt.subplots(4, 2)

# Create a histogram of each feature in a separate subplot
for i in range(4):
    for j in range(2):
        feature = df.columns[i * 2 + j]
        axes[i, j].hist(df[feature])
        axes[i, j].set_title(feature)

# Tighten the layout of the subplots
plt.tight_layout()

# Display the plot
plt.show()


remaining_columns = list(df.columns[8:])


num_features = len(remaining_columns)
num_rows = int(num_features ** 0.5)
num_cols = int(num_features / num_rows)

fig, axes = plt.subplots(num_rows, num_cols)


for i in range(num_rows):
    for j in range(num_cols):
        feature = remaining_columns[i * num_cols + j]
        axes[i, j].hist(df[feature])
        axes[i, j].set_title(feature)


plt.tight_layout()


plt.show()


# In[27]:


x = data.iloc[:,:-1]
y = data.iloc[: ,-1]
y


# In[28]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,stratify=y)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[29]:


from sklearn.svm import SVC

svc_model = SVC(kernel='rbf',gamma=8)
svc_model.fit(x_train, y_train)


# In[50]:


from sklearn.metrics import accuracy_score, confusion_matrix
predictions= svc_model .predict(x_train)
percentage=svc_model.score(x_train,y_train)
res=confusion_matrix(y_train,predictions)


predictions= svc_model .predict(x_test)
percentage=svc_model.score(x_test,y_test)
res=confusion_matrix(y_test,predictions)
print("validation confusion matrix")
print(res)


print('testing accuracy = '+str(svc_model.score(x_test, y_test)*100))


# In[31]:


from sklearn.metrics import roc_curve, auc


predictions_train = svc_model.predict(x_train)
predictions_test = svc_model.predict(x_test)


def calculate_roc_auc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


fpr_train, tpr_train, thresholds_train, roc_auc_train = calculate_roc_auc(y_train, predictions_train)


fpr_test, tpr_test, thresholds_test, roc_auc_test = calculate_roc_auc(y_test, predictions_test)


print('Training ROC AUC score:', roc_auc_train)
print('Test ROC AUC score:', roc_auc_test)


# In[32]:


from sklearn.metrics import roc_curve, auc, precision_recall_curve
fpr, tpr, thresholds = roc_curve(y_test, predictions_test)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_test, predictions_test)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()


# In[33]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[ ]:


# Decision Tree


# In[40]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
acc = []
model = []
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(x_train,y_train)

predicted_values = RF.predict(x_test)

x = metrics.accuracy_score(y_test,predictions)
acc.append(x)
model.append('RF')
print("Random Forest's Accuracy is: ", x)

print(classification_report(y_test,predicted_values))


# In[49]:


res=confusion_matrix(y_test,predictions)
print("validation confusion matrix")
print(res)

print('testing accuracy = '+str(RF.score(x_test, y_test)*100))


# In[41]:


from sklearn.metrics import roc_curve, auc


predictions_train = RF.predict(x_train)
predictions_test = RF.predict(x_test)


def calculate_roc_auc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


fpr_train, tpr_train, thresholds_train, roc_auc_train = calculate_roc_auc(y_train, predictions_train)


fpr_test, tpr_test, thresholds_test, roc_auc_test = calculate_roc_auc(y_test, predictions_test)


print('Training ROC AUC score:', roc_auc_train)
print('Test ROC AUC score:', roc_auc_test)


# In[42]:


from sklearn.metrics import roc_curve, auc, precision_recall_curve
fpr, tpr, thresholds = roc_curve(y_test, predictions_test)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_test, predictions_test)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()


# In[ ]:


#Gradient Boosting Classifier


# In[46]:


# Import models and utility functions
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
 
# Setting SEED for reproducibility
SEED = 23
 

# Instantiate Gradient Boosting Regressor
gbc = GradientBoostingClassifier(n_estimators=300,
                                 learning_rate=0.05,
                                 random_state=100,
                                 max_features=5 )
# Fit to training set
gbc.fit(x_train, y_train)
 
# Predict on test set
pred_y = gbc.predict(x_test)
 
# accuracy
acc = accuracy_score(y_test, pred_y)
print("Gradient Boosting Classifier accuracy is : {:.2f}".format(acc))

print(classification_report(y_test,pred_y))


# In[48]:


res=confusion_matrix(y_test,pred_y)
print("validation confusion matrix")
print(res)

print('testing accuracy = '+str(gbc.score(x_test, y_test)*100))


# In[ ]:


# Performence Evaluation


# In[51]:


# Evaluate the model on the training data
import matplotlib.pyplot as plt

# Sample data (replace with your own data)
categories = ['1.SVM ', '2.Random Forest','3.Improved Gradient Boosting Classifier' ]
values = [97.03,97.34,98]

# Set the figure size
plt.figure(figsize=(18, 8))
# Create a bar chart
plt.bar(categories, values)

# Add labels and title
plt.xlabel('Algorithm Name')
plt.ylabel('Accuracy(%)')
plt.title('Performance Evaluation (Lumpy Skin Disease Prediction)')

# Annotate values inside the bars
for category, value in zip(categories, values):
    plt.text(category, value, str(value), ha='center', va='bottom')
# Display the chart
plt.show()


# In[52]:


# Evaluate the model on the training data
import matplotlib.pyplot as plt

# Sample data (replace with your own data)
categories = ['1.SVM ', '2.Random Forest','3.Improved Gradient Boosting Classifier' ]
values = [97,97,98]

# Set the figure size
plt.figure(figsize=(18, 8))
# Create a bar chart
plt.bar(categories, values)

# Add labels and title
plt.xlabel('Algorithm Name')
plt.ylabel('Precision(%)')
plt.title('Performance Evaluation (Lumpy Skin Disease Prediction)')

# Annotate values inside the bars
for category, value in zip(categories, values):
    plt.text(category, value, str(value), ha='center', va='bottom')
# Display the chart
plt.show()


# In[53]:


# Evaluate the model on the training data
import matplotlib.pyplot as plt

# Sample data (replace with your own data)
categories = ['1.SVM ', '2.Random Forest','3.Improved Gradient Boosting Classifier' ]
values = [97,97,98]

# Set the figure size
plt.figure(figsize=(18, 8))
# Create a bar chart
plt.bar(categories, values)

# Add labels and title
plt.xlabel('Algorithm Name')
plt.ylabel('Recall(%)')
plt.title('Performance Evaluation (Lumpy Skin Disease Prediction)')

# Annotate values inside the bars
for category, value in zip(categories, values):
    plt.text(category, value, str(value), ha='center', va='bottom')
# Display the chart
plt.show()


# In[54]:


# Evaluate the model on the training data
import matplotlib.pyplot as plt

# Sample data (replace with your own data)
categories = ['1.SVM ', '2.Random Forest','3.Improved Gradient Boosting Classifier' ]
values = [97,97,98]

# Set the figure size
plt.figure(figsize=(18, 8))
# Create a bar chart
plt.bar(categories, values)

# Add labels and title
plt.xlabel('Algorithm Name')
plt.ylabel('F1-Score(%)')
plt.title('Performance Evaluation (Lumpy Skin Disease Prediction)')

# Annotate values inside the bars
for category, value in zip(categories, values):
    plt.text(category, value, str(value), ha='center', va='bottom')
# Display the chart
plt.show()
