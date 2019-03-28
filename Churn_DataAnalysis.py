
# coding: utf-8

# In[104]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import  Image
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[105]:


data = pd.read_csv(r"D:/Development/TechMahindra/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# # Data Insights
# 
# In the data provided, Churn(Yes/No) is visualised against all other features of the data to decide on the features to be used for the Classification Model.

# In[106]:


data.info()

#Data PreProcessing :

For the data to be compatible with Machine Learning algorithm, Preprocessing is performed.
1. In the 'SeniorCitizen' column 1 or 0 is replaced for Yes or No
2. 'No Internet Service' is replaced with No in few columns as both No and 'No internet service' mean the same
3. TotalCharges column had few blanks, which are being replaced with 'nan'
4. Tenure is divided into categories and the values are updated in the dataframe
# In[107]:


data["SeniorCitizen"] = data["SeniorCitizen"].replace({1:"Yes",0:"No"})


# In[108]:


columns_to_replace = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies']
for i in columns_to_replace : 
    data[i]  = data[i].replace({'No internet service' : 'No'})


# In[109]:


data['TotalCharges'] = data["TotalCharges"].replace(" ",np.nan)


# In[111]:


def tenure_category(data) :
    
    if data["tenure"] <= 12 :
        return "Tenure_0-12"
    elif (data["tenure"] > 12) & (data["tenure"] <= 24 ):
        return "Tenure_12-24"
    elif (data["tenure"] > 24) & (data["tenure"] <= 48) :
        return "Tenure_24-48"
    elif (data["tenure"] > 48) & (data["tenure"] <= 60) :
        return "Tenure_48-60"
    elif data["tenure"] > 60 :
        return "Tenure_gt_60"
data["tenure_category"] = data.apply(lambda data:tenure_category(data),axis = 1)

#Average TotalCharge is found for each tenure Category and the blanks in TotalCharge are replaced with average TotalCharge for the respective tenure category
# In[112]:


data_copy = data.copy()


# In[113]:


data.groupby(['tenure_category']).mean()


# In[114]:


data_copy = data_copy[data_copy["TotalCharges"].notnull()]
data_copy = data_copy.reset_index()[data_copy.columns]


# In[115]:


sns.heatmap(data_copy.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[116]:


data_copy["TotalCharges"] = data_copy["TotalCharges"].astype(float)
Avg_totalcharge = data_copy.groupby(['tenure_category']).mean()
Avg_totalcharge.loc['Tenure_0-12']['TotalCharges']


# In[117]:


def impute_TotalCharges(cols):
    TotalCharges = cols[0]
    tenure_category = cols[1]
    if pd.isnull(TotalCharges):
        return Avg_totalcharge.loc[tenure_category]['TotalCharges']
    else:
        return TotalCharges


# In[118]:


data['TotalCharges']=data[['TotalCharges','tenure_category']].apply(impute_TotalCharges,axis=1)

#Data is checked for Null values
# In[119]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[120]:


#Cross-validation of TotalCharge update
data[(data.customerID == '4075-WKNIU')]


# In[121]:


data["TotalCharges"] = data["TotalCharges"].astype(float)


# In[122]:


sns.set_style('whitegrid')
sns.countplot(x='Churn',data=data,palette='RdBu_r')


# In[123]:


import cufflinks as cf
cf.go_offline()
data['Churn'].iplot(kind='hist',title="Churn count")


# In[124]:


sns.set_style('whitegrid')
sns.countplot(x='Churn',hue='gender',data=data,palette='rainbow').set_title("Gender Distribution")


# In[125]:


sns.countplot(x='Churn',hue='SeniorCitizen',data=data,palette='rainbow').set_title("SeniorCitizen Distribution")


# In[126]:


sns.countplot(x='Churn',hue='Partner',data=data,palette='rainbow').set_title("Partner Distribution")


# In[127]:


sns.countplot(x='Churn',hue='Dependents',data=data,palette='rainbow').set_title("Dependents Distribution")


# In[128]:


sns.countplot(x='Churn',hue='PhoneService',data=data,palette='rainbow').set_title("PhoneService Distribution")


# In[129]:


sns.countplot(x='Churn',hue='MultipleLines',data=data,palette='rainbow').set_title("MultipleLines Distribution")


# In[130]:


sns.countplot(x='Churn',hue='InternetService',data=data,palette='rainbow').set_title("InternetService Distribution")


# In[131]:


sns.countplot(x='Churn',hue='OnlineSecurity',data=data,palette='rainbow').set_title("OnlineSecurity Distribution")


# In[132]:


sns.countplot(x='Churn',hue='OnlineBackup',data=data,palette='rainbow').set_title("OnlineBackup Distribution")


# In[133]:


sns.countplot(x='Churn',hue='DeviceProtection',data=data,palette='rainbow').set_title("DeviceProtection Distribution")


# In[134]:


sns.countplot(x='Churn',hue='TechSupport',data=data,palette='rainbow').set_title("TechSupport Distribution")


# In[135]:


sns.countplot(x='Churn',hue='StreamingTV',data=data,palette='rainbow').set_title("StreamingTV Distribution")


# In[136]:


sns.countplot(x='Churn',hue='StreamingMovies',data=data,palette='rainbow').set_title("StreamingMovies Distribution")


# In[137]:


sns.countplot(x='Churn',hue='Contract',data=data,palette='rainbow').set_title("Contract Distribution")


# In[138]:


sns.countplot(x='Churn',hue='PaperlessBilling',data=data,palette='rainbow').set_title("PaperlessBilling Distribution")


# In[139]:


sns.countplot(x='Churn',hue='PaymentMethod',data=data,palette='rainbow').set_title("PaymentMethod Distribution")


# In[140]:


sns.countplot(x='Churn',hue='tenure_category',data=data,palette='rainbow').set_title("tenure_ategory Distribution")


# In[141]:


sns.set(rc={'figure.figsize':(18,8.27)})
sns.countplot(x='tenure',hue='Churn',data=data,palette='rainbow').set_title("tenure Distribution")


# In[142]:


sns.set(rc={'figure.figsize':(20,8.27)})
sns.countplot(x='MonthlyCharges',hue='Churn',data=data,palette='rainbow').set_title("MonthlyCharges Distribution")


# In[143]:


sns.set(rc={'figure.figsize':(20,8.27)})
sns.countplot(x='TotalCharges',hue='Churn',data=data,palette='rainbow').set_title("TotalCharges Distribution")

#Numerical Columns are plotted in scater plot for the Churn values
# In[144]:


sns.set(rc={'figure.figsize':(10,5)})
sns.pairplot(data, 
             vars = ['tenure', 'MonthlyCharges', 'TotalCharges'], 
             hue = 'Churn', diag_kind = 'hist', 
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
             size = 3);


# In[145]:


sns.countplot(x='tenure_category',hue='Churn',data=data,palette='rainbow').set_title("tenure_ategory Distribution")

#The categorical columns are converted to dummies, which is the process to handle categorical values for a classifier model.
# In[146]:


data = pd.get_dummies(data, columns=['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','StreamingTV','SeniorCitizen'])


# In[147]:


data = pd.get_dummies(data, columns=['tenure_category'])

#Scaling of numerical values
# In[148]:


std = StandardScaler()
ncols=['tenure','MonthlyCharges','TotalCharges']
scaled = std.fit_transform(data[ncols])
scaled = pd.DataFrame(scaled,columns=ncols)


# In[149]:


data_copy=data.copy()
data = data.drop(columns = ncols,axis = 1)


# In[150]:


data = data.merge(scaled,left_index=True,right_index=True,how = "left")

#Finding the correlation between features
# In[151]:


correlation = data.corr()
correlation


# In[162]:


matrix_cols = correlation.columns.tolist()
matrix_cols


# In[153]:


corr_array  = np.array(correlation)
corr_array


# # Logistic Regression Model

# In[154]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import precision_score,recall_score

#30% data is randomly chosen as test data
# In[155]:


train,test = train_test_split(data,test_size=.30 ,random_state=50)
Id_col     = ['customerID']
target_col = ["Churn"]


# In[156]:


cols    = [i for i in data.columns if i not in Id_col + target_col]
train_X = train[cols]
train_Y = train[target_col]
test_X  = test[cols]
test_Y  = test[target_col]


# In[157]:


logmodel = LogisticRegression()


# In[158]:


logmodel.fit(train_X,train_Y)


# In[159]:


predictions=logmodel.predict(test_X)
print(classification_report(test_Y,predictions))

The model prediction is 80.70% accurate
# In[160]:


print(accuracy_score(test_Y,predictions))


# In[161]:


print(confusion_matrix(test_Y,predictions))

According to the confusion matrix :
* (1390+315) ie 1705 customers were predicted correctly
* (242+166) ie 408 customers were predicted incorrectlyBased on the above data visualisation and analysis. Below are our recommendation on which customers don't tenure:
1. Customers who are not senior citizens
2. Customers who are partners
3. Who have phone service
4. Customers with Interner service as DSL or no internet service
5. Customers with contract of 1 or 2 years
6. Customers who use the payment methods as Mailed cheque, Bank transfer, Credit card
7. Customers whose tenure is greater than 60For a telecommunication company, it is more important to retain the customers than getting new customers. Hence the company has to focus on the customers in the above category.