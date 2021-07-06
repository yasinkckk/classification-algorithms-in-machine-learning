 ### VERİ SETİNİN AKTARILMASI

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
gradadm= pd.read_csv("Admission_Predict_Ver1.1.csv")


# In[2]:


gradadm.shape


# In[3]:


gradadm.head()


# In[4]:


gradadm.tail()


# ### KAYIP GÖZLEM KONTROLÜ

# In[5]:


gradadm.info()


# ### TANIMLAYICI İSTATİSTİKLERİN KONTROLÜ

# In[6]:


gradadm.describe()


# ### SINIFLANDIRMA İÇİN KULLANILMAYACAK DEĞİŞKENİ ÇIKARMA

# In[2]:


gradadm= gradadm.drop(columns='Serial No.')


# In[8]:


gradadm.head()


# In[3]:


gradadm.rename(columns={'LOR ' : 'LOR', 'Chance of Admit ':'Admit'},
                 inplace=True)


# In[10]:


gradadm['LOR'].value_counts()


# In[11]:


gradadm['Admit'].value_counts()


# ### KUTU GRAFİKLERİ

# In[12]:


import matplotlib.pyplot as plt
plt.show()
plt.figure(1, figsize=(15,6)) 
plt.subplot(1,22,1) 
plt.boxplot(gradadm['GRE Score']) 
plt.title('GRE Score') 

plt.subplot(1,22,4) 
plt.boxplot(gradadm['TOEFL Score'])
plt.title('TOEFL Score')

plt.subplot(1,22,7) 
plt.boxplot(gradadm['University Rating']) 
plt.title('University Rating') 

plt.subplot(1,22,10)
plt.boxplot(gradadm['SOP'])
plt.title('SOP')

plt.subplot(1,22,13)
plt.boxplot(gradadm['LOR']) 
plt.title('LOR') 

plt.subplot(1,22,16) 
plt.boxplot(gradadm['CGPA']) 
plt.title('CGPA') 

plt.subplot(1,22,19)
plt.boxplot(gradadm['Research']) 
plt.title('Research') 

plt.subplot(1,22,22) 
plt.boxplot(gradadm['Admit']) 
plt.title('Chance of Admit') 
plt.show()


# In[13]:


gradadm


# ### KATEGORİK DEĞİŞKENLERİN KIYASLANMASI

# In[14]:


gradadm['University Rating'].value_counts().head(10).plot.bar()
plt.title('Candidates by university rating') 
plt.ylabel("Frequency")
plt.xlabel("University Rating")


# In[15]:


gradadm['Research'].value_counts().head().plot.bar()
plt.title('Candidates by Research experience') 
plt.ylabel("Frequency")
plt.xlabel("Research Experience (1- Yes, 0 - No)")


# - HİSTOGRAM GRAFİKLERİNE BAKILDIĞINDA 
# - basvurulara olan katılımen cok 3. ve 2. seviyediki üniversitelerden gelmiştir
# - araştırma deneyimi olan aday sayısı daha fazladır
# 

# ### DAĞILIM GRAFİKLERİ

# In[16]:


from pandas.plotting import scatter_matrix
attributes = ["GRE Score","TOEFL Score","University Rating","SOP","LOR","CGPA","Research","Admit"]
scatter_matrix(gradadm[attributes], figsize=(20, 10))


# - Grafiğe bakıldığında GRE puanı ile kabul şansı ile arasında pozitif bir doğrusal ilişki vardır.
# - TOEFL puanı ile kabul şansı arasında pozitif doğrusal bir ilişki vardır.
# -  CGPA puanı ile kabul şansı arasında pozitif doğrusal bir ilişki vardır.

# ### KORELASYON İNCELENMESİ

# In[17]:


gradadm.corr()


# In[18]:


print('Correlation between all variables')
plt.figure(figsize=(10, 10))
sns.heatmap(gradadm.corr(), annot=True, linewidths=0.05, fmt= '.2f',
            cmap="rainbow") 
plt.show()


# ### ÇOKLU DEĞİŞKEN İNCELENMESİ

# In[12]:


gradadm_v1 = gradadm.copy()


# In[13]:


gradadm_v1.plot(kind="scatter", x="CGPA", y="GRE Score")


# In[28]:


fig, ax = plt.subplots()
gradadm_v1.plot(kind='scatter',y='GRE Score', x='CGPA', ax=ax,label="Relationship between admit chance with CGPA and GRE score",
                figsize=(9,5),c="Admit",cmap=plt.get_cmap("jet"),colorbar=True,)
ax.set_xlabel("CGPA")
plt.show()


# In[22]:


print('Relationship between admit chance with CGPA and GRE score')
print('Relationship between admit chance with CGPA and GRE score')
def modiffy(row):
    if row['Admit'] > 0.82 :
        return 1
    else :
        return 0
gradadm_v1['Admit'] = gradadm.apply(modiffy,axis=1)
sns.scatterplot(data=gradadm_v1,x='CGPA',y='GRE Score',hue='Admit') 


# In[4]:


Ad2= gradadm.copy()
Chance = Ad2['Admit']


# In[5]:


Chance = ['low' if each < 0.4 else 'medium' if each <0.75 else 'high' for each in Chance]
Ad2.insert(8,'Chance', Chance)
Ad2


# iki score değerlerinin ortalama scorelarına rağmen yüksek kabul şansı görülüyor.
# 

# In[27]:


gradadm


# ### VERİ SETİNİ BÖLME (LOR DEĞİŞKENİ KULLANACAKSINIZ gradadm2 olarak devam edin!!!)

# In[28]:


gradadm2 = gradadm.copy()
target = gradadm2['Admit']


# In[28]:


gradadm3=gradadm.copy()
target = gradadm3['Admit']


# In[30]:


gradadm2 = gradadm2.drop(columns='Admit')


# In[29]:


gradadm3=gradadm3.drop(['Admit','LOR'],axis=1)


# In[30]:


gradadm3


# In[33]:


gradadm2


# In[31]:


target= [1 if each > 0.82 else 0 for each in target]


# In[32]:


target_feature = target.copy()


# In[33]:


target_feature = pd.DataFrame(target_feature)
target_feature.value_counts()


# In[37]:


GradAdm_onehot = pd.get_dummies(gradadm2, columns=['University Rating'])

GradAdm_onehot.head(5)


# In[36]:


GradAdm_onehot1 = pd.get_dummies(gradadm3, columns=['University Rating'])

GradAdm_onehot1.head(5)


# In[57]:


from sklearn import preprocessing

Data_df = GradAdm_onehot.copy()

Data_scaler = preprocessing.MinMaxScaler()
Data_scaler.fit(GradAdm_onehot)
Data = Data_scaler.fit_transform(GradAdm_onehot)


# In[37]:


from sklearn import preprocessing

Data_df = GradAdm_onehot1.copy()

Data_scaler = preprocessing.MinMaxScaler()
Data_scaler.fit(GradAdm_onehot1)
Data = Data_scaler.fit_transform(GradAdm_onehot1)


# In[38]:


from sklearn.model_selection import train_test_split

Data_sample_train, Data_sample_test, target_sample_train, target_sample_test = train_test_split(Data, target, 
                                                    test_size = 0.3, random_state=999,
                                                    stratify = target)

print(Data_sample_train.shape)
print(Data_sample_test.shape)


# In[60]:


gradadm_num=gradadm.drop(["University Rating","Admit","Research"],axis=1)
#df.drop(['column_nameA', 'column_nameB'], axis=1, inplace=True)


# In[39]:


gradadm_num1=gradadm3.drop(["University Rating","Research"],axis=1)
#df.drop(['column_nameA', 'column_nameB'], axis=1, inplace=True)


# In[62]:


from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rnd_clf.fit(Data, target)
for name, importance in zip(gradadm_num.iloc[1:1], rnd_clf.feature_importances_):
    print(name, "=", importance)


# In[63]:


from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rnd_clf.fit(Data, target)
for name, importance in zip(gradadm_num1.iloc[1:1], rnd_clf.feature_importances_):
    print(name, "=", importance)


# # KNN

# In[40]:


from sklearn.neighbors import KNeighborsClassifier


# In[41]:


knn_clf = KNeighborsClassifier()
knn_clf.fit(Data_sample_train, target_sample_train)


# In[42]:


y_pred=knn_clf.predict(Data_sample_train)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(target_sample_train, y_pred)
print(accuracy)


# In[43]:


from sklearn.model_selection import GridSearchCV

param_grid = [{'weights': ["uniform", "distance"] ,'n_neighbors': [3, 4, 5,10,15,20],'p':[1,2,7]}]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=10, verbose=3)
grid_search.fit(Data_sample_train, target_sample_train)


# In[44]:


y_pred = grid_search.predict(Data_sample_train)


# p=1 manhattan
# p=2 euclidean
# p=7 minkowski

# In[45]:


grid_search.best_params_


# In[46]:


grid_search.best_score_


# In[47]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(target_sample_train, y_pred)
print(cm)


#  # accuracy aucscore train

# In[48]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(target_sample_train, y_pred)
print(accuracy)


# In[49]:


from sklearn.metrics import roc_auc_score
roc_auc_score(target_sample_train, y_pred)


# # accuracy aucscore test

# In[51]:


from sklearn.metrics import accuracy_score

y_pred = grid_search.predict(Data_sample_test)
accuracy_score(target_sample_test, y_pred)


# In[52]:


from sklearn.metrics import roc_auc_score
roc_auc_score(target_sample_test, y_pred)


# In[53]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_pred, target_sample_test))


# # DECISION TREE

# In[54]:


from sklearn.tree import DecisionTreeClassifier


# In[55]:


dt_clf = DecisionTreeClassifier()
dt_clf.fit(Data_sample_train, target_sample_train)


# In[56]:


y_pred=dt_clf.predict(Data_sample_train)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(target_sample_train, y_pred)
print(accuracy)


# In[57]:


from sklearn.model_selection import GridSearchCV

param_grid = [{'criterion': ["gini", "entropy"],'splitter':["best","random"],'max_depth':[1,2,3]}]
dt_clf = DecisionTreeClassifier()
grid_search = GridSearchCV(dt_clf, param_grid, cv=10, verbose=3)
grid_search.fit(Data_sample_train, target_sample_train)


# In[58]:


grid_search.best_params_


# In[59]:


grid_search.best_score_


# In[ ]:





# # accuracy aucscore train

# In[60]:


y_pred=grid_search.predict(Data_sample_train)


# In[61]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(target_sample_train, y_pred))


# In[62]:


from sklearn.metrics import roc_auc_score
roc_auc_score(target_sample_train, y_pred)


# # accuracy aucscore test

# In[63]:


from sklearn.metrics import accuracy_score

y_pred = grid_search.predict(Data_sample_test)
accuracy_score(target_sample_test, y_pred)


# In[64]:


from sklearn.metrics import roc_auc_score
roc_auc_score(target_sample_test, y_pred)


# In[65]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_pred, target_sample_test))


# # graph

# In[66]:


import numpy
gradadm_target_names=numpy.array(['1','0'])


# In[67]:


gradadm_target_names
list(gradadm_target_names)


# In[68]:


GradAdm_onehot1.iloc[1:1]


# In[69]:


a=DecisionTreeClassifier(criterion='gini', max_depth= 2, splitter= 'best')


# In[70]:


gradadm3_feature_names=['GRE Score','TOEFL Score','SOP','CGPA','Research','University Rating_1','University Rating_2','University Rating_3','University Rating_4','University Rating_5']


# In[71]:


type(gradadm3_feature_names)


# In[72]:


b=gradadm3_feature_names[:2]


# In[73]:


x=Data_sample_train[:,:2]
dt_clf.fit(x,target_sample_train)


# In[74]:


a.fit(Data_sample_train,target_sample_train)


# In[75]:



from sklearn.tree import export_graphviz

export_graphviz(
        a,
        out_file="gradadm1.dot",
        feature_names=gradadm3_feature_names,
        class_names=gradadm_target_names,
        rounded=True,
        filled=True
       
    )


# ### png cinsinden grafik kaydedilmiştir 

# In[78]:


get_ipython().system('dot -Tpng gradadm1.dot -o gradadm1.png')


# In[79]:


X = Data_sample_train[:, 2:] # petal length and width
y = target_sample_train

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)


# In[ ]:





# # RandomForest

# In[80]:


from sklearn.ensemble import RandomForestClassifier


# In[81]:


from sklearn.model_selection import GridSearchCV

param_grid = [{'n_estimators':[20,50,100],'criterion': ["gini", "entropy"],'max_depth':[7,8,9,10]}]
rf_clf =  RandomForestClassifier()
grid_search = GridSearchCV(rf_clf, param_grid, cv=10, verbose=3)
grid_search.fit(Data_sample_train, target_sample_train)


# In[82]:


grid_search.best_params_


# In[83]:


grid_search.best_score_


# In[84]:


y_pred=grid_search.predict(Data_sample_train)


# In[85]:


from sklearn.metrics import roc_auc_score
roc_auc_score(target_sample_train, y_pred)


# In[86]:


from sklearn.metrics import accuracy_score
accuracy_score(target_sample_train, y_pred)


# In[87]:


a=RandomForestClassifier(criterion='gini', max_depth= 8, n_estimators= 100)


# In[88]:


a.fit(Data_sample_train,target_sample_train)


# In[ ]:





# In[89]:


from sklearn.metrics import accuracy_score

y_pred = grid_search.predict(Data_sample_test)
accuracy_score(target_sample_test, y_pred)


# In[90]:


from sklearn.metrics import roc_auc_score
roc_auc_score(target_sample_test, y_pred)


# In[91]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_pred, target_sample_test))


# In[ ]:





# # LOGISTIC REGRESSION

# In[92]:


from sklearn.linear_model import LogisticRegression


# In[93]:


y_test = np.array(target_sample_test)


# In[94]:


logistic_regression= LogisticRegression()
logistic_regression.fit(Data_sample_train,target_sample_train)
y_pred=logistic_regression.predict(Data_sample_test)
y_pred


# In[95]:


from sklearn.metrics import classification_report, confusion_matrix


# In[96]:


print(classification_report(y_pred, y_test))


# In[97]:


from sklearn.metrics import roc_auc_score
roc_auc_score(target_sample_test, y_pred)


# In[98]:


from sklearn.metrics import accuracy_score

accuracy_score(target_sample_test, y_pred)


# In[99]:


conda install -c anaconda pydot


# # support vector machine

# In[100]:


from sklearn.svm import SVC


# In[101]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
svm_clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
svm_clf.fit(Data_sample_train, target_sample_train)


# In[102]:


from sklearn.svm import SVC


# In[103]:


from sklearn.model_selection import GridSearchCV

param_grid = [{'kernel':["linear","poly","rbf","sigmoid"],'C':[0.25,0.30,0.50],'gamma':["auto","scale"],
              'degree':[1,2,3,4]}]
svm_clf =  svm.SVC()
grid_search = GridSearchCV(svm_clf, param_grid, cv=10, verbose=3)
grid_search.fit(Data_sample_train, target_sample_train)


# In[104]:


grid_search.best_params_


# In[105]:


grid_search.best_score_


# In[106]:


y_pred=grid_search.predict(Data_sample_test)


# In[107]:


from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(target_sample_test, y_pred))


# In[108]:


from sklearn.metrics import roc_auc_score
roc_auc_score(target_sample_test, y_pred)


# In[109]:


from sklearn import metrics 
import matplotlib.pyplot as plt  
metrics.plot_roc_curve(grid_search, Data_sample_test, target_sample_test) 
plt.show()  


# In[110]:


from sklearn.metrics import accuracy_score

y_pred = grid_search.predict(Data_sample_test)
accuracy_score(target_sample_test, y_pred)


# In[ ]:





# # XGBOOST

# In[111]:


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[112]:


get_ipython().system('pip install xgboost')


# In[113]:


import xgboost as xgb
from xgboost import XGBClassifier


# In[114]:


xgb_clf=XGBClassifier()


# In[115]:


xgb_clf.fit(Data_sample_train,target_sample_train)


# In[116]:


y_pred=xgb_clf.predict(Data_sample_train)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(target_sample_train, y_pred)
print(accuracy)


# In[117]:


y_pred=xgb_clf.predict(Data_sample_test)
accuracy=accuracy_score(target_sample_test,y_pred)
print(accuracy)


# In[118]:


from sklearn.model_selection import GridSearchCV

param_grid = [{'min_child_weight':[1,3,5,7],'eta':[0.1,0.2,0,3],'max_depth':[3,5,7,9,11]}]
xvg_clf =  XGBClassifier()
grid_search = GridSearchCV(xvg_clf, param_grid, cv=10, verbose=3)
grid_search.fit(Data_sample_train, target_sample_train)


# In[119]:


grid_search.best_params_


# In[120]:


grid_search.best_score_


# In[121]:


y_pred=grid_search.predict(Data_sample_test)


# In[122]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(target_sample_test, y_pred))


# In[123]:


from sklearn.metrics import roc_auc_score
roc_auc_score(target_sample_test, y_pred)


# In[124]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_pred, target_sample_test))
