# In[]

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("Vertebral.csv")
df.head()

#coloca o uma fração do dataframe em df de froma aleatória, como frac=1 coloca tudo
df.sample(frac=1)
#cria um data frame temporario sem a coluna da classe
df_temp = pd.DataFrame(df,columns=df.columns[:-1])


#separa o dataframe sem a classe em treino teste e validacao
from sklearn.model_selection import train_test_split
x_treino,x_teste,y_treino,y_teste=train_test_split(df_temp,df['Class'],test_size=0.5,train_size=0.5)
x_treino,x_validacao,y_treino,y_validacao=train_test_split(x_teste,df['Class'],test_size=0.5,train_size=0.5)


x_treino.info()
x_teste.info()
x_validacao.info()

# In[19]:

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_treino,y_treino)





pred = knn.predict(x_teste)
pred


# In[22]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_teste,pred))


# In[23]:


confusion_matrix(y_teste,pred)


# In[24]:


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_teste,pred,pos_label=2)
metrics.auc(fpr,tpr)


# In[26]:


tx_erro = []
for i in range (1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_treino,y_treino)
    pred = knn.predict(x_teste)
    tx_erro.append(np.mean(pred!=y_teste))
tx_erro 
    


# In[28]:


plt.figure (figsize=(14,8))
plt.plot(range(1,50),tx_erro,color='blue',linestyle='dashed',marker='o')
plt.xlabel('K')
plt.ylabel('Erro')


# In[30]:


min(tx_erro)
np.argmin(tx_erro)+1


# In[ ]:




