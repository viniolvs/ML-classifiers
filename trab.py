import pandas as pd
import numpy as np
import seaborn as sns
import statistics
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.tree import DecisionTreeClassifier, export_graphviz # Árvore de decisão
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.svm import SVC # SVM
from sklearn.neural_network import MLPClassifier # MLP

#Importa a base
df = pd.read_csv("Vertebral.csv")
#cria um dataframe somente com a coluna das classes
df_class = pd.DataFrame(df,columns=['Class'])
##cria um dataframe sem a coluna das classes
df_temp = pd.DataFrame(df,columns=df.columns[:-1])


df.head()
df_temp.plot()
df_class.plot()

x_treino, x_resto, y_treino, y_resto = train_test_split(df_temp,df['Class'],test_size=0.5, stratify=df['Class'], shuffle=True)
x_validacao, x_teste, y_validacao, y_teste = train_test_split(x_resto, y_resto, test_size=0.5, stratify=y_resto, shuffle=True)

print("Dataframe 100%")
print(df.shape)

print("Treino 50%")
print(x_treino.shape)

print("Teste 25%")
print(x_teste.shape)

print("Validação 25%")
print(x_validacao.shape)

print("MLP")
# Inicializa um array para acuracia
acuracia = []

# Varia as épocas de treino de 100 em 100 até 10000
# Varia camadas escondidades de 50 em 50 até 5000
# Varia as taxas de aprnedizagem de 0.0 ate 1.0
# Testa todos com todos
for i in range (10):
    for j in range (5):
        for k in range (5):
            mlp = MLPClassifier(max_iter=((k+1)*200) , hidden_layer_sizes=((j+1)*100), learning_rate_init=((i+1)/10) )
            mlp.fit(x_treino, y_treino)
            pred = mlp.predict(x_validacao)
            acuracia.append(accuracy_score(y_validacao, pred))

melhor_mlp = np.argmax(acuracia)+1
print("Acurácias: ",*acuracia, sep=' | ')

melhor_epoca_treino = (melhor_mlp%5)*200 
melhor_camada_escondida = (melhor_mlp%5)*100
melhor_taxa_de_aprendizagem = (melhor_mlp%10)/10
print("Melhor época de treino = ", melhor_epoca_treino)
print("Melhor camada escondida = ",melhor_camada_escondida)
print("Melhor taxa deaprendizagem = ",melhor_taxa_de_aprendizagem)
print("Acuracia com os melhores parametros = ", acuracia[melhor_mlp-1]*100,'%')