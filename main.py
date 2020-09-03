import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

arquivo = pd.read_csv("C:\BSI\Python programador\Website Phishing.csv")
arquivo.isnull().sum()
# arquivo.info()
# print(arquivo.describe())
# print(arquivo.groupby(by="Result").size())
# print(arquivo.groupby(by="web_traffic").size())

x = arquivo.iloc[:, :-1]
y = arquivo.iloc[:, -1]
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.40, random_state=42)

arvore = RandomForestClassifier()
arvore.fit(x_treino, y_treino)
resultadoRF = arvore.predict(x_teste)
tabelaDeFrequenciaRF = pd.crosstab(y_teste, resultadoRF, rownames=["Reais"], colnames=["Previstos"],
                                   margins=True)

# print(tabelaDeFrequenciaRF)


naive_bayes = GaussianNB()
naive_bayes.fit(x_treino, y_treino)
resultadoNB = naive_bayes.predict(x_teste)
tabelaDeFrequenciaNB = pd.crosstab(y_teste, resultadoNB, rownames=["Reais"], colnames=['Previstos'],
                                   margins=True)

KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(x_treino, y_treino)
resultadoKNN = KNN.predict(x_teste)
tabelaDeFrequenciaKNN = pd.crosstab(y_teste, resultadoKNN, rownames=['Reais'], colnames=['Previstos'],
                                    margins=True)

# tabelaDeMetricasNB = metrics.classification_report(y_teste, resultadoNB)
# print(tabelaDeMetricasNB)
# tabelaDeMetricasKNN = metrics.classification_report(y_teste, resultadoKNN)
# print(tabelaDeMetricasKNN)
# tabelaDeMetricasRF = metrics.classification_report(y_teste, resultadoRF)
# print(tabelaDeMetricasRF)

accRandomForest, accNB, accKNN = accuracy_score(y_teste, resultadoRF), accuracy_score(y_teste, resultadoNB),\
                               accuracy_score(y_teste, resultadoKNN)
listaAccuracy = [accRandomForest, accNB, accKNN]

recallRF, recallNb, recallKNN = recall_score(y_teste, resultadoRF, average="macro"), \
                                recall_score(y_teste, resultadoNB, average="macro"), \
                                recall_score(y_teste, resultadoKNN, average="macro")
listarecall = [recallRF, recallKNN, recallNb]

precRandomForest, precNB, precKNN = precision_score(y_teste, resultadoRF, average="macro"), \
                                    precision_score(y_teste, resultadoNB, average="macro"), \
                                    precision_score(y_teste, resultadoKNN, average="macro")
listaPrecision = [precRandomForest, precNB, precKNN]

f1Random, f1NB, f1KNN = f1_score(y_teste, resultadoRF, average="macro"), \
                        f1_score(y_teste, resultadoNB, average="macro"), \
                        f1_score(y_teste, resultadoKNN, average="macro")
listaf1 = [f1Random, f1NB, f1KNN]

plt.title("Gráfico de comparação dos algoritmos")
plt.plot(listaAccuracy, marker='s')
plt.plot(listarecall, marker='s')
plt.plot(listaPrecision, marker='s')
plt.plot(listaf1, marker='s')
plt.xlabel("0: Random Fores 1: Naive Bayes 2: KNN")
plt.ylabel("Valores")
plt.legend(["Accuracy", "Recall", "Precision", "F1"])
plt.show()
