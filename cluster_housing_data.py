import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from receber_dados import preencher, eh_categorica, eh_normal

#carregar dataset
dados = pd.read_csv('HousingData.csv', sep=',')

dados = preencher(dados) #preencher NaN

#NORMALIZAÇÃO

scaler = MinMaxScaler() #instanciando normalizador

normalizador = scaler.fit(dados) #treinar normalizador com dados

pickle.dump(normalizador, open('normalizador_housing_data.pkl', 'wb')) #salvar normalizador para uso posterior

dados_norm = normalizador.transform(dados) #normalizar os dados

#converter a matriz numérica dados_norm em DataFrame

dados_dataframe = pd.DataFrame(dados_norm, columns=dados.columns)

#HIPERPARAMETRIZAÇÃO - Determinar o número ótimos de clusters antes do treinamento

from sklearn.cluster import KMeans # kmeans é um clusterizador
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist # método para cálculo de distâncias cartesianas
import numpy as np

distortions = [] #matriz para armazenar as distorções

K = range(1, dados.shape[0])

for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42).fit(dados_dataframe)

    # calcular e armazenar a distorção de cada treinamento
    distortions.append(
        sum(
            np.min(
                cdist(dados_dataframe, cluster_model.cluster_centers_, 'euclidean'),
                  axis=1)/dados.shape[0]
            )
        )
    
# Determinar o número ótimo de clusters para o modelo
x0 = K[0]
y0 = distortions[0]
xn = K[-1]
yn = distortions[-1]
distances = []
for i in range(len(distortions)):
    x = K[i]
    y= distortions[i]
    numerador = abs(
        (yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0
    )
    denominador = math.sqrt(
        (yn-y0)**2 + (xn-x0)**2
    )
    distances.append(numerador/denominador)

numero_clusters_otimo = K[distances.index(np.max(distances))]

# Treinar o modelo com o número ótimo
cluster_model = KMeans(n_clusters= numero_clusters_otimo, random_state=42).fit(dados_dataframe)

#salvar o modelo para uso posterior
pickle.dump(cluster_model, open('cluster_housing_data.pkl', 'wb'))

print(f"Número ótimo de clusters: {numero_clusters_otimo}")