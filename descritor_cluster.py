# imports
import pickle
import pandas as pd

# abrir modelo de clusters
cluster_model = pickle.load(open('cluster_housing_data.pkl', 'rb'))

# abrir normalizador numérico salvo anteriormente
normalizador = pickle.load(open('normalizador_housing_data.pkl', 'rb'))

# DESNORMALIZAR OS CENTRÓIDES

# obter os nomes das colunas
columns_names = pd.read_csv('HousingData.csv', nrows=0).columns.tolist()

# converter os centroides em dataframe
dataframe = pd.DataFrame(cluster_model.cluster_centers_,
                         columns = columns_names)

# desnormalizar os centróides
atributos_desnorm = pd.DataFrame(normalizador
                   .inverse_transform(
                       dataframe[columns_names]
                    ), columns=columns_names
                )

print(atributos_desnorm)