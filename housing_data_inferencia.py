import pandas as pd
import pickle

#dados para criar os dataframes
columns_names = pd.read_csv('HousingData.csv', nrows=0).columns.tolist()

# novo dado
novo_bairro = [[
    0.08,   # CRIM
    25.0,   # ZN
    5.0,    # INDUS
    0,      # CHAS
    0.45,   # NOX
    6.8,    # RM
    45.0,   # AGE
    5.0,    # DIS
    4.0,    # RAD
    300.0,  # TAX
    16.0,   # PTRATIO
    390.0,  # B
    8.0,    # LSTAT
    24.0    # MEDV
]]


# abrir normalizador
normalizador = pickle.load(open('normalizador_housing_data.pkl', 'rb'))

# abrir o modelo salvo
cluster_housing_data = pickle.load(open('cluster_housing_data.pkl', 'rb'))

# Normalizar os dados de entrada
novo_dado_norm = normalizador.transform(novo_bairro)

# converter a nova instância normalizada em dataframe
novo_dataframe = pd.DataFrame(novo_dado_norm, columns=columns_names)


cluster_bairro = cluster_housing_data.predict(novo_dataframe)
print("Cluster do novo bairro:")
print(cluster_bairro)