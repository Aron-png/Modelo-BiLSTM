#python -m venv bilstm
#bilstm\Scripts\activate
'''
Recomiendo instalar uno por uno:
pip install pandas
pip install openpyxl
pip install numpy
pip install scikit-learn
pip install torch
pip install tensorflow
pip install tensorflow_hub
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow_hub as hub
import tensorflow as tf

# Cargar los datos del archivo Excel
data = pd.read_excel('input1.xlsx')
#data = data[data['ID'].isin(data['ID'].unique()[:500])] #Para entrenar con los primeros 150 IDs
# Filtrar filas donde la etiqueta BIO no sea 'O'
# filtered_data = data[data['Etiqueta BIO'].str.startswith('B-')]

# Filtrar las columnas necesarias
filtered_data = data[['ID', 'Lematizado', 'Etiqueta BIO', 'Sentimiento_palabra']]

# Asegurar que las predicciones y etiquetas tengan el mismo tamaño
filtered_data = filtered_data.dropna()

# Agrupar por ID para mantener consistencia en la partición
grouped_data = filtered_data.groupby('ID').agg(list)
# Función para verificar si todos los elementos son 'O'
def extrae_ceros(label_list):
    return all(label == 'O' for label in label_list)

# Filtrar el DataFrame
#   ponemos ~ para reinvertir resultados, donde retornaran las dilas que no cumple esa condicion
grouped_data = grouped_data[~grouped_data['Etiqueta BIO'].apply(extrae_ceros)]
#grouped_data.to_excel('prueba2.xlsx', index=False)

# Dividir los IDs en 80% para entrenamiento y 20% para evaluación
train_ids, test_ids = train_test_split(grouped_data.index, test_size=0.2, random_state=42)

train_data = grouped_data.loc[train_ids]
test_data = grouped_data.loc[test_ids]

#============== PRE PROCESADO
df = pd.read_excel('resultados del procesamiento2.xlsx')
df_lista = df.set_index('ID_producto').comentario.str.split(expand=True).stack().reset_index(level=1, drop=True).reset_index()
df_lista.columns = ['ID_producto', 'comentario']
grupo_datos = df_lista.groupby('ID_producto').agg(list)
#============== PRE PROCESADO

# Cargar el modelo ELMo desde TensorFlow Hub
elmo = hub.load("https://tfhub.dev/google/elmo/3")
#Fuente:
#https://medium.com/@harsh.vardhan7695/a-comprehensive-guide-to-word-embeddings-in-nlp-ee3f9e4663ed 
# Función para obtener embeddings ELMo
def elmo_embeddings(sentences):
    return elmo.signatures['default'](tf.constant(sentences))['elmo']

# Ejemplo de cómo convertir una lista de palabras a embeddings ELMo
def get_elmo_embeddings(sentences):
    embeddings = elmo_embeddings(sentences)
    return embeddings.numpy()

#Fuente (Bidirectional RNN):
#https://medium.com/@reddyyashu20/bidirectional-rnn-python-code-in-keras-and-pytorch-22b9a9a3c034
#input_size = número de dimensiones q tiene cada texto
#hidden_size = Número de neuronas ocultas (128, 256 o 512)
# A medida que aumenta, tiene mayor capacidad de capturar dependencia entre textos, pero + costo compu.
#num_classes_ner = numero de palabras a clasificar

#Salida del modelo: (batch_first(), sequence_length, input_size=1024)
#dropout_rate (Tasa de abandono) = Numero de neuronas que se abandonan en el entrenamiento
#bath = son el número de datos que coge el modelo para entrenarlo de forma simultanea
#sequence_length = len(texto)
# Indicado en ELMO: "lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024]"
# https://tfhub.dev/google/elmo/3

class BiLSTMNERModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes_ner, num_classes_sentiment, dropout_rate):
        super(BiLSTMNERModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)  # Agregar Dropout
        self.fc_ner = nn.Linear(hidden_size * 2, num_classes_ner)  # Salida para NER
        self.fc_sentiment = nn.Linear(hidden_size * 2, num_classes_sentiment)  # Salida para sentimiento

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)  # Aplicar Dropout
        ner_out = self.fc_ner(lstm_out)
        sentiment_out = self.fc_sentiment(lstm_out)
        return ner_out, sentiment_out

# Función para mapear etiquetas a índices
#Numero de clases NER
df_ner = filtered_data['Etiqueta BIO'].unique()
# Número de clases de sentimiento
df_sent = filtered_data['Sentimiento_palabra'].unique()

# Se pone una id y luego el valor de la etiqueta en un diccionario.
# Luego se retorna la etiqueta que pertenece el texto, si dice camara tendra etique B-camara pero con una id.
# De modo que el texto se interprete como numeros.
def bio_label_to_index(label):
    indice_bio = {}
    for idx, valor in enumerate(df_ner):
        indice_bio[valor] = idx
    return indice_bio[label]

def sentiment_label_to_index(label):
    sentiment_to_index = {}
    for idx, valor in enumerate(df_sent):
        sentiment_to_index[valor] = idx
    return sentiment_to_index[label]

# Configuración del modelo
input_size = 1024  # ELMo produce embeddings de 1024 dimensiones
hidden_size = 512 #256 - 512
num_classes_ner = len(df_ner)  # Número de clases de NER
num_classes_sentiment = len(df_sent)  # Número de clases de sentimiento

# Almacenar los resultados en una lista
results = []

clase1 = df_ner.tolist()
print("lista clase 1: ",clase1)

clase2 = df_sent.tolist()
print("lista clase 2: ",clase2)
# Probar diferentes tasas de abandono
dropout_rate = 0.75
#==================== Función de entrenamiento
#Gradientes: Se optimiza los parámetros del modelo basandose en gradientes calculados.
#Estos gradientes se deben calcular por separado, x eso se reinicia en cada iteración con optimizer.zero_grad()
#Fuente: https://github.com/hannirio/Chinese-sentiment-analysis/blob/main/BERT.final.py
#==================0.25, 0.76, 0.05 - 0.55, 0.76, 0.05
print(f"Entrenando con Dropout Rate: {dropout_rate}")
model = BiLSTMNERModel(input_size, hidden_size, num_classes_ner, num_classes_sentiment, dropout_rate)
criterion_ner = nn.CrossEntropyLoss()
criterion_sentiment = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Función para entrenar el modelo
def train_model(model, train_data, epochs=50):#=========================20
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for idx, row in train_data.iterrows():
            sentences = row['Lematizado']
            bio_labels = row['Etiqueta BIO']
            sentiment_labels = row['Sentimiento_palabra']

            # Obtener embeddings ELMo
            embeddings = get_elmo_embeddings(sentences)
            embeddings = torch.tensor(embeddings).float()

            # Convertir las etiquetas a tensores
            #       Funcion 1 para NER:
            bio_labels_tensor = []
            for valor in bio_labels:
                bio_labels_tensor.append(bio_label_to_index(valor))
            bio_labels_tensor = torch.tensor(bio_labels_tensor)#Lo convierte a tensor
            #       Funcion 2 para ABSA:
            sentiment_labels_tensor = []
            for valor in sentiment_labels:
                sentiment_labels_tensor.append(sentiment_label_to_index(valor))
            sentiment_labels_tensor = torch.tensor(sentiment_labels_tensor)

            # Limpiar el gradiente
            optimizer.zero_grad()

            # Forward pass
            ner_out, sentiment_out = model(embeddings)

            # Calcular la pérdida
            loss_ner = criterion_ner(ner_out.view(-1, num_classes_ner), bio_labels_tensor)
            loss_sentiment = criterion_sentiment(sentiment_out.view(-1, num_classes_sentiment), sentiment_labels_tensor)
            loss = loss_ner + loss_sentiment

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_data)}')

# Entrenar el modelo
train_model(model, train_data)


# Evaluar el modelo
def evaluate_model(model, test_data):
    model.eval()
    lista_bio_predicciones = []
    lista_bio_test = []
    lista_sentimiento_test = []
    lista_sentimiento_predicciones = []
    
    with torch.no_grad():
        for idx, row in test_data.iterrows():
            sentences = row['Lematizado']
            bio_labels = row['Etiqueta BIO']
            sentiment_labels = row['Sentimiento_palabra']

            embeddings = get_elmo_embeddings(sentences)
            embeddings = torch.tensor(embeddings).float()

            # Forward pass
            ner_out, sentiment_out = model(embeddings)
            # Obtener las predicciones
            ner_preds = torch.argmax(ner_out, dim=-1).cpu().numpy()
            sentiment_preds = torch.argmax(sentiment_out, dim=-1).cpu().numpy()

            lista_bio_predicciones.extend(ner_preds)#Lista de predicciones del modelo en NER
            for value in bio_labels:
                lista_bio_test.extend([bio_label_to_index(value)])#Lista de test para evaluar al modelo en NER

            lista_sentimiento_predicciones.extend(sentiment_preds)#Lista de predicciones del modelo en sentimiento
            for value in sentiment_labels:
                lista_sentimiento_test.extend([bio_label_to_index(value)])#Lista de test para evaluar al modelo en sentimiento

    # Calcular métricas
    accuracy_ner = accuracy_score(lista_bio_test, lista_bio_predicciones)
    precision_ner, recall_ner, f1_ner, _ = precision_recall_fscore_support(lista_bio_test, lista_bio_predicciones, average='macro')

    accuracy_sentiment = accuracy_score(lista_sentimiento_predicciones, lista_sentimiento_test)
    precision_sentiment, recall_sentiment, f1_sentiment, _ = precision_recall_fscore_support(lista_sentimiento_predicciones, lista_sentimiento_test, average='macro')

    conf_matrix_ner = confusion_matrix(lista_bio_test, lista_bio_predicciones)
    conf_matrix_sentiment = confusion_matrix(lista_sentimiento_predicciones, lista_sentimiento_test)
    
    clasi_matriz_ner = classification_report(lista_bio_test, lista_bio_predicciones)
    clasi_matriz_sentiment = classification_report(lista_sentimiento_predicciones, lista_sentimiento_test)
    
    return {
        'accuracy_ner': accuracy_ner,
        'precision_ner': precision_ner,
        'recall_ner': recall_ner,
        'f1_ner': f1_ner,
        'conf_matrix_ner':str(conf_matrix_ner),
        'clasi_matriz_ner':str(clasi_matriz_ner),
        'accuracy_sentiment': accuracy_sentiment,
        'precision_sentiment': precision_sentiment,
        'recall_sentiment': recall_sentiment,
        'f1_sentiment': f1_sentiment,
        'conf_matrix_sentiment': str(conf_matrix_sentiment),
        'clasi_matriz_sentiment':str(clasi_matriz_sentiment)
    }
# Evaluar el modelo
metrics = evaluate_model(model, test_data)
todo_prediccion_ner = []
todo_prediccion_sentiment = []
def prueba(model, grupo_datos):
    model.eval()
    with torch.no_grad():
        for id, row in grupo_datos.iterrows():
                cadena = row["comentario"]
                embeddings = get_elmo_embeddings(cadena)
                embeddings_tensor = torch.tensor(embeddings).float()
                ner_out, sentiment_out = model(embeddings_tensor)
                ner_preds = torch.argmax(ner_out, dim=-1).cpu().numpy()
                sentiment_preds = torch.argmax(sentiment_out, dim=-1).cpu().numpy()

                #añadirlo a una lista de listas
                todo_prediccion_ner.append(ner_preds)
                todo_prediccion_sentiment.append(sentiment_preds)
                #print("tipo de dato q retorna embeddings elmo: ", type(embeddings))
                #Generar embeddings de elmo
prueba(model, grupo_datos)

lista_pura = []
for sublista in todo_prediccion_ner:
    temp = []
    for elemento in sublista:
        temp.append(int(elemento))
    lista_pura.append(temp)

lista_sen = []
for sublista in todo_prediccion_sentiment:
    temp = []
    for elemento in sublista:
        temp.append(int(elemento))
    lista_sen.append(temp)
print("prediccion_ner: ", lista_pura[0])
#agre
dataframe = pd.DataFrame(
     {'NER': [str(pred) for pred in lista_pura]}
     )
dataframe2 = pd.DataFrame(
    {'Sentiment':[str(elem) for elem in lista_sen]}
    )

dataframe.to_excel('pruebaNer.xlsx', index=False)
dataframe2.to_excel('pruebaSentiment.xlsx', index=False)
# Almacenar los resultados
results.append({
    'dropout_rate': dropout_rate,
    **metrics
})

# Convertir resultados a DataFrame y guardar en Excel
results_df = pd.DataFrame(results)
results_df.to_excel('results4.xlsx', index=False)

print("Resultados guardados en 'results4.xlsx'.")

'''
'''