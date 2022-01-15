# Librería para generar una semilla aleatoria
import random  # creo que no es necesario porque no creo que no lo estoy utilizando
# Librerías de fechas


from datetime import datetime
import datetime as datetime

import numpy as np
# Librerías clásicas dataframes y álgebra
import pandas as pd
# Librería para descargar datos financieros


from pandas_datareader import data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Librerías de modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
# Librerías para dividir el DataFrame entre entrenamiento y test
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PowerTransformer  # necesaria para el tune de nb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Librerias de transformación de variables
random.seed(113)  # lo dicho, creo que no es necesario

# Librerías evaluación
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# Librería para ignorar mensajes de warning
import warnings

warnings.filterwarnings('ignore')

# Librerías de visualización
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import streamlit as st

# The %matplotlib inline is a jupyter notebook specific command that let’s you see the plots
# in the notbook itself.

plt.style.use('ggplot')  # the "ggplot" style, which adjusts the style to
# emulate ggplot (a popular plotting package for R).


st.markdown('### PROYECTO')

st.text_area('', '''

1.1.   Desarrollar un modelo que prediga, con datos a cierre de hoy la dirección del mercado de valores al día siguiente en base a las variaciones porcentuales de los días anteriores (retardos o lags).

1.2.- El sistema analizará el resultado de Accuracy (cross validation) de distintos modelos de clasificación dentro del conjunto de días de retardo (lags) que se le ha facilitado en la entrada de datos.

1.3.- El sistema ajustará los parámetros de ese modelo para el caso concreto del número de retardos (lags) ganador

1.4.- El sistema realizará la predicción de la dirección de mercado para los días de test.

1.5.- Estrategia: 
    En función de la fiabilidad (accuracy) del modelo, el sistema decidirá si toma posiciones L (comprar), S (cortas o a crédito), L/S (cualquiera de las dos direcciones) o no toma posiciones y cierra la ejecución del código.

    Cuándo la predicción de dirección de mercado coincide con la posición que me ha marcado el sistema, entro en mercado. En caso contrario, el sistema no actúa.

    kuuLas entradas son al inicio de la sesión y las salidas al final de la sesión.

1.6.-   No considerar slippage ni comisiones.
     ''', height=700)

st.text("")
st.text("")

# 3.- ENTRADA DE DATOS

st.sidebar.markdown('## Entrada de Datos')

# Defino un número aleatorio para replicar experimentos
seed = 113

# riskfree: tipo de interés de activo sin riesgo (bono a 10 años o ultimamente = 0). Lo necesito para
# calcular ratios como el de Sharpe
risk_free = 0

# Número de días de trading al año
trading_days_ann = 252

# fuente de datos
data_source = "yahoo"

# Cantidad de margen para E-mini SP (se introducirá por defecto)
futures_margin = 15000.00

# Declaración de starting_capital, safetymargin, working_capital_0, asset_type, futures_margin
# symbol, start_date, end_date, lags, lags_entry y choose_close y fiabilidad del sistema


starting_capital = st.sidebar.number_input('Capital Inicial. Ejemplo formato: 25000', value= 25000)
# st.write('El Capital Inicial es: ', starting_capital)

safety_margin = st.sidebar.slider('Margen de Seguridad como nº entre 0.0 y 1.0', 0.0, 1.0, 0.2)
# st.write('El margen de seguridad es: ', safety_margin)

working_capital_0 = starting_capital * (1 - safety_margin)
# st.write('Tu Working Capital es: ', working_capital_0)

asset_type = st.sidebar.radio(
    "Tipo de Activo (por el momento, stocks)",
    ('stocks', 'futures (por el momento, NO)'))

# if asset_type == 'stocks':
#   st.write('Inviertes en stocks.')
# else:
#  st.write("futures más adelante. Elige stocks.")

symbol = st.sidebar.selectbox(
    '¿En qué compañía DJI inviertes?',
    ('AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD',
     'HON', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT',
     'NKE', 'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS', 'DOW'))

# st.write('Has elegido:', symbol)

choose_close = st.sidebar.selectbox(
    'Tipo de cierre preferido (recomendado Close)',
    ('Close', 'Adj Close'))

# st.write('Has elegido:', choose_close)

start_date = st.sidebar.date_input(
    "Primer día de datos",
    datetime.date(2020, 1, 1))
# st.write('El primer día de datos es:', start_date)

end_date = st.sidebar.date_input(
    "Último día de datos",
    datetime.datetime.now())
# st.write('El último día de datos es:', end_date)

system_rel = st.sidebar.slider('Exigencia sistema (0.0 - 1.0: recomendado. 0.51)', 0.0, 1.0, 0.51)
# st.write('La exigencia del sistema es: ', system_rel)

lags = st.sidebar.radio(
    'Periodo de lag days (retardos) a analizar',
    (1, 2, 3, 4, 5))

# st.write('El periodo a analizar es de ', lags, ' días')


st.markdown('### Resumen Entrada Datos')
col1, col2 = st.columns(2)

with col1:
    st.write('El Capital Inicial es: ', starting_capital)
    st.write('El margen de seguridad es: ', safety_margin)
    st.write('Tu Working Capital es: ', working_capital_0)
    st.write('Inviertes en: ', asset_type)
    st.write('Has elegido el valor: ', symbol)

with col2:
    st.write('Tu cierre preferido es: ', choose_close)
    st.write('El primer día de datos es: ', start_date)
    st.write('El último día de datos es: ', end_date)
    st.write('El sistema tiene una exigencia de: ', asset_type)
    st.write('El periodo a analizar es de: ', lags, ' días')

st.text("")
st.text("")

# 4.- DESCARGA DE DATOS


# print('\n\n4.- DESCARGA DE DATOS')
# print('\nDescarga datos desde Yahoo Finance OHLC, Volume y Adj Close\n\n')


# Creo dataframe de los datos de cotizaciones (trading data = td).
# noinspection PyUnboundLocalVariable
td0 = data.DataReader(symbol, data_source, start_date, end_date)
# print(td0)

st.markdown('### Decarga de Datos')

st.dataframe(td0)

st.markdown('###### ¿Quieres ver el gráfico de precio?')
agree = st.checkbox('OK')

if agree:
    st.line_chart(td0['Close'])

st.text("")
st.text("")

st.markdown('### EDA')

# 5.- EDA


# 5.1.- EXPLORACIÓN BÁSICA
print('\n\n5.1.- EXPLORACIÓN BÁSICA')

# Primeros datos
print('\n\nPRIMERAS FILAS')
print(td0.head())

# Últimos datos
print('\nÚLTIMAS FILAS')
print((td0.tail()))

# Dimensión del dataset
print('\nDIMENSIÓN DEL DATAFRAME')
print(td0.shape)

# Nulos y tipos de las variables
print('\nNULOS Y TIPOS')
print(td0.info())

# Dataset secillo y limpio. Sin nulos y todas las variables son del mismo tipo float (numéricas)

# El tipo del índice es datetime
print('\nTIPO DEL ÍNDICE')
print(type(td0.index))

# Compruebo los estadísticos básicos y veo que la std es muy grande debido a la evolución
# de la serie temporal en tendencia. Todo cambiará al trabajar con %
print('\nESTADÍSTICOS BÁSICOS')
print(td0.describe())

# Dibujo el gráfico de la serie Close
print('\nGRÁFICO DE PRECIO')
td0['Close'].plot(figsize=(15, 8), linewidth=0.5)
plt.legend()
plt.show()

# Registros Duplicados: NO HAY
print('\nREGISTROS DUPLICADOS')
print(td0.duplicated().sum())

# Valores repetidos en el Close (no hay por lo que mi clasificación será -1 para posiciones cortas y
# 1 para posiciones largas porque todos los días el mercado sube o baja, nunca repite cierre
# El número coincide con el nº de filas del dataframe por lo que no hay repetición.
print('\nVALORES ÚNICOS')
print(td0.groupby('Date').Close.nunique())

# 5.2.- OUTLIERS
print('\n\n5.2.- OUTLIERS')

# Los outliers son muchos, pero tienen poco sentido porque se trata de una serie que
# se ha acelerado en su tendencia en los últimos tiempos. No puedo elimnarlos porque son fundamentales
# Desaparecerán cuando haga las transformaciones a crecimientos porcentuales
# Estoy calculando sobre el cierre, pero debería calcularlo sobre los %

print('\n\nGRÁFICO BOXPLOT DEL CLOSE')

td0.boxplot(['Close'], figsize=(5, 8))
plt.show()

# 5.3.- ANÁLISIS UNIVARIANTE

print('\n\n5.3.- ANÁLISIS UNIVARIANTE')

# Dibujo el boxplot de todas las variables menos el volumen (tiene otra escala) y veo que el comportamiento es el mismo que en el Close
print('\n\nBOXPLOT DE VARIABLES Y TARGET')
print('NO lo dibuja al colocarle el print delante')
# td0.drop('Volume').boxplot(figsize=(15, 8)


# El volumen también tiene el mismo comportamiento de outliers por la misma razón

print('\nBOXPLOT DEL VOLUMEN')
td0.boxplot(['Volume'], figsize=(5, 8))
plt.show()

# Dibujo las distribuciones de las variables y veo que todas son muy parecidas. Cambia un poco la de Volumen
# Es de esperar que se "normalicen" cuando trabaje con los % de los retardos (lags)
print('\nDISTRIBUCIÓN DE LAS VARIABLES')
td0.hist(figsize=(20, 12), bins=50, xlabelsize=8, ylabelsize=8)
plt.show()

# ; avoid having the matplotlib verbose informations


# 5.4.- CORRELACIÓN ENTRE VARIABLES

print('\n\n5.4.- CORRELACIÓN ENTRE VARIABLES')

print('\n\nMATRIZ DE CORRELACIÓN')

correlation_mat = td0.corr()

sns.heatmap(correlation_mat, annot=True)
plt.show()

# La correlación es absoluta (el OHLC se mueve siempre igual), excepto para el volumen.


print('\nCOMPARACIÓN RELACIÓN VARIABLES')
sns.pairplot(td0, diag_kind='kde', hue='Close')

# Código streamlit

st.markdown('###### Primeras Filas')

st.dataframe(td0.head())

with st.form("my_form"):
    options = st.multiselect(
        '¿Qué elementos quieres revisar?',
        ['Dimensión Dataset', 'Valores Nulos', 'Registros Duplicados', 'Tipo del Índice'])

    submitted = st.form_submit_button("Submit")
    # ver_resultados = st.button("Ver resultados")

    if submitted:

        # if ver_resultados:
        if 'Dimensión Dataset' in options:
            st.write('La dimensión del dataframe es: ', td0.shape)
        if 'Valores Nulos' in options:
            st.write('El número de valores nulos es: ', td0.isnull().sum())
        if 'Registros Duplicados' in options:
            st.write('El nº de registros duplicados es: ', td0.duplicated().sum())
        if 'Tipo del Index' in options:
            st.write('El tipo del Índice es: ', type(td0.index))

# st.write('El tipo del Índice es: ', type(td0.index)) No funciona dentro de st.form , pero sí lo hace fuera


st.markdown('###### Estadísticos Básicos')

st.dataframe(td0.describe())

st.markdown('###### Distribución de las variables')

# fig_barras1 = td0.hist(figsize=(20, 12), bins=50, xlabelsize=8, ylabelsize=8) NO FUNCIONA
# st.pyplot(fig_barras1)  NO FUNCIONA


# fig_barras1 = sns.histplot(td0, bins=50, xlabelsize=8, ylabelsize=8)  NO FUNCIONA
# st.pyplot(fig_barras1)  NO FUNCIONA

# Create histograms


lista_columns_td0 = ['Selecciona en el desplegable', ]
for i in td0.columns:
    lista_columns_td0.append(i)

# st.write(lista_columns_td0)


with st.form("my_form2"):
    options = st.multiselect(
        'Comprueba la distribución de las variables',
        lista_columns_td0)

    submitted = st.form_submit_button("Submit")
    if submitted:
        for i in options:
            fig_hist1 = sns.displot(td0, x=i)
            st.pyplot(fig_hist1)
    else:
        st.write('Realiza tu selección')

st.markdown('###### Correlación entre variables')

correlation_mat = td0.corr()

fig_corr_1, ax = plt.subplots()
sns.heatmap(correlation_mat, annot=True)

st.pyplot(fig_corr_1)

st.markdown('###### Análisis Bivariante')
fig_bivariante1 = sns.pairplot(td0, diag_kind='kde', hue='Close')
st.pyplot(fig_bivariante1)

st.text("")
st.text("")

st.markdown('### INGENIERÍA DE VARIABLES')


# Transformo los datos anteriores en aquellos que necesito para desarrollar mi estrategia
# CREAR UNA FUNCIÓN DE LOS DÍAS DE RETARDO Y SUS RETORNOS

def create_lagged_series(td, symbol, start_date, end_date, lags):
    # Crear DataFrame con días de retardo (lag days). Añadir cierre Today y volumen Today.
    # FINALMENTE NO AÑADO VOLUMEN PORQUE NO ME APORTA NADA

    tdlag = pd.DataFrame(index=td0.index)
    tdlag["Today"] = td0[choose_close]
    # tdlag["Volume"] = td0["Volume"]

    # Incorporar columnas con los días de retardo y sus valores de cierre

    for i in range(0, lags):
        tdlag["Lagprice" + str(i + 1)] = tdlag['Today'].shift(i + 1)
        # tdlag["LagVolume" + str(i+1)] = tdlag['VolToday'].shift(i+1)  # QUITAR SI QUIERO ELIMINAR EL VOLUMEN

    # Crear DataFrame con retornos en % (primero volumen y retorno Today).
    # FINALMENTE NO INCORPORO EL VOLUMEN PORQUE NO CREO QUE APORTE NADA A ESTE MODELO.

    tdret = pd.DataFrame(index=tdlag.index)
    # tdret["Volume"] = tdlag["Volume"]
    # tdret["Today%"] = tdlag["Today"].pct_change()*100.0 Hay que quitarle el 100.
    # Además, pct_change lo deja neto porque le resta 1 al cociente. Es decir, no hay que hacer nada.

    tdret["Today%"] = tdlag["Today"].pct_change()
    # tdret["VolToday%"] = tdlag["VolToday"].pct_change() # QUITAR SI QUIERO ELIMINAR EL VOLUMEN

    # Incorporar columnas con los retornos de los días de retardo en %

    for i in range(0, lags):
        # tdlag["Lagprice" + str(i+1)].pct_change()*100.0 Hay que quitarle el 100. Además, pct_change lo deja neto porque le resta 1 al cociente. Es decir, no hay que hacer nada.
        tdret["Lags%" + str(i + 1)] = tdlag["Lagprice" + str(i + 1)].pct_change()
        # tdret["VolLags%" + str(i+1)] = tdlag["LagVolume" + str(i+1)].pct_change() # QUITAR SI QUIERO ELIMINAR EL VOLUMEN

    # Incorporar columna de dirección del asset hoy (Today): +1 (sube), -1 (baja). La podría incluir más adelante y realizar
    # EDA de ingeniería de variables con este dataframe (también podría incorporar el volumen)

    # tdret["Direction Today%"] = np.sign(tdret["Today%"])

    # Retorna el dataframe de los retornos de los retardos, eliminando los nulos (AL ITNRODUCIR iloc.[lags+1]) ESTOY ELIMINANDO LOS NULOS QUE ME PROVOCAN LOS RETARDOS
    return tdret.iloc[lags + 1:]  # QUITAR # QUITAR SI QUIERO ELIMINAR EL VOLUMEN


# Llamo a la función de creación de dataframe con retorno de retardos... y la guardo el dataframe en una variable
td_returns0 = create_lagged_series(td0, symbol, start_date, end_date, lags)
print(td_returns0.head())

st.markdown('###### Matriz de Retardos')

st.dataframe(td_returns0)

st.markdown('###### EDA Matriz de Retardos (lags)')

# Código streamlit

# st.markdown('###### Primeras Filas')

# st.dataframe(td_returns0.head())


with st.form("my_form3"):
    options = st.multiselect(
        '¿Qué elementos quieres revisar?',
        ['Dimensión Lags', 'Nulos Lags', 'Duplicados Lags', 'Tipo del Index Lags'])

    submitted = st.form_submit_button("Submit")
    # ver_resultados = st.button("Ver resultados")

    if submitted:

        # if ver_resultados:
        if 'Dimensión Lags' in options:
            st.write('La dimensión del dataframe es: ', td_returns0.shape)
        if 'Nulos Lags' in options:
            st.write('El número de valores nulos es: ', td_returns0.isnull().sum())
        if 'Duplicados Lags' in options:
            st.write('El nº de registros duplicados es: ', td_returns0.duplicated().sum())
        if 'Tipo del Index Lags' in options:
            st.write('El tipo del Índice es: ', type(td_returns0.index))

# st.write('El tipo del Índice es: ', type(td_returns0.index)) No funciona dentro de st.form , pero sí lo hace fuera


st.markdown('###### Estadísticos Básicos')

st.dataframe(td_returns0.describe())

st.markdown('###### Gráfico Retorno hoy (variable target)')
chart_data = td_returns0['Today%']

st.line_chart(chart_data)

st.markdown('###### Distribución de las variables')

# fig_barras1 = td0.hist(figsize=(20, 12), bins=50, xlabelsize=8, ylabelsize=8) NO FUNCIONA
# st.pyplot(fig_barras1)  NO FUNCIONA


# fig_barras1 = sns.histplot(td0, bins=50, xlabelsize=8, ylabelsize=8)  NO FUNCIONA
# st.pyplot(fig_barras1)  NO FUNCIONA

# Create histograms


lista_columns_td_returns0 = ['Selecciona en el desplegable', ]
for i in td_returns0.columns:
    lista_columns_td_returns0.append(i)

# st.write(lista_columns_td0)


with st.form("my_form4"):
    options = st.multiselect(
        'Comprueba la distribución de las variables',
        lista_columns_td_returns0)

    submitted = st.form_submit_button("Submit")
    if submitted:
        for i in options:
            fig_hist1 = sns.displot(td_returns0, x=i)
            st.pyplot(fig_hist1)
    else:
        st.write('Realiza tu selección')

st.markdown('###### Correlación entre variables')

correlation_mat_lags = td_returns0.corr()

fig_corr_2, ax = plt.subplots()
sns.heatmap(correlation_mat_lags, annot=True)

st.pyplot(fig_corr_2)

st.markdown('###### Análisis Bivariante')
fig_bivariante2 = sns.pairplot(td_returns0, diag_kind='kde', hue='Today%')
st.pyplot(fig_bivariante2)

st.text("")
st.text("")

st.markdown('### CONSTRUCCIÓN MODELOS')

# 7.- CONSTRUCCIÓN DE MODELOS


# 7.1.- ELECCIÓN DE ALGORITMOS
print('\n\n7.1.- ELECCIÓN ALGORITMOS\n')


# defino the stacking ensemble

# noinspection PyShadowingNames
def get_stacking():
    # define the base models
    level0 = list()
    # level0.append(('lr', lr))
    level0.append(('knn', knn))
    # level0.append(('rf', rf))
    # level0.append(('cart', cart))
    # level0.append(('svm', svm))
    level0.append(('nb', nb))

    # defino meta learner mode
    level1 = LogisticRegression()

    # instancio el stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model


# noinspection PyShadowingNames
def get_models():
    models = dict()
    models['lr'] = lr
    models['knn'] = knn
    models['cart'] = cart
    models['svm'] = svm
    models['nb'] = nb
    models['sc'] = get_stacking()  # no sé si está bien o tengo que poner model
    models['lda'] = lda
    models['rf'] = rf
    # models['qda'] = qda  #LO puedo scar PORQUE ME ESTÁ DANDO PROBLEMAS
    models['mlpc'] = mlpc
    models['ada'] = ada
    models['xgbc'] = xgbc
    models['gbc'] = gbc

    return models


# Instancio los modelos porque los necesitaré más tarde
# Podría suprimir este bloque y escribirlo más adelante en el código

lr = LogisticRegression(random_state=seed)
knn = KNeighborsClassifier()
cart = DecisionTreeClassifier(random_state=seed)
svm = SVC(random_state=seed)
nb = GaussianNB()
sc = get_stacking()
lda = LinearDiscriminantAnalysis()
rf = RandomForestClassifier(random_state=seed)
# qda = QuadraticDiscriminantAnalysis()  #LO SACO PORQUE ME ESTÁ DANDO PROBLEMAS
mlpc = MLPClassifier(max_iter=50, random_state=seed)
ada = AdaBoostClassifier(random_state=seed)
xgbc = XGBClassifier()
gbc = GradientBoostingClassifier(random_state=seed)

# Incluyo la función get_models en la variable models
print('DEFINO FUNCIÓN DE ELECCIÓN DE ALGORITMOS\n')
models = get_models()
print(models)

# 7.2.- DATASETS TRAIN/TEST


# Incorporo la dirección del mercado con el método sign (transforma el incremento, decremento o no moviento en 1, -1, 0)
# Por eso aparece luego el 0 en la confusion matrix

td_returns0["Direction Today%"] = np.sign(td_returns0["Today%"])
print('\nINCORPORO LA DIRECCIÓN DE MERCADO HOY: 1 (SUBE), -1 (BAJA)\n')
print(td_returns0)

# Decido los lag days que utilizaré como predictores (lags_entry: este dato tengo que introducirlo).

lags_list = []

for i in range(0, lags):  # puedo cambiar lags_entry por lags y hacerlo todo desde lags o al revés
    new_element = "Lags%" + str(i + 1)
    lags_list.append(new_element)

print('\n\nLISTA DE LAGS SEGÚN LA ENTRADA DE LAGS QUE HICE\n')
print(lags_list)

# Defino los datasets: identifico columnas de predictores X = td_returns[lags_list], los predictores = retardos,
# identifico columna de variable target y = td_returns["Direction Today%"], la dirección es lo que quiero predecir
# después, divido el dataset original entre train y test (respetando el orden de los días)
# Me reservaré el test para utilizarlo en el backtesting (días más recientes)

# Puedo hacerlo directamente desde la variable de entrada lags y eliminar el lags_entry

X = td_returns0[lags_list]
y = td_returns0["Direction Today%"]

# Supongo que si pongo shuffle como False el random_state no actúa
# Pongo shuffle False para que la matriz de test sean los últimos días
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=0.25,
                                                    random_state=seed, shuffle=False)

print('\n\n7.2.- DATASET TRAIN/TEST')
print('\n\nREALIZO PARTICIONES TRAIN/TEST. Ejemplo X_train\n')
print(X_train)

models = get_models()

# 7.3.- MÉTRICA DE EVALUACIÓN

print('\n\n7.3.- DEFINO MÉTRICA DE EVALUACIÓN: CROSS VALIDATION\n')


# Defino una función para evaluar los modelos utilizando cross-validation
# La evaluación la realizaré sobre el dataset de entrenamiento X_train e y_train

# noinspection PyShadowingNames
def evaluate_model(model, X_train, y_train):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


cv_print = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)

print(cv_print)

# Me queda la duda de qué scoring me está dando ¿el de KFold Train o el KFold validate?
# Supongo que es el de KFold validate y luego lo utilizaré para elegir el algoritmo
# en base a la mejora accuracy.


st.markdown('###### Modelos Utilizados')

st.write(models)

st.markdown('###### Métrica de Evaluación: Cross Validation')

code = '''cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores'''
st.code(code, language='python')

st.markdown('### SELECCIÓN DEL MODELO')

st.text("")
st.text("")

# 8.- SELECCIÓN DEL MODELO


# 8.1.- EVALUACIÓN DE MODELOS

print('\n\n8.1.- EVALUACIÓN DE MODELOS')

print('\n\n8.1.- EVALUACIÓN DE MODELOS\n\n')

# Itero primero por X_train on un sólo lag, después por X_train con 2 lagas y así sucesivamente

#  ¡OJO! Si cambio el orden de las columnas me dará error. Tengo que hacerlo sacando las columnas por nombre
# en lugar de sacarlas por orden


results, names = list(), list()

for i in range(0,
               lags):  # unir lags_entry y lags porque es lo mismo a la hora de optimizar (puedo intercambiarlos en función de necesidades)
    X_train_iter = X_train.iloc[:, :i + 1]  # Este dataframe va incorporando columnas con cada iteración

    for name, model in models.items():
        scores = evaluate_model(model, X_train_iter, y_train)
        name_lag = name + '.' + str(i + 1)
        names.append(name_lag)
        results.append(scores)

        # print('>%s %.3f (%.3f)' % (name_lag, np.mean(scores), np.std(scores)))

# Compruebo el resultado calculando la media de results calculando un array
# utilizaré más abajo este resultado
results_mean = []
for i in results:
    mean_acc = np.mean(i)
    results_mean.append(mean_acc)

# print(np.round(results_mean, 3))


# Compruebo el resultado de las std calculando un array
std = []
for i in results:
    dvstandar = np.std(i)
    std.append(dvstandar)

# print(np.round(std, 3))


# Convierto la accuracy (con cross validation) y desviación típica del train en un dataframe

df0 = pd.DataFrame(
    {'Name': names, 'Accuracy': np.round(results_mean, 10)})  # , 'STD': np.round(std, 3)  (por si quiero

# incorporar la std)

# print(df0)


# Separo el nombre del algoritmo de la iteración (n_lags) gracias al '.'

name = df0["Name"].str.split('.', expand=True)
# print(name)


# Concateno la matriz con la separación de nombre y nº interación columna (n_lags) a la original

name.columns = ['Algo', 'n_lags']
df0 = pd.concat([df0, name], axis=1)
print(df0)

st.markdown('###### Resultados de los distintos modelos')
st.dataframe(df0)

# Extraigo. para el máximo de la accuracy, el nombre del algoritmo, el n_lags y la Accuracy

df0maxim = df0.loc[df0.Accuracy == df0.Accuracy.max(), ['Algo', 'n_lags', 'Accuracy']]
# print(df0maxim)


# Reseteo el índice para poder extraer, luego, el primer máximo (gracias a loc[0]) si hubiese más de uno

df0maxim_reset = df0maxim.reset_index(drop=False)
# print(df0maxim_reset)


# Hago el print con el algoritmo ganador, el n_lags y el accuracy
winning_index1 = df0maxim_reset.loc[0]['index']
winning_model1 = df0maxim_reset.loc[0]['Algo']
winning_lags1 = df0maxim_reset.loc[0]['n_lags']
winning_accuracy1 = df0maxim_reset.loc[0]['Accuracy']

print('\n\nMODELO ELEGIDO')
print(
    f'\nEl modelo elegido es {winning_model1} ({winning_index1}) con nº lags= {winning_lags1} y un accuracy de {np.round(winning_accuracy1, 3)} con crossvalidation')

# 8.2.- GRÁFICO DE MODELOS

print('\n\n8.2.- GRÁFICO DE MODELOS\n\n')

# Dibujo un gráfico boxplot para compararlos

# fig_modelo = pyplot.boxplot(results, labels=names, showmeans=True)  NO FUNCIONA
# pyplot.show()   NO FUNCIONA
# st.pyplot(fig_modelo)   NO FUNCIONA


st.markdown('###### Modelo Elegido')
st.write('El modelo elegido es ', winning_model1, ' con nº de lags= ', winning_lags1, ' y un accuracy de ',
         winning_accuracy1, ' en train con cross validation')

# 8.3.- ACCURACY TRAIN/TEST
print('\n\n8.3.- ACCURACY TRAIN/TEST')

print('Dos particiones distintas Train/Test para dos accuracies distintas:\n')
print('Primera partición (X_train/y-test): ')
print(' 1.1.- Origen: sobre el dataframe original con todos los lags (retardos)')
print(
    ' 1.2.- Objetivo: calcular el modelo y retardo con mejor accuracy (modelo elegido) para, posteriormente, ajustar parámetros')
print(' 1.3.- Accuracy: cross validation\n')
print('Segunda partición: (X_train_1/y_test_1):')
print(
    ' 2.1.- Origen: sobre el dataframe con los retardos elegidos (ganadores con el modelo). Es decir, X_train_val (ver código)')
print(
    ' 2.2.- Objetivo: comparar sobre dataframe retardos elegidos (X_train_val e y_train) la acc. train y test (y_train_q, y_test_1)')
print(' 2.3.- Accuracy: accuracy estándar (75/25)\n\n')

st.markdown('###### Accuracy Train/Test')

st.text_area('Racional Análisis', '''
     Dos particiones distintas Train/Test para dos accuracies distintas:

     Primera partición (X_train/y-test):
        1.1.- Origen: sobre el dataframe original con todos los lags (retardos)
        1.2.- Objetivo: calcular el modelo y retardo con mejor accuracy (modelo elegido) para, posteriormente, ajustar parámetros
        1.3.- Accuracy: cross validation

     Segunda partición: (X_train_1/y_test_1):
        2.1.- Origen: sobre el dataframe con los retardos elegidos (ganadores con el modelo). Es decir, X_train_val (ver código)
        2.2.- Objetivo: comparar sobre dataframe retardos elegidos (X_train_val e y_train) la acc. train y test (y_train_q, y_test_1)
        2.3.- Accuracy: accuracy estándar (75/25)
     ''', height=430)
# st.write('Sentiment:', txt)


# Para el cálculo de accuracy sobre test tengo que rehacer el proceso con el nº lags óptimo que me ha dado el sistema
# Para eso, tengo que transformar el X_train en el X_train_wl que incorpora el n_lags ganador (winning_lags 'wl')
# De esta forma, trabajo ya con el modelo ganador que sólo tengo que optimizarlo con el ajuste de parámetros.
# A este nuevo X_train le llamaré X_train_val

# Transformo la variabe winning_lags1 de str a int porque voy a utilizarla en un rango como número

winning_lags1 = int(winning_lags1)
# print(winning_lags1)

X_train_val = X_train.iloc[:, range(0, winning_lags1)]  # int(winning_lags1)]
# print(X_train_val)


# Divido el dataframe de X_train_val en dos datasets: uno de train (X_train1, y_train1) y otro de validación (X_test1, y_test1)
# para calcular la confusion matrix sobre un set de validación y no sobre el set de test (X_test), que dejo sin tocar hasta
# aplicarle la estrategia. Así, decido si tomo posiciones largas o cortas

# Supongo que si pongo shuffle como False el random_state no actúa
# Pongo shuffle False para que la matriz de validación sean los últimos días
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train_val,
                                                        y_train, test_size=0.25,
                                                        random_state=seed, shuffle=False)

# Calculo, de nuevo, el accuracy sobre la matriz de validación X_test1, y_test1 que actúa como la de Test
# Creo que no lo necesito y dejo en comentario el código


st.markdown('###### Comparativa de Accuracies')

if winning_model1 == 'lr':
    lr.fit(X_train1, y_train1)
    y_test1_predict = lr.predict(X_test1)
    # print(
    #   f'El accuracy_score de lr sobre el set de validación dentro del X_train es {np.round(accuracy_score(y_test1, y_test1_predict), 3)}')
    st.write('El accuracy_score de lr sobre el set de validación dentro del X_train es ',
             np.round(accuracy_score(y_test1, y_test1_predict), 3))
elif winning_model1 == 'knn':
    knn.fit(X_train1, y_train1)
    y_test1_predict = knn.predict(X_test1)
    # print(
    #   f'El accuracy_score de knn sobre el set de validación dentro del X_train es {np.round(accuracy_score(y_test1, y_test1_predict), 3)}')
    st.write('El accuracy_score de knn sobre el set de validación dentro del X_train es ',
             np.round(accuracy_score(y_test1, y_test1_predict), 3))
elif winning_model1 == 'cart':
    cart.fit(X_train1, y_train1)
    y_test1_predict = cart.predict(X_test1)
    # print(
    #   f'El accuracy_score de cart sobre el set de validación dentro del X_train es {np.round(accuracy_score(y_test1, y_test1_predict), 3)}')
    st.write('El accuracy_score de cart sobre el set de validación dentro del X_train es ',
             np.round(accuracy_score(y_test1, y_test1_predict), 3))
elif winning_model1 == 'svm':
    svm.fit(X_train1, y_train1)
    y_test1_predict = svm.predict(X_test1)
    # print(
    #   f'El accuracy_score de svm sobre el set de validación dentro del X_train es {np.round(accuracy_score(y_test1, y_test1_predict), 3)}')
    st.write('El accuracy_score de svm sobre el set de validación dentro del X_train es ',
             np.round(accuracy_score(y_test1, y_test1_predict), 3))
elif winning_model1 == 'nb':
    nb.fit(X_train1, y_train1)
    y_test1_predict = nb.predict(X_test1)
    # print(
    #   f'El accuracy_score de nb sobre el set de validación dentro del X_train es {np.round(accuracy_score(y_test1, y_test1_predict), 3)}')
    st.write('El accuracy_score de nb sobre el set de validación dentro del X_train es ',
             np.round(accuracy_score(y_test1, y_test1_predict), 3))
elif winning_model1 == 'sc':
    sc.fit(X_train1, y_train1)
    y_test1_predict = sc.predict(X_test1)
    # print(
    #   f'El accuracy_score de sc sobre el set de validación dentro del X_train es {np.round(accuracy_score(y_test1, y_test1_predict), 3)}')
    st.write('El accuracy_score de sc sobre el set de validación dentro del X_train es ',
             np.round(accuracy_score(y_test1, y_test1_predict), 3))
elif winning_model1 == 'lda':
    lda.fit(X_train1, y_train1)
    y_test1_predict = lda.predict(X_test1)
    # print(
    #   f'El accuracy_score de lda sobre el set de validación dentro del X_train es {np.round(accuracy_score(y_test1, y_test1_predict), 3)}')
    st.write('El accuracy_score de lda sobre el set de validación dentro del X_train es ',
             np.round(accuracy_score(y_test1, y_test1_predict), 3))
elif winning_model1 == 'rf':
    rf.fit(X_train1, y_train1)
    y_test1_predict = rf.predict(X_test1)
    # print(
    #   f'El accuracy_score de rf sobre el set de validación dentro del X_train es {np.round(accuracy_score(y_test1, y_test1_predict), 3)}')
    st.write('El accuracy_score de rf sobre el set de validación dentro del X_train es ',
             np.round(accuracy_score(y_test1, y_test1_predict), 3))
# elif winning_model1 == 'qda':
# qda.fit(X_train1,y_train1)
# y_test1_predict = qda.predict(X_test1)
# Esta línea no vale, hay que utilizar la de abajo print(f'El accuracy_score de lr sobre el set de validación dentro del X_train es {np.round(accuracy_score(y_test1,y_test1_predict), 3)}')
# st.write('El accuracy_score de mlpc sobre el set de validación dentro del X_train es ',
#   np.round(accuracy_score(y_test1, y_test1_predict), 3))
elif winning_model1 == 'mlpc':
    mlpc.fit(X_train1, y_train1)
    y_test1_predict = mlpc.predict(X_test1)
    # print(
    #   f'El accuracy_score de mlpc sobre el set de validación dentro del X_train es {np.round(accuracy_score(y_test1, y_test1_predict), 3)}')
    st.write('El accuracy_score de mlpc sobre el set de validación dentro del X_train es ',
             np.round(accuracy_score(y_test1, y_test1_predict), 3))
elif winning_model1 == 'ada':
    ada.fit(X_train1, y_train1)
    y_test1_predict = ada.predict(X_test1)
    # print(
    #   f'El accuracy_score de ada sobre el set de validación dentro del X_train es {np.round(accuracy_score(y_test1, y_test1_predict), 3)}')
    st.write('El accuracy_score de ada sobre el set de validación dentro del X_train es ',
             np.round(accuracy_score(y_test1, y_test1_predict), 3))
elif winning_model1 == 'xgbc':
    xgbc.fit(X_train1, y_train1)
    y_test1_predict = xgbc.predict(X_test1)
    # print(
    #   f'El accuracy_score de xgbc sobre el set de validación dentro del X_train es {np.round(accuracy_score(y_test1, y_test1_predict), 3)}')
    st.write('El accuracy_score de xgbc sobre el set de validación dentro del X_train es ',
             np.round(accuracy_score(y_test1, y_test1_predict), 3))
elif winning_model1 == 'gbc':
    gbc.fit(X_train1, y_train1)
    y_test1_predict = gbc.predict(X_test1)
    # print(
    #   f'El accuracy_score de gbc sobre el set de validación dentro del X_train es {np.round(accuracy_score(y_test1, y_test1_predict), 3)}')
    st.write('El accuracy_score de gbc sobre el set de validación dentro del X_train es ',
             np.round(accuracy_score(y_test1, y_test1_predict), 3))
else:
    print('ERROR: no se recoge ningún modelo contemplado')

print('\nACCURACY PRIMERA PARTICIÓN (utilizada para elegir la mejor combinación modelo/lags)')
print(np.round(winning_accuracy1, 3))

print('\nACCURACY SEGUNDA PARTICIÓN (utilizada para comparar acc. train_1/test_1 dentro del antiguo train)')
print(np.round(winning_accuracy1, 3))

st.write('ACCURACY 1ª PARTICIÓN (utilizada para elegir la mejor combinación modelo/lags) ',
         (np.round(winning_accuracy1, 3)))
st.write('ACCURACY 2ª PARTICIÓN (utilizada para comparar acc. train_1/test_1 dentro del antiguo train) ',
         (np.round(winning_accuracy1, 3)))

# 8.4.- CONFUSION MATRIX

print('\n\n8.4.- CONFUSION MATRIX\n')

# noinspection PyUnboundLocalVariable
cf = confusion_matrix(y_test1, y_test1_predict)
# print(pd.DataFrame(cf))

# print(y_test1)

# count = y_test1.value_counts()[0]
# print(count)


st.markdown('###### Confusion Matrix')

# Calculo la matriz de confusión para ver si me intersa más ponerme largo o corto o tomar las dos posiciones.
crosstab = pd.crosstab(y_test1, y_test1_predict, rownames=['True'], colnames=['Predicted'], margins=True)

# Visualize the confusion matrix
fig_cfm1 = plt.figure(figsize=(5, 5))
sns.heatmap(crosstab, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
# all_sample_title = "Confusion Matrix"  #: %0.2f" % (accuracy_score)
# plt.title(all_sample_title, size=15)

st.pyplot(fig_cfm1)

st.text("")
st.text("")

st.markdown('### AJUSTE DE PARÁMETROS')

# 9.- AJUSTE DE PARÁMETROS


st.markdown('###### GRID SEARCH; Best Score, Best Hyperparameters, Best Estimator')

print('\n\n9.- AJUSTE DE PARÁMETROS\n')
print('\nCalcula con Grid Search el Best Score, Best Hyperparameters y Best Estimator\n')

# utilizo X_train, y_train porque es el ajuste final que utilizaré sobre mi X_test, y_test final.


if winning_model1 == 'lr':
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    space = dict()
    space['penalty'] = ['l1', 'l2']
    space['C'] = [0.01, 0.1, 1, 2, 10, 100]
    grid_obj = GridSearchCV(lr, space, scoring='accuracy', n_jobs=-1, cv=cv)
    grid_obj = grid_obj.fit(X_train, y_train)

    lrtune = grid_obj.best_estimator_
    lrtune.fit(X_train, y_train)

    print('Best Score: %s' % grid_obj.best_score_)
    print('Best Hyperparameters: %s' % grid_obj.best_params_)
    print('Best Estimator: %s' % lrtune)

    st.write('El Best Score es: ', grid_obj.best_score_)
    st.write('Los Best Hyperparameters son: ', grid_obj.best_params_)
    st.write('El Best Estimator es: ', lrtune)

elif winning_model1 == 'knn':
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    space = dict()
    space['n_neighbors'] = [3, 5, 7, 9]
    space['weights'] = ['uniform', 'distance']
    space['leaf_size'] = [10, 20, 30, 50]
    grid_obj = GridSearchCV(knn, space, refit=True, scoring='accuracy', verbose=0, cv=cv)
    grid_obj = grid_obj.fit(X_train, y_train)

    knntune = grid_obj.best_estimator_
    knntune.fit(X_train, y_train)

    print('Best Score: %s' % grid_obj.best_score_)
    print('Best Hyperparameters: %s' % grid_obj.best_params_)
    print('Best Estimator: %s' % knntune)

    st.write('El Best Score es: ', grid_obj.best_score_)
    st.write('Los Best Hyperparameters son: ', grid_obj.best_params_)
    st.write('El Best Estimator es: ', knntune)

elif winning_model1 == 'cart':
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    space = dict()
    space['max_leaf_nodes'] = [2, 4, 6, 8, 10]
    space['min_samples_split'] = [2, 3, 4]
    grid_obj = GridSearchCV(cart, space, scoring='accuracy', verbose=0, cv=cv)
    grid_obj = grid_obj.fit(X_train, y_train)

    carttune = grid_obj.best_estimator_
    carttune.fit(X_train, y_train)

    print('Best Score: %s' % grid_obj.best_score_)
    print('Best Hyperparameters: %s' % grid_obj.best_params_)
    print('Best Estimator: %s' % carttune)

    st.write('El Best Score es: ', grid_obj.best_score_)
    st.write('Los Best Hyperparameters son: ', grid_obj.best_params_)
    st.write('El Best Estimator es: ', carttune)

elif winning_model1 == 'svm':
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    space = dict()
    space['C'] = [0.1, 1, 10]
    space['gamma'] = ['scale']
    space['kernel'] = ['rbf', 'sigmoid']
    grid_obj = GridSearchCV(svm, space, refit=True, scoring='accuracy', verbose=0, cv=cv)
    grid_obj = grid_obj.fit(X_train, y_train)

    svmtune = grid_obj.best_estimator_
    svmtune.fit(X_train, y_train)

    print('Best Score: %s' % grid_obj.best_score_)
    print('Best Hyperparameters: %s' % grid_obj.best_params_)
    print('Best Estimator: %s' % svmtune)

    st.write('El Best Score es: ', grid_obj.best_score_)
    st.write('Los Best Hyperparameters son: ', grid_obj.best_params_)
    st.write('El Best Estimator es: ', svmtune)



elif winning_model1 == 'nb':
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    space = {'var_smoothing': np.logspace(0, -9, num=100)}
    ## var_smoothing is a stability calculation to widen (or smooth) the curve and therefore account
    ## for more samples that are further away from the distribution mean. In this case, np.logspace
    ## returns numbers spaced evenly on a log scale, starts from 0, ends at -9, and generates 100 samples.
    grid_obj = GridSearchCV(nb,
                            space,
                            cv=cv,
                            verbose=0,
                            scoring='accuracy')
    Data_transformed = PowerTransformer().fit_transform(X_train)
    grid_obj.fit(Data_transformed, y_train)

    nbtune = grid_obj.best_estimator_

    print(grid_obj.best_params_)
    print(grid_obj.best_score_)
    print('Best Estimator %s' % nbtune)

    st.write('El Best Score es: ', grid_obj.best_score_)
    st.write('Los Best Hyperparameters son: ', grid_obj.best_params_)
    st.write('El Best Estimator es: ', nbtune)



elif winning_model1 == 'sc':
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    space = dict()
    space['knn__n_neighbors'] = [3, 5]
    # space['rf__max_depth'] = [3,5]
    space['final_estimator__C'] = [1]
    grid_obj = GridSearchCV(sc, space, refit=True, scoring='accuracy', verbose=0, cv=cv)
    grid_obj = grid_obj.fit(X_train, y_train)

    sctune = grid_obj.best_estimator_
    sctune.fit(X_train, y_train)

    print('Best Score: %s' % grid_obj.best_score_)
    print('Best Hyperparameters: %s' % grid_obj.best_params_)
    print('Best Estimator: %s' % sctune)

    st.write('El Best Score es: ', grid_obj.best_score_)
    st.write('Los Best Hyperparameters son: ', grid_obj.best_params_)
    st.write('El Best Estimator es: ', sctune)


elif winning_model1 == 'lda':
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    space = dict()
    space['solver'] = ['svd']
    space['tol'] = [0.0001, 0.0002, 0.0003]
    grid_obj = GridSearchCV(lda, space, refit=True, scoring='accuracy', verbose=0, cv=cv)
    grid_obj = grid_obj.fit(X_train, y_train)

    ldatune = grid_obj.best_estimator_
    ldatune.fit(X_train, y_train)

    print('Best Score: %s' % grid_obj.best_score_)
    print('Best Hyperparameters: %s' % grid_obj.best_params_)
    print('Best Estimator: %s' % ldatune)

    st.write('El Best Score es: ', grid_obj.best_score_)
    st.write('Los Best Hyperparameters son: ', grid_obj.best_params_)
    st.write('El Best Estimator es: ', ldatune)

elif winning_model1 == 'rf':
    # define evaluation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    # define search space
    space = dict()
    space['n_estimators'] = [4, 6, 9, 10, 15]
    space['max_depth'] = [2, 3, 5, 10]
    space['min_samples_split'] = [2, 3, 5]
    space['min_samples_leaf'] = [1, 5, 8]

    # Run the grid search
    grid_obj = GridSearchCV(rf, space, scoring='accuracy', cv=cv)
    grid_obj = grid_obj.fit(X_train, y_train)

    # Set the rf to the best combination of parameters
    rftune = grid_obj.best_estimator_

    # Train the model using the training sets
    rftune.fit(X_train, y_train)

    print('Best Score: %s' % grid_obj.best_score_)
    print('Best Hyperparameters: %s' % grid_obj.best_params_)
    print('Best Estimator: %s' % rftune)

    st.write('El Best Score es: ', grid_obj.best_score_)
    st.write('Los Best Hyperparameters son: ', grid_obj.best_params_)
    st.write('El Best Estimator es: ', rftune)




# YA ESTÁ HECHO, PERO, POR EL MOMENTO, DEJAMOS FUERA QDA POR LOS PROBLEMS QUE ME DA

elif winning_model1 == 'qda':
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    space = dict()
    space['reg_param'] = [0.1, 0.2, 0.3, 0.4, 0.5]
    grid_obj = GridSearchCV(qda, space, refit=True, scoring='accuracy', verbose=0, cv=cv)
    grid_obj = grid_obj.fit(X_train, y_train)

    qdatune = grid_obj.best_estimator_
    qdatune.fit(X_train, y_train)

    print('Best Score: %s' % grid_obj.best_score_)
    print('Best Hyperparameters: %s' % grid_obj.best_params_)
    print('Best Estimator: %s' % qdatune)

    st.write('El Best Score es: ', grid_obj.best_score_)
    st.write('Los Best Hyperparameters son: ', grid_obj.best_params_)
    st.write('El Best Estimator es: ', qdatune)



elif winning_model1 == 'mlpc':
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    space = dict()
    space['hidden_layer_sizes'] = [(10, 30, 10), (20,)]
    space['activation'] = ['relu']
    space['solver'] = ['sgd', 'adam']
    space['alpha'] = [0.0001, 0.05]
    space['learning_rate'] = ['constant', 'adaptive']
    grid_obj = GridSearchCV(mlpc, space, refit=True, scoring='accuracy', verbose=0, cv=cv)
    grid_obj = grid_obj.fit(X_train, y_train)

    mlpctune = grid_obj.best_estimator_
    mlpctune.fit(X_train, y_train)

    print('Best Score: %s' % grid_obj.best_score_)
    print('Best Hyperparameters: %s' % grid_obj.best_params_)
    print('Best Estimator: %s' % mlpctune)

    st.write('El Best Score es: ', grid_obj.best_score_)
    st.write('Los Best Hyperparameters son: ', grid_obj.best_params_)
    st.write('El Best Estimator es: ', mlpctune)


elif winning_model1 == 'ada':
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    space = dict()
    space['n_estimators'] = [2, 4, 6, 8]
    space['learning_rate'] = [.001, 0.01, .1]
    grid_obj = GridSearchCV(ada, space, refit=True, scoring='accuracy', verbose=0, cv=cv)
    grid_obj = grid_obj.fit(X_train, y_train)

    adatune = grid_obj.best_estimator_
    adatune.fit(X_train, y_train)

    print('Best Score: %s' % grid_obj.best_score_)
    print('Best Hyperparameters: %s' % grid_obj.best_params_)
    print('Best Estimator: %s' % adatune)

    st.write('El Best Score es: ', grid_obj.best_score_)
    st.write('Los Best Hyperparameters son: ', grid_obj.best_params_)
    st.write('El Best Estimator es: ', adatune)


elif winning_model1 == 'xgbc':
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    space = dict()
    space['n_estimators'] = [50, 75, 100, 300]
    space['learning_rate'] = [0.1, 0.01, 0.05]
    grid_obj = GridSearchCV(xgbc, space, refit=True, scoring='accuracy', verbose=0, cv=cv)
    grid_obj = grid_obj.fit(X_train, y_train)

    xgbctune = grid_obj.best_estimator_
    xgbctune.fit(X_train, y_train)

    print('Best Score: %s' % grid_obj.best_score_)
    print('Best Hyperparameters: %s' % grid_obj.best_params_)
    print('Best Estimator: %s' % xgbctune)

    st.write('El Best Score es: ', grid_obj.best_score_)
    st.write('Los Best Hyperparameters son: ', grid_obj.best_params_)
    st.write('El Best Estimator es: ', xgbctune)


elif winning_model1 == 'gbc':
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    space = dict()
    space['n_estimators'] = [50, 250]
    space['max_depth'] = [3, 4, 5]
    space['learning_rate'] = [0.01, 0.1, 1]
    grid_obj = GridSearchCV(gbc, space, refit=True, scoring='accuracy', verbose=0, cv=cv)
    grid_obj = grid_obj.fit(X_train, y_train)

    gbctune = grid_obj.best_estimator_
    gbctune.fit(X_train, y_train)

    print('Best Score: %s' % grid_obj.best_score_)
    print('Best Hyperparameters: %s' % grid_obj.best_params_)
    print('Best Estimator: %s' % gbctune)

    st.write('El Best Score es: ', grid_obj.best_score_)
    st.write('Los Best Hyperparameters son: ', grid_obj.best_params_)
    st.write('El Best Estimator es: ', gbctune)

st.text("")
st.text("")

st.markdown('### BACKTESTING ESTRATEGIA')

# 10.- BACKTESTING ESTRATEGIA


# 10.1.- CONSTRUCCIÓN DATAFRAME BACKTESTING

print('\n\n10.1.- CONSTRUCCIÓN DATAFRAME BACKTESTING\n')
print('\nTrabajo sobre dos dataframes: el de datos descargados original y el de los retardos')
print("Al original le tengo que incorporar las columnas de Cierre preferido ('choosen_close') y 'Volatility'")
print("Al de los retardos le tengo que incorporar la columna de predicción")
print("Finalmente, los concateno y ya tengo el dataframe con el que voy a trabajar la Estrategia\n\n")

# Trabajo sobre dos dataframes: el de datos descargados original y el de los retardos
# Al original le tengo que incorporar las columnas de Cierre preferido ('choosen_close') y 'Volatility'
# Al de los retardos le tengo que incorporar la columna de predicción


# Dataset sobre el que voy a testear la estrategia (lo definí en la partición y no lo he tocado)
# print(X_test)


# incorporo el cierre preferido sobre el dataframe de datos originales
# td['choosen_close'] = td[elegir_cierre]

td0['choosen_close'] = td0[choose_close]
# print(td0)


# Incorporo la volatilidad sobre datos originales (logaritmo natural del incremento de la variación del precio).
# ESTÁ HECHO CON EL LOGARITMO NATURAL. PENSAR SI AL COCIENTE DE LA VOLATILIDAD DEBERÍAMOS RESTARLE 1¿??


td0['Volatility'] = np.log(td0['choosen_close'] / td0['choosen_close'].shift())
td0['choosen_close_shift'] = td0['choosen_close'].shift()

# print(td0.head())


# Creo variables con el primer y último día hábiles para testear
first_available_testday = X_test.index[0]
last_available_testday = X_test.index[-1]

# print("Primer día en el dataframe de test " + str(first_available_testday))
# print("Último día en el dataframe de test  " + str(last_available_testday))


# Guardo el dataframe de retornos que calculé en su momento (con el subset de días habiles para testear)
# en otra variable (otro nombre para el dataframe)

td_returns = td_returns0[td_returns0.index >= first_available_testday]
# print(td_returns)


# Cálculo la columna de predicción sobre mis datos de test (X_test) según el modelo
# que me ha entregado el mayor nivel de accuracy. Estas predicciones sobre el X_test son las que utilizaré

if winning_model1 == 'lr':
    # noinspection PyUnboundLocalVariable
    lrtune = lrtune.fit(X_train, y_train)
    pred_y_total = lrtune.predict(X_test)
    # print(pred_y_total)
elif winning_model1 == 'knn':
    # noinspection PyUnboundLocalVariable
    knntune = knntune.fit(X_train, y_train)
    pred_y_total = knntune.predict(X_test)
    # print(pred_y_total)
elif winning_model1 == 'cart':
    # noinspection PyUnboundLocalVariable
    carttune = carttune.fit(X_train, y_train)
    pred_y_total = carttune.predict(X_test)
    # print(pred_y_total)
elif winning_model1 == 'svm':
    # noinspection PyUnboundLocalVariable
    svmtune = svmtune.fit(X_train, y_train)
    pred_y_total = svmtune.predict(X_test)
    # print(pred_y_total)
elif winning_model1 == 'nb':
    # noinspection PyUnboundLocalVariable
    nbtune = nbtune.fit(X_train, y_train)
    pred_y_total = nbtune.predict(X_test)
    # print(pred_y_total)
elif winning_model1 == 'sc':
    # noinspection PyUnboundLocalVariable
    sctune = sctune.fit(X_train, y_train)
    pred_y_total = sctune.predict(X_test)
    # print(pred_y_total)
elif winning_model1 == 'lda':
    # noinspection PyUnboundLocalVariable
    ldatune = ldatune.fit(X_train, y_train)
    pred_y_total = ldatune.predict(X_test)
    # print(pred_y_total)
elif winning_model1 == 'rf':
    # noinspection PyUnboundLocalVariable
    rftune = rftune.fit(X_train, y_train)
    pred_y_total = rftune.predict(X_test)
    # print(pred_y_total)

# elif winning_model1 == 'qda':
#  qdatune = qdatune.fit(X_train, y_train)
#  pred_y_total = qdatune.predict(X_test)
#  print(pred_y_total)

elif winning_model1 == 'mlpc':
    # noinspection PyUnboundLocalVariable
    mlpctune = mlpctune.fit(X_train, y_train)
    pred_y_total = mlpctune.predict(X_test)
    # print(pred_y_total)
elif winning_model1 == 'ada':
    # noinspection PyUnboundLocalVariable
    adatune = adatune.fit(X_train, y_train)
    pred_y_total = adatune.predict(X_test)
    # print(pred_y_total)
elif winning_model1 == 'xgbc':
    # noinspection PyUnboundLocalVariable
    xgbctune = xgbctune.fit(X_train, y_train)
    pred_y_total = xgbctune.predict(X_test)
    # print(pred_y_total)
elif winning_model1 == 'gbc':
    # noinspection PyUnboundLocalVariable
    gbctune = gbctune.fit(X_train, y_train)
    pred_y_total = gbctune.predict(X_test)
    # print(pred_y_total)
else:
    print('ERROR: no se recoge ningún modelo contemplado')

# Copio el dataframe de retardos en otro (preparación) para preparar mi dataframe final
# concatenaré después para incorporar OHLC
# Aunque utilizo td_returns en lugar de X_test no hay problema porque lo que realmente me interesa
# es el rango de fechas hábiles para el test que luego se convertirán en hábiles para backtesting y,
# finalmente, cuando itroduzca las fechas de la estrategia, se convertirán en las fechas de la prueba.
# También, lo que me interesa es la columna de pedicción que calculé antes sobre X_test y que es
# independiente de las columnas que veremos en este dataframe.

td_returns_prep = td_returns.copy(deep=True)
# print(td_returns_prep)


# Incorporo al dataframe de retardos (retornos) la columna de predicción

# noinspection PyUnboundLocalVariable
td_returns_prep['Prediction'] = pred_y_total
# print(td_returns_prep)


# Filtro el dataframe de datos originales por el primer día disponible para realizar tests

td = td0[td0.index >= first_available_testday]
# print(td)


# Ya tengo los dos dataframes con la misma dimensión y los puedo concatenar
# Concateno mi dataframe de retardos con predicción con el dataframe de datos originales (OHLC)
# para poder calcular los resutados finales del backtesting
# y lo copio en otro

td_returns_prep_1 = pd.concat([td, td_returns_prep], axis=1)
print(td_returns_prep_1)

# 10.2.- Entrada posición L, S, L/S

print('\n\n10.2.- Entrada posición L, S, L/S\n')

# Coloco la confusion matrix en una columna
tn, fp, fn, tp = confusion_matrix(y_test1, y_test1_predict, labels=[-1, 1]).ravel()
cfh = pd.DataFrame([tn, fp, fn, tp])
# print(cfh)


# Calculo la transpuesta horizontal de la confusión matrix y la convierto en una fila
cfht = cfh.T

# print(cfht)


# Cambio los nombres de las columnas

# cfht = cfht.rename(columns = {'0': 'tn', '1': 'fp', '2': 'fn', '3': 'tp'}, inplace = True)
# print(cfht)

cfht.columns = ['tn', 'fp', 'fn', 'tp']
# print(pd.DataFrame(cfht))


tns = (cfht.iloc[0]['tn'])
tpl = (cfht.iloc[0]['tp'])

print(f'\nNúmero de True Negative (Shorts con predicción OK): {tns}')
print(f'Número de True Positive (Longs con predicción OK): {tpl}')

# Calculo los % de predicción OK de largos/cortos sobre total predicción largos/cortos respectivamente
cfht['total_pred_short'] = cfht['tn'] + cfht['fn']
cfht['total_pred_long'] = cfht['fp'] + cfht['tp']
cfht['pred_short_ok'] = np.round(cfht['tn'] / cfht['total_pred_short'], 3)
cfht['pred_long_ok'] = np.round(cfht['tp'] / cfht['total_pred_long'], 3)
# print(cfht)


# Defino qué dirección de trade utilizaré (Long, Short o Long/Short) dependiendo de
# si el modelo supera el mínimo de fiabilidad exigido (definido al principio al
# entrar la variable system_rel) para operar en cualquiera de las dos direcciones

prob_short = cfht.iloc[0]['pred_short_ok']
prob_long = cfht.iloc[0]['pred_long_ok']

print(f'\nHit Rate Shorts {np.round(prob_short * 100, 2)} %')
print(f'Hit Rate Longs {np.round(prob_long * 100, 2)} % \n')

st.markdown('###### Probabilidades de Predicción')
st.write('la exigencia del sistema es ', system_rel * 100, '%')
st.write('la probabilidad de éxito de las posiciones cortas es ', prob_short * 100, '%')
st.write('la probabilidad de éxito de las posiciones largas es ', prob_long * 100, '%')

st.text("")

recommendation = 'pending'

if prob_short >= system_rel and prob_long >= system_rel:
    recommendation = 'L/S'
elif prob_short >= system_rel:
    recommendation = 'S'
elif prob_long >= system_rel:
    recommendation = 'L'
else:
    recommendation = 'NO OPERAR'
    print('No se cumple el mínimo de fiabilidad exigido. NO OPERAR.')
    print('Se detiene la ejecución y se cierra el programa')
    st.warning('La probabilidad de éxito es menor que la exigencia del sistema. NO operar')
    st.markdown('### NO OPERAR: SE CIERRA EL SISTEMA')
    quit()

print(f'Tomar posiciones: {recommendation}')

st.markdown('###### Dirección del sistema')
st.write('El sistema toma posiciones: ', recommendation)

# Asociar a las posiciones Largas la etiqueta 1 y a las Short la etiqueta -1

st.text("")

direction_trade = recommendation

one_direction_trading = True

if direction_trade == 'L':
    direction_trade = 1
elif direction_trade == 'S':
    direction_trade = -1
elif direction_trade == 'L/S':
    one_direction_trading = False
else:
    print("El sistema NO OPERA por no llegar al mínimo de fiabilidad exigido'")

# Lo introduzco para poder ver más cerca el dataframe en colab
# td_returns_prep_1


# 10.3.- COLUMNA SEÑAL

print('\n\n10.3.- COLUMNA SEÑAL')
print('\nColumna señal que incorporará un 1 cuando hay señal para entrar y un 0 cuando no la hay.\n')

# Creo una columna señal que incorporará un 1 cuando hay señal para entrar y un 0 cuando no la hay.
# La clumna señal es independiente de si la posicion es L o S o si tomaré las dos.

if one_direction_trading:
    td_returns_prep_1['Signal'] = np.where(td_returns_prep_1['Prediction'] == direction_trade, 1.0, 0.0)

else:
    td_returns_prep_1['Signal'] = 1.0

td_returns_signal = td_returns_prep_1
print(td_returns_signal.head(10))

# 10.4.- DÍAS DE BACKTESTING DISPONIBLES

print('\n\n10.4.- DÍAS DE BACKTESTING DISPONIBLES')

# Veo sobre qué días puedo realizar el backtesting

first_available_trading_day = td_returns_signal.index[0]
last_available_trading_day = td_returns_signal.index[-1]

print('\nPrimer día disponible para backtesting ' + str(first_available_trading_day))
print('Último día disponible para backtesting ' + str(last_available_trading_day))

# 10.5.- DÍAS DE BACKTESTING ESTRATEGIA


start_date_strategy = first_available_trading_day
end_date_strategy = last_available_trading_day

st.markdown('###### Días disponibles para Backtesting Estrategia')
st.write('Primer día disponible para backtesting: ', start_date_strategy)
st.write('Último día disponible para backtesting: ', end_date_strategy)

st.text("")

# 10.5.- DATAFRAME CON LOS DÍAS DE BACKTESTING SELECCIONADOS

print('\n\n10.5.- DATAFRAME CON LOS DÍAS DE BACKTESTING SELECCIONADOS\n')

# Defino una máscara para trabajar con el dataframe con las fechas de la estrategia que he definido.
mask = (td_returns_signal.index >= start_date_strategy) & (td_returns_signal.index <= end_date_strategy)
td_returns_signal_strategy = td_returns_signal.loc[mask]
print(td_returns_signal_strategy)

# 10.6.- COLUMNAS DE ENTRADAS, PROFIT, TIPO ACTIVO, Nº ACTIVOS, VOLATILIDAD, %PROFIT DIARIO VS ACUMULADO-1

print('\n\n10.6.- COLUMNAS DE ENTRADAS, PROFIT, TIPO ACTIVO, Nº ACTIVOS, VOLATILIDAD, %PROFIT DIARIO VS ACUMULADO-1\n')

# COLUMNAS DE ENTRADAS


# Incorporo las columnas con los precios de apertura que coinciden con los precios a los que
# entraré en mis posiciones (la estrategia entra en la apertura y sale en el cierre y no considera
# delizamientos ni comisiones). Da lo mismo que sean posiciones largas o cortas.


td_returns_signal_strategy['Entry_Long'] = np.where(
    (td_returns_signal_strategy['Signal'] == 1) & (td_returns_signal_strategy['Prediction'] == 1),
    td_returns_signal_strategy['Open'], 0)

td_returns_signal_strategy['Entry_Short'] = np.where(
    (td_returns_signal_strategy['Signal'] == 1) & (td_returns_signal_strategy['Prediction'] == -1),
    td_returns_signal_strategy['Open'], 0)
# print(td_returns_signal_strategy.head())


# COLUMNA DE PROFIT


# Profit de las posiciones largas y cortas. Hay que incororar las comisiones y el spread si quiero hacerlo más fino
# Tendría que incorporarlo cuando conozca el tamaño de mi posición

td_returns_signal_strategy['Profit_Long'] = np.where(
    (td_returns_signal_strategy['Signal'] == 1) & (td_returns_signal_strategy['Prediction'] == 1),
    td_returns_signal_strategy['choosen_close'] - td_returns_signal_strategy['Open'], 0)

td_returns_signal_strategy['Profit_Short'] = np.where(
    (td_returns_signal_strategy['Signal'] == 1) & (td_returns_signal_strategy['Prediction'] == -1),
    td_returns_signal_strategy['Open'] - td_returns_signal_strategy['choosen_close'], 0)

# Incorporo una columna que contenga los profit de los Longs y de los Shorts

td_returns_signal_strategy['Profit_Unit'] = td_returns_signal_strategy['Profit_Long'] + td_returns_signal_strategy[
    'Profit_Short']

# print(td_returns_signal_strategy.head())


# COLUMNA DE TIPO DE ACTIVO


# Introduzco Asset_Type dependiendo de la naturaleza del activo

td_returns_signal_strategy['Asset_Type'] = asset_type
# print(td_returns_signal_strategy.head())


# COLUMNA NÚMERO DE ACTIVOS


# necesito el close_asset para calcular posteriormente el numero de assets en función del asset que estoy tradeando
# En futuros no tiene nada que ver con el precio pero me permite calcularel número de contratos
# en función del margen y el capital

td_returns_signal_strategy['close_asset'] = np.where((td_returns_signal_strategy['Asset_Type'] == 'stocks'),
                                                     (td_returns_signal_strategy['choosen_close_shift']),
                                                     futures_margin
                                                     )
# print(td_returns_signal_strategy.head())


# Utilizo el acumulado del día anterior CumSum-1 porque lo necesito para calcular el asset_number (nº activos)
# Calculo también el CumSum al final de hoy (es igual a CumSum-1 de mañana) para todos los calculos de rentabilidad

td_returns_signal_strategy['Profit'] = 0
td_returns_signal_strategy['asset_number'] = 0
td_returns_signal_strategy['CumSum-1'] = 0
td_returns_signal_strategy['CumSum'] = 0

# print(td_returns_signal_strategy.head())

# Con el for construyo las columnas que había creado a 0, pero no es eficiente porque
# recorre la tabla entera para cada iteración. Habría que cambiarlo

for i in td_returns_signal_strategy.index:
    td_returns_signal_strategy['CumSum-1'] = np.where(td_returns_signal_strategy.index == start_date_strategy,
                                                      working_capital_0,
                                                      td_returns_signal_strategy['CumSum'].shift())
    td_returns_signal_strategy['asset_number'] = td_returns_signal_strategy['CumSum-1'] / td_returns_signal_strategy[
        'close_asset']
    td_returns_signal_strategy['Profit'] = td_returns_signal_strategy['asset_number'] * td_returns_signal_strategy[
        'Profit_Unit']
    td_returns_signal_strategy['CumSum'] = td_returns_signal_strategy['CumSum-1'] + td_returns_signal_strategy['Profit']

# print(td_returns_signal_strategy.head())

# Le quito los decimales al número de assets (asset_number)

td_returns_signal_strategy['asset_number'] = td_returns_signal_strategy['asset_number'].astype(np.int64)
# print(td_returns_signal_strategy.head())


# COLUMNA DE VOLATILIDAD


# Incorporo la columna de strategy_volatility que es el logaritmo natural de 'CumSum'/'CumSum-1'
# (la variación de mi capital),para calcular posteriormente el ratio de sharpe (lo necesito para el cálculo)

td_returns_signal_strategy['strategy_volatility'] = np.log(
    td_returns_signal_strategy['CumSum'] / td_returns_signal_strategy['CumSum-1'])

# print(td_returns_signal_strategy.head())

# COLUMNA % PROFIT DIARIO MENOS ACUMULADO-1

td_returns_signal_strategy['trade_pc'] = td_returns_signal_strategy['Profit'] / td_returns_signal_strategy['CumSum-1']
# print(td_returns_signal_strategy.head())


print(td_returns_signal_strategy[
          ['Entry_Long', 'Entry_Short', 'Profit', 'Asset_Type', 'asset_number', 'strategy_volatility', 'trade_pc']])

st.markdown('###### DataFrame utilizado para el Backtesting de la Estrategia')

st.text_area('Contenido DataFrame Backtesting', '''
     El DataFrame para Backtesting es el resultado de concatenar los datos originales que me bajé del proveedor de datos con el DataFrame que he creado con los retardos. A estos datos les he ido incorporando columnas necesarias para poder realizar el Backtesting como la volatilidad, la predicción para el día siguiente, la señal de entrada de la estrategia, los beneficios diarios y acumulados, el nº de activos (variable en función del desempeño de la estrategia que puedo comprar, el equity,...)  
     ''', height=220)

st.dataframe(td_returns_signal_strategy)

st.text("")
st.text("")

# 11.- KPIs ESTRATEGIA


# trades_number: número de trades
trades_number = round(td_returns_signal_strategy['Signal'].sum().astype(int), 0)
# print(trades_number)


# duration: Duración de la estrategia: Número de filas
duration = td_returns_signal_strategy.shape[0]
# print(duration)


# exposure_time: tiempo en el mercado
exposure_time = round(trades_number / td_returns_signal_strategy.shape[0], 2) * 100
# print(exposure_time)


# equity_final: capital al final de la estrategia en $
equity_final = round(td_returns_signal_strategy['CumSum'][-1], 2)
# print(equity_final)

# equity_peak: máximo nivel de capital alcanzado
equity_peak = round(td_returns_signal_strategy['CumSum'].max(), 2)
# print(equity_peak)


# profit_total (o net profit): dinero neto ganado

profit_total = round(td_returns_signal_strategy['Profit'].sum(), 2)

# print(profit_total)


# return_total: Retorno total en %
# Calculamos el return_total (en porcentaje) sobre el capital inicial (starting capital)
# y no sobre el working capital que incluye el safety_margin

return_total = round(((profit_total / starting_capital) * 100), 2)
# print(return_total)


# buy_and_hold: rentabilidad de una estrategia buy and hold (comprar y no hacer nada)

# Calculamos primero el punto de entrada: open del primer día de la estrategia

open_first_date = td_returns_signal_strategy['Open'][0]
# print(open_first_date)

# Close End Day (en stocks hay que hacerlo con Adj Close porque es Buy and hold
# y se supone que cobran dividendos,... y en futuros con el close)


close_end_date = 'pending'

if asset_type == 'stocks':
    close_end_date = td_returns_signal_strategy.loc[td_returns_signal_strategy.index[-1], 'Adj Close']

else:
    close_end_date = td_returns_signal_strategy.loc[td_returns_signal_strategy.index[-1], 'Close']

# print(close_end_date)

# Ya podemos calcular el Buy and Hold
buy_and_hold = round(((close_end_date / open_first_date) - 1) * 100, 2)
# print(buy_and_hold)


# profit_ann: profit anualizado
rows_number = len(td_returns_signal_strategy.index)

profit_ann = round((profit_total / rows_number) * trading_days_ann, 2)
# print(profit_ann)


# return_ann: retorno anualizado en %
return_ann = round((return_total / rows_number) * trading_days_ann, 2)
# print(return_ann)


# volatility_ann: volatilidad anualizada en %
volatility_ann = round((td_returns_signal_strategy['strategy_volatility'].std() * np.sqrt(trading_days_ann)) * 100, 2)


# print(volatility_ann)


# sharpe_r: ratio de Sharpe


# noinspection PyShadowingNames
def sharpe_ratio(return_series, N, riskfree):
    mean = return_series.mean() * N - riskfree
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma


sharpe_r = round((sharpe_ratio(td_returns_signal_strategy['strategy_volatility'], trading_days_ann, risk_free)), 2)
# print(sharpe_r)


# max_dd: máximo drawdown

# Primero: calculo la curva de máximos de equity acumulado
td_returns_signal_strategy['max_acum_profit'] = td_returns_signal_strategy['CumSum'].cummax()

# Segundo: incorporo una columna al dataframe con el drawdown acumulado por día
drawdown_1 = (td_returns_signal_strategy['CumSum'] - td_returns_signal_strategy['max_acum_profit']) / \
             td_returns_signal_strategy['max_acum_profit']
td_returns_signal_strategy['drawdown_1'] = drawdown_1
# td_returns_signal_strategy.head()

# Tercero: calculo el mínimo de la columna de drawdown acumulado por día
max_dd = round(drawdown_1.min() * 100, 2)
# print(max_dd)


# calcular el drawdown en dinero para poder calcular el recovery_factor después
td_returns_signal_strategy['drawdown_1$'] = td_returns_signal_strategy['drawdown_1'] * td_returns_signal_strategy[
    'max_acum_profit']
max_dd_dollar = round(td_returns_signal_strategy['drawdown_1$'].min(), 2)
# print(max_dd_dollar)
# print(td_returns_signal_strategy.head())


# recovery_factor: es el retorno total/máximo drawdown (cuantas veces recupero el máximo drawdown)

dd_major = td_returns_signal_strategy['drawdown_1$'].min()

recovery_factor = round(profit_total / (-dd_major), 2)
# print(recovery_factor)


# win_rate: % de días positivos sobre el total de días de trading

# Primero: calculo el número trades (días) positivos
profit_pos = len(td_returns_signal_strategy.loc[td_returns_signal_strategy.Profit > 0])
# print(profit_pos)

# Segundo: calculo el número trades (días) negativos
profit_neg = len(td_returns_signal_strategy.loc[td_returns_signal_strategy.Profit < 0])
# print(profit_neg)

# Win rate = días positivos dividido por el total de días que he hecho trading
win_rate = round((profit_pos / (profit_pos + profit_neg)) * 100, 2)
# print(win_rate)


# profit_factor: profits positivos / profits negativos (cuantas veces son los profits positivos vs los negativos)

# suma del Profit de los trades positivos
subset_pos = td_returns_signal_strategy[td_returns_signal_strategy['Profit'] > 0]
sum_pos = round(subset_pos['Profit'].sum(), 2)
# print(sum_pos)

# Suma del Profit negativo de los trades negativos
subset_neg = td_returns_signal_strategy[td_returns_signal_strategy['Profit'] < 0]
sum_neg = round(subset_neg['Profit'].sum(), 2)
# print(sum_neg)

# Profit factor  Suma Profit positivos / Suma de los Profit negativos
profit_factor = round(-(sum_pos / sum_neg), 2)
# print(profit_factor)


# best_trade_dollar: trade con mejor resultado en dinero
best_trade_dollar = round(td_returns_signal_strategy['Profit'].max(), 2)
# print(best_trade_dollar)


# best_trade_pc = el mejor trade en % sobre el capital acumulado
best_trade_pc = round((td_returns_signal_strategy['trade_pc'].max()) * 100, 2)
# print(best_trade_pc)


# worst_trade_dollar: el peor trade en dinero
worst_trade_dollar = round(td_returns_signal_strategy['Profit'].min(), 2)
# print(worst_trade_dollar)


# worst_trade_pc: el peor trade en % sobre el capital acumulado
worst_trade_pc = round(td_returns_signal_strategy['trade_pc'].min() * 100, 2)
# print(worst_trade_pc)


# expectancy_dollar: esperanza matemática del sistema en dólares
expectancy_dollar = round((profit_pos / (profit_pos + profit_neg)) * (sum_pos / profit_pos) + (
        profit_neg / (profit_pos + profit_neg)) * (sum_neg / profit_neg), 2)
# print(expectancy_dollar)


# expectancy_pc: esperanza matemática de un trade en %

# Calculo la suma del acumulado del capital del día anterior, luego, el total de operaciones y divido para
# obtener la media y utilizarla posteriormente en el cálculo de la esperanza en %

cumsum_sum1 = td_returns_signal_strategy['CumSum-1'].sum()
cumsum_len1 = len(td_returns_signal_strategy['CumSum-1'])
cumsum_mean1 = cumsum_sum1 / cumsum_len1
# noinspection PyStatementEffect
# cumsum_mean1

# Esperanza en % sobre la media del capital acumulado
expectancy_pc = round(((expectancy_dollar / cumsum_mean1) * 100), 2)
# print(expectancy_pc)


# 11.- KPIs EVALUACIÓN DE ESTRATEGIA

# print('\n\n11.- KPIs EVALUACIÓN DE ESTRATEGIA\n')

# print('Starting Capital     '+str(starting_capital))
# print('Strategy Start Date  '+start_date_strategy)
# print('Strategy End Date    '+end_date_strategy)
# print('# Trades             '+str(trades_number))
# print('Duration             '+str(duration))
# print('Exposure Time %      '+str(exposure_time))
# print('Equity Final $       '+str(equity_final))
# print('Equity Peak $        '+str(equity_peak))
# print('Net Profit $         '+str(profit_total))
# print('Return %             '+str(return_total))
# print('Buy & Hold Return %  '+str(buy_and_hold))
# print('Profit (ann.) $      '+str(profit_ann))
# print('Return (Ann.) %      '+str(return_ann))
# print('Volatility (Ann.) %  '+str(volatility_ann))
# print('Sharpe Ratio         '+str(sharpe_r))
# print('Max, Drawdown %      '+str(max_dd))
# print('Max, Drawdown %$     '+str(max_dd_dollar))
# print('Recovery Factor      '+str(recovery_factor))
# print('Win Rate %           '+str(win_rate))
# print('Profit Factor        '+str(profit_factor))
# print('Best Trade Dollar    '+str(best_trade_dollar))
# print('Best Trade %         '+str(best_trade_pc))
# print('Best Trade Dollar    '+str(worst_trade_dollar))
# print('Best Trade %         '+str(worst_trade_pc))
# print('Expectancy dollar    '+str(expectancy_dollar))
# print('Expectancy %         '+str(expectancy_pc))


st.markdown('### EVALUACIÓN ESTRATEGIA')

st.markdown('###### Fechas de la Estrategia')
st.write('La estrategia empieza: ', start_date_strategy)
st.write('La estrategia termina: ', end_date_strategy)

st.info('Los cálculos de rentabilidad se realizan sobre el Capital Inicial (nunca sobre el Working Capital)')

df_resultados = pd.DataFrame()

kpis = ['Capital Inicial', 'Capital Final', 'Nº Trades', 'Duración', 'Tiempo de Exposición %',
        'Working Capital Inicial',
        'Working Capital Final', 'Working Capital Máximo', 'Resultado $', 'Resultado %', 'Buy & Hold %',
        'Resultado Anualizado $',
        'Resultado Anualizado %', 'Volatilidad Anualizada %', 'Ratio de Sharpe', 'Drawdown Máximo %',
        'Drawdown Máximo $', 'Recovery Factor', 'Win Rate %', 'Profit Factor', 'Mejor Trade $', 'Mejor Trade %',
        'Peor Trade $', 'Peor Trade %', 'Expectancy $', 'Expectancy %']

kpis_variable = [starting_capital, starting_capital + profit_total, trades_number, duration, exposure_time,
                 working_capital_0,
                 equity_final, equity_peak, profit_total, return_total, buy_and_hold, profit_ann, return_ann,
                 volatility_ann, sharpe_r, max_dd, max_dd_dollar, recovery_factor, win_rate, profit_factor,
                 best_trade_dollar, best_trade_pc, worst_trade_dollar, worst_trade_pc, expectancy_dollar,
                 expectancy_pc]

df_resultados['KPIs'] = kpis
df_resultados['Resultados'] = kpis_variable

df_resultados.index = df_resultados.index + 1

st.table(df_resultados)

st.text("")
st.text("")

st.markdown('### GRÁFICOS ESTRATEGIA')

# 12.- GRÁFICOS ESTRATEGIA


print('\n\n12.- GRÁFICOS ESTRATEGIA')

print('\n\n12.1.- GRÁFICO DE ENTRADAS\n')

# Gráfico de entradas en las posiciones

# 1. We will start by importing this library:
import matplotlib.pyplot as plt

# 2. Next, we will define a figure that will contain our chart. El argumento 111 ó 1,1,1 dice 1 fila, 1 columna y
# posición en ese grid (es decir, un único cuadro).

st.markdown('###### Entradas')

fig_en = plt.figure(figsize=(15, 7))
ax1 = fig_en.add_subplot(111, ylabel='Asset price in $')

# 3. Now, we will plot the price within the range of days we initially chose:
td_returns_signal_strategy['choosen_close'].plot(ax=ax1, color='r', lw=2.)

# 4. Next, we will draw a down arrow when we sell one Google share:
if direction_trade == -1:

    ax1.plot(td_returns_signal_strategy.loc[td_returns_signal_strategy.Entry_Short > 0].index,
             td_returns_signal_strategy.choosen_close[td_returns_signal_strategy.Prediction == -1.0], 'v', markersize=5,
             color='m')

    # plt.show()
    st.pyplot(fig_en)

# 5. Next, we will draw an up arrow when we buy one Google share:
elif direction_trade == 1:

    ax1.plot(td_returns_signal_strategy.loc[td_returns_signal_strategy.Entry_Long > 0].index,
             td_returns_signal_strategy.choosen_close[td_returns_signal_strategy.Prediction == 1.0], '^', markersize=5,
             color='k')

    # plt.show()
    st.pyplot(fig_en)

elif recommendation == 'L/S':
    ax1.plot(td_returns_signal_strategy.loc[td_returns_signal_strategy.Entry_Short > 0].index,
             td_returns_signal_strategy.choosen_close[td_returns_signal_strategy.Prediction == -1.0], 'v', markersize=5,
             color='m')
    ax1.plot(td_returns_signal_strategy.loc[td_returns_signal_strategy.Entry_Long > 0].index,
             td_returns_signal_strategy.choosen_close[td_returns_signal_strategy.Prediction == 1], '^', markersize=5,
             color='k')

    # plt.show()
    st.pyplot(fig_en)

else:
    fig_en = plt.figure(figsize=(15, 7))
    ax1 = fig_en.add_subplot(111, ylabel='Asset price in $')

# 12.2.- GRÁFICO DE RESULTADOS DE LOS TRADES
print('\n\n12.2.- GRÁFICO DE RESULTADOS DE LOS TRADES\n')

# Gráfico de resultados de los trades en $

st.markdown('###### Resultados de Trades individuales')

fig_trades = plt.figure(figsize=(15, 7))

ax1 = fig_trades.add_subplot(ylabel='Trade performance in $')
ax1.plot(td_returns_signal_strategy['Profit'])

# plt.show()
st.pyplot(fig_trades)

# 12.3.- GRÁFICO DE LA CURVA DE EQUITY
print('\n\n12.3.- GRÁFICO DE LA CURVA DE EQUITY\n')

# Gráfico de la curva de Equity (capital acumulado durante la estrategia)

st.markdown('###### Curva de Equity')

fig_eq = plt.figure(figsize=(15, 7))

ax1 = fig_eq.add_subplot(xlabel='Dates', ylabel='Accumulated Equity')
ax1.plot(td_returns_signal_strategy['CumSum'])

# plt.show()
st.pyplot(fig_eq)

# 12.4.- GRÁFICO DE MÁXIMOS ACUMULADOS O RECIENTES
print('\n\n12.4.- GRÁFICO DE MÁXIMOS ACUMULADOS O RECIENTES\n')

# Gráfico de máximos de la curva de equity acumulada  LLEVAR A GRÁFICOS

st.markdown('###### Máximos Acumulados')

fig_ma = plt.figure(figsize=(15, 7))

ax1 = fig_ma.add_subplot(xlabel='Dates', ylabel='Max Accumulated Equity')
ax1.plot(td_returns_signal_strategy['max_acum_profit'])

# plt.show()
st.pyplot(fig_ma)

# 12.5.- GRÁFICO DE DRAWDOWNS
print('\n\n12.5.- GRÁFICO DE DRAWDOWNS\n')

# Gráfico de drawdowns   LLEVAR A GRÁFICOS

st.markdown('###### Drawdowns')

fig_dd = plt.figure(figsize=(15, 5))

ax1 = fig_dd.add_subplot(xlabel='Dates', ylabel='Drawdown %')
ax1.plot(td_returns_signal_strategy['drawdown_1'])

# plt.show()
st.pyplot(fig_dd)

st.text("")
st.text("")

st.markdown('### PREDICCIÓN DIRECCIÓN MERCADO DÍA SIGUIENTE')

futuro = td_returns_signal_strategy.loc[td_returns_signal_strategy.index[-1], 'Prediction']

st.write('La predicción para la dirección del mercado de mañana es (1 = Sube, -1 = Baja): ', futuro)

futuro_en = td_returns_signal_strategy.loc[td_returns_signal_strategy.index[-1], 'Signal']
st.write('¿Entraré mañana en el mercado? (1 = SÍ, 0 = NO): ', futuro)

st.markdown("***")  # Introduce una línea de separación
st.text("")  # Introduce un espacio en blanco




