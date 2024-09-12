#!/usr/bin/env python
# coding: utf-8

# # ¡Hola, Denisse!  
# 
# Mi nombre es Carlos Ortiz, soy code reviewer de TripleTen y voy a revisar el proyecto que acabas de desarrollar.
# 
# Cuando vea un error la primera vez, lo señalaré. Deberás encontrarlo y arreglarlo. La intención es que te prepares para un espacio real de trabajo. En un trabajo, el líder de tu equipo hará lo mismo. Si no puedes solucionar el error, te daré más información en la próxima ocasión. 
# 
# Encontrarás mis comentarios más abajo - **por favor, no los muevas, no los modifiques ni los borres**.
# 
# ¿Cómo lo voy a hacer? Voy a leer detenidamente cada una de las implementaciones que has llevado a cabo para cumplir con lo solicitado. Verás los comentarios de esta forma:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Si todo está perfecto.
# </div>
# 
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta.
# </div>
# 
# 
# <div class="alert alert-block alert-danger">
#     
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
#     
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# 
# Puedes responderme de esta forma: 
# 
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# </div>
# ¡Empecemos!

# # Descripción

# La compañía de seguros Sure Tomorrow quiere resolver varias tareas con la ayuda de machine learning y te pide que evalúes esa posibilidad.
# - Tarea 1: encontrar clientes que sean similares a un cliente determinado. Esto ayudará a los agentes de la compañía con el marketing.
# - Tarea 2: predecir la probabilidad de que un nuevo cliente reciba una prestación del seguro. ¿Puede un modelo de predictivo funcionar mejor que un modelo dummy?
# - Tarea 3: predecir el número de prestaciones de seguro que un nuevo cliente pueda recibir utilizando un modelo de regresión lineal.
# - Tarea 4: proteger los datos personales de los clientes sin afectar al modelo del ejercicio anterior. Es necesario desarrollar un algoritmo de transformación de datos que dificulte la recuperación de la información personal si los datos caen en manos equivocadas. Esto se denomina enmascaramiento u ofuscación de datos. Pero los datos deben protegerse de tal manera que no se vea afectada la calidad de los modelos de machine learning. No es necesario elegir el mejor modelo, basta con demostrar que el algoritmo funciona correctamente.
# 

# # Preprocesamiento y exploración de datos
# 
# ## Inicialización

# In[50]:


pip install scikit-learn --upgrade


# In[51]:


import numpy as np
import pandas as pd

import seaborn as sns

import math

import sklearn.linear_model
import sklearn.metrics
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from IPython.display import display


# ## Carga de datos

# Carga los datos y haz una revisión básica para comprobar que no hay problemas obvios.

# In[52]:


df = pd.read_csv('/datasets/insurance_us.csv')


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Buen trabajo con la importación de datos y de librerías.
# </div>

# Renombramos las columnas para que el código se vea más coherente con su estilo.

# In[53]:


df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})


# In[54]:


df.sample(10)


# In[55]:


df.info()


# Cambiaremos el tipo de datos de: age, por int

# In[56]:


df['age']=df['age'].astype('int')


# In[57]:


df.info()


# Echaremos un vistazo a las estadísticas descriptivas de los datos.

# In[58]:


df.describe()


# ## Análisis exploratorio de datos

# Vamos a comprobar rápidamente si existen determinados grupos de clientes observando el gráfico de pares.

# In[59]:


g = sns.pairplot(df, kind='hist')
g.fig.set_size_inches(12, 12)


# De acuerdo, es un poco complicado detectar grupos obvios (clústeres) ya que es difícil combinar diversas variables simultáneamente (para analizar distribuciones multivariadas). Ahí es donde LA y ML pueden ser bastante útiles.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Buen trabajo con esta exploración inicial.
# </div>

# # Tarea 1. Clientes similares

# En el lenguaje de ML, es necesario desarrollar un procedimiento que devuelva los k vecinos más cercanos (objetos) para un objeto dado basándose en la distancia entre los objetos.
# Es posible que quieras revisar las siguientes lecciones (capítulo -> lección)- Distancia entre vectores -> Distancia euclidiana
# - Distancia entre vectores -> Distancia Manhattan
# 
# Para resolver la tarea, podemos probar diferentes métricas de distancia.

# Escribe una función que devuelva los k vecinos más cercanos para un $n^{th}$ objeto basándose en una métrica de distancia especificada. A la hora de realizar esta tarea no debe tenerse en cuenta el número de prestaciones de seguro recibidas.
# Puedes utilizar una implementación ya existente del algoritmo kNN de scikit-learn (consulta [el enlace](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors)) o tu propia implementación.
# Pruébalo para cuatro combinaciones de dos casos- Escalado
#   - los datos no están escalados
#   - los datos se escalan con el escalador [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)
# - Métricas de distancia
#   - Euclidiana
#   - Manhattan
# 
# Responde a estas preguntas:- ¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?- ¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)?

# In[60]:


feature_names = ['gender', 'age', 'income', 'family_members']


# In[61]:


def get_knn(df, n, k, metric):


    #param df: DataFrame de pandas utilizado para encontrar objetos similares dentro del mismo lugar
    #param n: número de objetos para los que se buscan los vecinos más cercanos
    #param k: número de vecinos más cercanos a devolver
    #param métrica: nombre de la métrica de distancia

    nbrs = NearestNeighbors(n_neighbors=k, metric=metric)
    nbrs.fit(df[feature_names])
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[n][feature_names]], k, return_distance=True)
    
    df_res = pd.concat([
        df.iloc[nbrs_indices[0]], 
        pd.DataFrame(nbrs_distances.T, index=nbrs_indices[0], columns=['distance'])
        ], axis=1)
    
    return df_res


# Escalar datos.

# In[62]:


feature_names = ['gender', 'age', 'income', 'family_members']

transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[feature_names].to_numpy())

df_scaled = df.copy()
df_scaled.loc[:, feature_names] = transformer_mas.transform(df[feature_names].to_numpy())


# In[63]:


df_scaled.sample(5)


# Ahora, vamos a obtener registros similares para uno determinado, para cada combinación

# In[64]:


get_knn(df_scaled, 1, 10, 'euclidean')


# In[65]:


get_knn(df_scaled, 1, 10, 'manhattan')


# In[66]:


get_knn(df, 1, 10, 'euclidean')


# In[67]:


get_knn(df, 1, 10, 'manhattan')


# Respuestas a las preguntas

# **¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?** 
# 
# Si, al estar escalados, es más preciso, y lo vemos al comparar los datos de los clientes, mientras que los datos escalados son similares en todo, los no escalados, solo son iguales en el salario. Pero sobre todo lo notamos en insurance_benefits, pues es el que nos interesa, que los demás datos sean similares, para poder obtener un cliente similar 

# **¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)?** 
# 
# Con los no datos escalados, en lo único que son iguales los datos es en el salario, pero el resto de los datos en realidad es bastante diferente, y de hecho "insurance_benefits" que es lo que nos interesa, todos son completamente diferentes
# Con los datos escalados, se parecen un poco más, aunque si comparamos las distancias con euclidean, vemos que son más cercanos los datos de euclidean 

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Concuerdo con las conclusiones que planteas. KNN es un algoritmo muy sensible a las unidades de medida de las variables y por eso siempre hay que realizar un escalamiento.
# </div>

# # Tarea 2. ¿Es probable que el cliente reciba una prestación del seguro?

# En términos de machine learning podemos considerarlo como una tarea de clasificación binaria.

# Con el valor de `insurance_benefits` superior a cero como objetivo, evalúa si el enfoque de clasificación kNN puede funcionar mejor que el modelo dummy.
# Instrucciones:
# - Construye un clasificador basado en KNN y mide su calidad con la métrica F1 para k=1...10 tanto para los datos originales como para los escalados. Sería interesante observar cómo k puede influir en la métrica de evaluación y si el escalado de los datos provoca alguna diferencia. Puedes utilizar una implementación ya existente del algoritmo de clasificación kNN de scikit-learn (consulta [el enlace](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)) o tu propia implementación.- Construye un modelo dummy que, en este caso, es simplemente un modelo aleatorio. Debería devolver "1" con cierta probabilidad. Probemos el modelo con cuatro valores de probabilidad: 0, la probabilidad de pagar cualquier prestación del seguro, 0.5, 1.
# La probabilidad de pagar cualquier prestación del seguro puede definirse como
# $$
# P\{\text{prestación de seguro recibida}\}=\frac{\text{número de clientes que han recibido alguna prestación de seguro}}{\text{número total de clientes}}.
# $$
# 
# Divide todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30.

# In[68]:


# сalcula el objetivo
df['insurance_benefits_received'] = df['insurance_benefits']>0


# In[69]:


df


# In[70]:


# comprueba el desequilibrio de clases con value_counts()
df.value_counts('insurance_benefits_received')


# In[71]:


features= df.drop(['insurance_benefits_received', 'insurance_benefits'], axis=1).to_numpy()
target= df['insurance_benefits_received'].to_numpy()


# In[72]:


features_train, features_valid, target_train, target_valid= train_test_split(features, target, test_size=.3, random_state=12345)


# In[73]:


def eval_classifier(y_true, y_pred):
    
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    print(f'F1: {f1_score:.2f}')
    
# si tienes algún problema con la siguiente línea, reinicia el kernel y ejecuta el cuaderno de nuevo    
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='all')
    print('Matriz de confusión')
    print(cm)


# In[74]:


# generar la salida de un modelo aleatorio

def rnd_model_predict(P, size, seed=42):

    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)


# In[75]:


for P in [0, df['insurance_benefits_received'].sum() / len(df), 0.5, 1]:

    print(f'La probabilidad: {P:.2f}')
    y_pred_rnd = rnd_model_predict(P, len(target), seed=42)
        
    eval_classifier(df['insurance_benefits_received'], y_pred_rnd)
    
    print()


# Creando el clasificador basado en KNN:

# In[76]:


for i in range(1,11):
    knn_classifier = KNeighborsClassifier(n_neighbors=i)
    
    knn_classifier.fit(features_train, target_train)
    y_pred_knn = knn_classifier.predict(features_valid)
    print('n_neighbors:', i)
    print(eval_classifier(target_valid, y_pred_knn))


# Lo mismo pero con los datos escalados:

# In[77]:


features_scaled= df_scaled.drop('insurance_benefits', axis=1)


# In[78]:


features_scaled


# In[79]:


features_train_sc, features_valid_sc, target_train, target_valid= train_test_split(features_scaled, target, test_size=.3, random_state=12345)


# In[80]:


for i in range(1,11):
    knn_classifier = KNeighborsClassifier(n_neighbors=i)
    knn_classifier.fit(features_train_sc, target_train)
    y_pred_knn = knn_classifier.predict(features_valid_sc)
    print('n_neighbors:', i)
    print(eval_classifier(target_valid, y_pred_knn))


# Vemos que los datos escalados hacen que el modelo sea mucho más exacto, y con n_neighbors=1 alcanza casi la perfección, pues marca un .97 de f1_score

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# ¡Perfecto!
# </div>

# # Tarea 3. Regresión (con regresión lineal)

# Con `insurance_benefits` como objetivo, evalúa cuál sería la RECM de un modelo de regresión lineal.

# Construye tu propia implementación de regresión lineal. Para ello, recuerda cómo está formulada la solución de la tarea de regresión lineal en términos de LA. Comprueba la RECM tanto para los datos originales como para los escalados. ¿Puedes ver alguna diferencia en la RECM con respecto a estos dos casos?
# 
# Denotemos- $X$: matriz de características; cada fila es un caso, cada columna es una característica, la primera columna está formada por unidades- $y$ — objetivo (un vector)- $\hat{y}$ — objetivo estimado (un vector)- $w$ — vector de pesos
# La tarea de regresión lineal en el lenguaje de las matrices puede formularse así:
# $$
# y = Xw
# $$
# 
# El objetivo de entrenamiento es entonces encontrar esa $w$ w que minimice la distancia L2 (ECM) entre $Xw$ y $y$:
# 
# $$
# \min_w d_2(Xw, y) \quad \text{or} \quad \min_w \text{MSE}(Xw, y)
# $$
# 
# Parece que hay una solución analítica para lo anteriormente expuesto:
# $$
# w = (X^T X)^{-1} X^T y
# $$
# 
# La fórmula anterior puede servir para encontrar los pesos $w$ y estos últimos pueden utilizarse para calcular los valores predichos
# $$
# \hat{y} = X_{val}w
# $$

# Divide todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30. Utiliza la métrica RECM para evaluar el modelo.

# In[81]:


class MyLinearRegression:
    
    def __init__(self):
        
        self.weights = None
    
    def fit(self, X, y):
        
        # añadir las unidades
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        self.weights = np.linalg.inv(X2.T.dot(X2))@ X2.T @ y

    def predict(self, X):
        
        # añadir las unidades
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        y_pred = X2.dot(self.weights)
        
        return y_pred


# In[82]:


def eval_regressor(y_true, y_pred):
    
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse:.2f}')
    
    r2_score = sklearn.metrics.r2_score(y_true, y_pred)
    print(f'R2: {r2_score:.2f}')    


# In[83]:


X = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train, y_train)
print(lr.weights)

y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)


# Para los datos escalados:

# In[84]:


X_sc = df_scaled[['age', 'gender', 'income', 'family_members']].to_numpy()
y_sc = df_scaled['insurance_benefits'].to_numpy()

X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(X_sc, y_sc, test_size=0.3, random_state=12345)

lr_sc = MyLinearRegression()

lr_sc.fit(X_train_sc, y_train_sc)
print(lr_sc.weights)

y_test_pred_sc = lr_sc.predict(X_test_sc)
eval_regressor(y_test_sc, y_test_pred_sc)


# Vemos que el r2_score del modelo es igual tanto para los datos normales como para los datos escalados

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Todo correcto en esta sección. Tu código demuestra que no importa si están o no escalados, las métricas son las mismas.
# </div>

# # Tarea 4. Ofuscar datos

# Lo mejor es ofuscar los datos multiplicando las características numéricas (recuerda que se pueden ver como la matriz $X$) por una matriz invertible $P$. 
# 
# $$
# X' = X \times P
# $$
# 
# Trata de hacerlo y comprueba cómo quedarán los valores de las características después de la transformación. Por cierto, la propiedad de invertibilidad es importante aquí, así que asegúrate de que $P$ sea realmente invertible.
# 
# Puedes revisar la lección 'Matrices y operaciones matriciales -> Multiplicación de matrices' para recordar la regla de multiplicación de matrices y su implementación con NumPy.

# In[85]:


personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]


# In[86]:


X = df_pn.to_numpy()


# Generar una matriz aleatoria $P$.

# In[87]:


rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))


# Comprobar que la matriz P sea invertible

# In[88]:


np.linalg.inv(P)


# Asi que ofuscaremos los datos multiplicando esta matriz por nuestros datos

# In[89]:


ofs= X @ P


# ¿Puedes adivinar la edad o los ingresos de los clientes después de la transformación?

# No

# In[90]:


ofs


# ¿Puedes recuperar los datos originales de $X'$ si conoces $P$? Intenta comprobarlo a través de los cálculos moviendo $P$ del lado derecho de la fórmula anterior al izquierdo. En este caso las reglas de la multiplicación matricial son realmente útiles

# In[91]:


P_inv= np.linalg.inv(P)
X_recovered= ofs @ P_inv
X_recovered


# Muestra los tres casos para algunos clientes- Datos originales
# - El que está transformado- El que está invertido (recuperado)

# Datos_originales:

# In[92]:


X[:3]


# Datos ofuscados (transformados):

# In[93]:


ofs[:3]


# Datos recuperados:

# In[94]:


X_recovered[:3]


# Seguramente puedes ver que algunos valores no son exactamente iguales a los de los datos originales. ¿Cuál podría ser la razón de ello?

# Podría ser por la precisión numérica y el redondeo al hacer los cálculos, pues transformamos los datos en repetidas ocsiones, lo que puede hcer que presenten ligeros cambios 

# ## Prueba de que la ofuscación de datos puede funcionar con regresión lineal

# En este proyecto la tarea de regresión se ha resuelto con la regresión lineal. Tu siguiente tarea es demostrar _analytically_ que el método de ofuscación no afectará a la regresión lineal en términos de valores predichos, es decir, que sus valores seguirán siendo los mismos. ¿Lo puedes creer? Pues no hace falta que lo creas, ¡tienes que que demostrarlo!

# Entonces, los datos están ofuscados y ahora tenemos $X \times P$ en lugar de tener solo $X$. En consecuencia, hay otros pesos $w_P$ como
# $$
# w = (X^T X)^{-1} X^T y \quad \Rightarrow \quad w_P = [(XP)^T XP]^{-1} (XP)^T y
# $$
# 
# ¿Cómo se relacionarían $w$ y $w_P$ si simplificáramos la fórmula de $w_P$ anterior? 
# 
# ¿Cuáles serían los valores predichos con $w_P$? 
# 
# ¿Qué significa esto para la calidad de la regresión lineal si esta se mide mediante la RECM?
# Revisa el Apéndice B Propiedades de las matrices al final del cuaderno. ¡Allí encontrarás fórmulas muy útiles!
# 
# No es necesario escribir código en esta sección, basta con una explicación analítica.

# **Respuesta**

# Utilizando la Reversibilidad de la transposición de un producto de matrices: $(AB)^T = B^TA^T$

# $w_P = [(XP)^T XP]^{-1} (XP)^T y \quad \Rightarrow \quad w_p=[P^T X^T X P]^{-1} P^T X^T y$
# 
# 
# Agrupamos las matrices del final:
# $w_p=[P^T X^T X P]^{-1} P^T X^T y \quad \Rightarrow \quad w_p=[P^T (X^T X P)]^{-1} P^T X^T y$
# 
# 
# Aplicamos las propiedades de la inversa:
# $w_p=[P^T (X^T X P)]^{-1} P^T X^T y \quad \Rightarrow \quad w_p= (X^T X P)^{-1} (P^T)^{-1} P^T X^T y$
# 
# Y con esto obtenemos una matriz identidad:
# $w_p= (X^T X P)^{-1} (P^T)^{-1} P^T X^T y \quad \Rightarrow \quad w_p= (X^T X P)^{-1} X^T y$
# 
# Si volvemos a aplicar las propiedades de la inversa, llegamos a que:
# $w_p=P^{-1}(X^TX)^{-1}X^Ty$

# Los valores predichos utilizando $w_p$ , serán los mismos que los valores predichos utilizando $w$ , ya que solo se están transformando junto con los datos ofuscados, con la matriz inversa
# Y esto significa que la cálidad será la misma para ambos pesos, el RECM no cambia 

# <table>
# <tr>
# <td>Distributividad</td><td>$A(B+C)=AB+AC$</td>
# </tr>
# <tr>
# <td>No conmutatividad</td><td>$AB \neq BA$</td>
# </tr>
# <tr>
# <td>Propiedad asociativa de la multiplicación</td><td>$(AB)C = A(BC)$</td>
# </tr>
# <tr>
# <td>Propiedad de identidad multiplicativa</td><td>$IA = AI = A$</td>
# </tr>
# <tr>
# <td></td><td>$A^{-1}A = AA^{-1} = I$
# </td>
# </tr>    
# <tr>
# <td></td><td>$(AB)^{-1} = B^{-1}A^{-1}$</td>
# </tr>    
# <tr>
# <td>Reversibilidad de la transposición de un producto de matrices,</td><td>$(AB)^T = B^TA^T$</td>
# </tr>    
# </table>

# ## Prueba de regresión lineal con ofuscación de datos

# Ahora, probemos que la regresión lineal pueda funcionar, en términos computacionales, con la transformación de ofuscación elegida.
# Construye un procedimiento o una clase que ejecute la regresión lineal opcionalmente con la ofuscación. Puedes usar una implementación de regresión lineal de scikit-learn o tu propia implementación.
# Ejecuta la regresión lineal para los datos originales y los ofuscados, compara los valores predichos y los valores de las métricas RMSE y $R^2$. ¿Hay alguna diferencia?

# **Procedimiento**
# 
# - Crea una matriz cuadrada $P$ de números aleatorios.- Comprueba que sea invertible. Si no lo es, repite el primer paso hasta obtener una matriz invertible.- <¡ tu comentario aquí !>
# - Utiliza $XP$ como la nueva matriz de características

# Probamos el modelo con los datos originales:

# In[95]:


model_lr= LinearRegression()


# In[96]:


model_lr.fit(X_train, y_train)
predictions= model_lr.predict(X_test)

rmse_or = math.sqrt(sklearn.metrics.mean_squared_error(y_test, predictions))
print('RMSE original: ', rmse_or)

r2_score_or = sklearn.metrics.r2_score(y_test, predictions)
print('R2 original: ', r2_score_or)   


# Ahora con los datos ofuscados:

# In[97]:


X = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits'].to_numpy()

P_check= np.random.random(size=(X.shape[1], X.shape[1]))
ofs_check= X @ P_check

ofs_train, ofs_test, y_train, y_test = train_test_split(ofs_check, y, test_size=0.3, random_state=12345)


# In[98]:


model_lr_ofs= LinearRegression()

model_lr_ofs.fit(ofs_train, y_train)
predictions_ofs= model_lr_ofs.predict(ofs_test)

rmse_or = math.sqrt(sklearn.metrics.mean_squared_error(y_test, predictions_ofs))
print('RMSE ofuscado: ', rmse_or)

r2_score_or = sklearn.metrics.r2_score(y_test, predictions_ofs)
print('R2 ofuscado: ', r2_score_or)   


# Vemos que el RMSE y el R2 son iguales tanto para los datos originales com para los ofuscados, pues como vimos anteriormente, al ofuscar los datos, los multiplcamos por cierta matriz invertible, y se hace lo mismo con los pesos, de esa manera se obtiene el mismo resultado para ambos datos 

# # Conclusiones

# Creamos una función para encontrar clientes que sean similares a un cliente determinado. Esto ayudará a los agentes de la compañía con el marketing.
# 
# Creamos un calsificador basado en KNN para predecir la probabilidad de que un nuevo cliente reciba una prestación del seguro. Y lo calificamos ycomparamos con un modelo dummy
# Creamos un modelo de regresión lineal desde cero para predecir el número de prestaciones de seguro que un nuevo cliente pueda recibir
# Ofuscamos los datos para proteger los datos personales de los clientes sin afectar al modelo. 
# Y demostramos que el modelo predice y trabaja igual con datos ofuscados y con los datos originales

# <div class="alert alert-block alert-danger">
#     
# # Comentarios generales
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Buen trabajo, Denisse. Nos quedan algunos elementos por resolver antes de poder aprobar tu proyecto. He dejado comentarios a lo largo del documento para ello.
# </div>

# <div class="alert alert-block alert-success">
#     
# # Comentarios generales
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Todo corregido. Has aprobado un nuevo proyecto. ¡Felicitaciones!
# </div>

# # Lista de control

# Escribe 'x' para verificar. Luego presiona Shift+Enter.

# - [x]  Jupyter Notebook está abierto
# - [ ]  El código no tiene errores- [ ]  Las celdas están ordenadas de acuerdo con la lógica y el orden de ejecución
# - [ ]  Se ha realizado la tarea 1
#     - [ ]  Está presente el procedimiento que puede devolver k clientes similares para un cliente determinado
#     - [ ]  Se probó el procedimiento para las cuatro combinaciones propuestas    - [ ]  Se respondieron las preguntas sobre la escala/distancia- [ ]  Se ha realizado la tarea 2
#     - [ ]  Se construyó y probó el modelo de clasificación aleatoria para todos los niveles de probabilidad    - [ ]  Se construyó y probó el modelo de clasificación kNN tanto para los datos originales como para los escalados. Se calculó la métrica F1.- [ ]  Se ha realizado la tarea 3
#     - [ ]  Se implementó la solución de regresión lineal mediante operaciones matriciales    - [ ]  Se calculó la RECM para la solución implementada- [ ]  Se ha realizado la tarea 4
#     - [ ]  Se ofuscaron los datos mediante una matriz aleatoria e invertible P    - [ ]  Se recuperaron los datos ofuscados y se han mostrado algunos ejemplos    - [ ]  Se proporcionó la prueba analítica de que la transformación no afecta a la RECM    - [ ]  Se proporcionó la prueba computacional de que la transformación no afecta a la RECM- [ ]  Se han sacado conclusiones

# # Apéndices
# 
# ## Apéndice A: Escribir fórmulas en los cuadernos de Jupyter

# Puedes escribir fórmulas en tu Jupyter Notebook utilizando un lenguaje de marcado proporcionado por un sistema de publicación de alta calidad llamado $\LaTeX$ (se pronuncia como "Lah-tech"). Las fórmulas se verán como las de los libros de texto.
# 
# Para incorporar una fórmula a un texto, pon el signo de dólar (\\$) antes y después del texto de la fórmula, por ejemplo: $\frac{1}{2} \times \frac{3}{2} = \frac{3}{4}$ or $y = x^2, x \ge 1$.
# 
# Si una fórmula debe estar en el mismo párrafo, pon el doble signo de dólar (\\$\\$) antes y después del texto de la fórmula, por ejemplo:
# $$
# \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.
# $$
# 
# El lenguaje de marcado de [LaTeX](https://es.wikipedia.org/wiki/LaTeX) es muy popular entre las personas que utilizan fórmulas en sus artículos, libros y textos. Puede resultar complicado, pero sus fundamentos son sencillos. Consulta esta [ficha de ayuda](http://tug.ctan.org/info/undergradmath/undergradmath.pdf) (materiales en inglés) de dos páginas para aprender a componer las fórmulas más comunes.

# ## Apéndice B: Propiedades de las matrices

# Las matrices tienen muchas propiedades en cuanto al álgebra lineal. Aquí se enumeran algunas de ellas que pueden ayudarte a la hora de realizar la prueba analítica de este proyecto.

# In[ ]:




