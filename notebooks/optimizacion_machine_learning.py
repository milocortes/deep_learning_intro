import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimización para Machine Learning
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Machine learning es una de las aplicaciones modernas más importantes de optimización. Cada modelo de machine learning es entrenado usando optimización.

    Desarrollar un modelo de aprendizaje de máquina involucra 5 etapas clave :

    - **El problema** : formular cuáles son las entradas y salidas a modelar,
    - **Los datos** : recolectar y seleccionar datos de entrenamiento para informar el modelo,
    - **El modelo** : elegir una arquitectura para representar el modelo,
    - **La función de pérdida** : diseñar una función de pérdida para evaluar el desempeño del modelo,
    - **La optimización** : implementar un algortimo de optimización para entrenar el modelo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Usamos optimización para ajustar los parámetros del modelo con el objetivo que las salidas de este se ajusten lo mejor posible a los datos al minimizar la función de pérdida. Fundamentalmente, este es un problema inverso que resolvemos usando optimización. Si los datos son el combustible de machine learning, la optimización es el motor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La organización de estos pasos es sólo aproximada, y hay traslapes entre cada etapa. Por ejemplo, elegir el problema y los datos son dos decisiones muy relacionadas. De forma similar, diseñar una función de pérdida personalizada e implementar el algoritmo de optimización están estrechamente vinculados.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Generalmente el modelo de machine learning puede ser escrito como :

    \begin{equation}
    \mathbf{y}=\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}),
    \end{equation}

    donde son $\mathbf{x}$ los insumos del modelo, $\mathbf{y}$ son las salidas, también conocidas como *targets*,  y $\boldsymbol{\theta}$ son los parámetros que se deben optimizar. La función objetivo, que llamamos *función de pérdida* (*loss function*), $L$, por lo general incluye varios términos que combinan diversos objetivos contrapuestos.

    Para un modelo de regresión, el término de pérdida principal es el Error Cuadrático Médio (MSE) del ajuste del modelo a los datos.

    \begin{equation}
    L=\sum_{j=1}^N\left\|\mathbf{y}_j-\mathbf{f}_{\boldsymbol{\theta}}\left(\mathbf{x}_j\right)\right\|^2 .
    \end{equation}

    Este término de error es sumado sobre los $N$ datos $\left\{\mathbf{x}_j, \mathbf{y}_j\right\}_{j=1}^N$..
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regresion por Mínimos Cuadrados

    La regresión por minimos cuadrados es uno de los conceptos fundacionales de matemáticas aplicadas y estadística, con aplicaciones al análisis de datos, problemas inversos, teoría de control y machine learning.

    El problema de regresión lineal por mínimos cuadrados involucra resolver para los parámetros desconocidos $\mathbf{x}$ de un modelo que es escrito como un sistema lineal de ecuaciones:

    \begin{equation}
    \mathbf{A x}=\mathbf{b} .
    \end{equation}

    Para una matriz invertible $\mathbf{A}$ esto equivale a un problema de inversión de matrices, pero cuando $\mathbf{A}$ no es cuadrada, este requiere una solución vía optimización para encontrar el *mejor* ajuste.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dado un conjunto de datos de las variables predictoras y sus resultados asociados, ordenados como filas de una matriz $\mathbf{A} \in \mathbb{R}^{m \times n}$ y un vector $\mathbf{b} \in \mathbb{R}^m$, respectivamente, la regresión busca encontrar la relación entre las columnas de $\mathbf{A}$ que es más consistente con resultados asociados en $\mathbf{b}$.

    Esta relación es cuantificada por el sistema linear de ecuaciones :

    \begin{equation}
    \mathbf{A x} \approx \mathbf{b},
    \end{equation}

    que indica que el vector $\mathbf{b}$ puede ser aproximado como una combinación lineal de las columnas de $\mathbf{A}$. Esta combinación lineal está dada por el vector, que está por determinarse.

    El vector de salida $\mathbf{b}$ está frecuentemente contaminado con ruido de medición, el cual tipicamente es modelado como la suma de un vector $\boldsymbol{\epsilon}$ de ruido blanco gausiano independiente e identicamente distribuido $\mathbf{b}=\mathbf{b}_{\text {true }}+\boldsymbol{\epsilon}$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Es posible resolver el problema lineal de mínimos cuadrados en (2) analiticamente al calcular el gradiente de la función objetivo e igualarlo a cero. Podemos expandir la función objetivo como :

    \begin{equation}
    \begin{aligned}
    \|\mathbf{A x}-\mathbf{b}\|_2^2 & =(\mathbf{A x}-\mathbf{b})^T(\mathbf{A x}-\mathbf{b}) \\
    & =\left(\mathbf{x}^T \mathbf{A}^T-\mathbf{b}^T\right)(\mathbf{A x}-\mathbf{b}) \\
    & =\mathbf{x}^T \mathbf{A}^T \mathbf{A} \mathbf{x}-\mathbf{x}^T \mathbf{A}^T \mathbf{b}-\mathbf{b}^T \mathbf{A} \mathbf{x}+\mathbf{b}^T \mathbf{b} .
    \end{aligned}
    \end{equation}

    Calcular el gradiente e igualarlo a cero produce:

    \begin{equation}
    \begin{aligned}
    \nabla\|\mathbf{A x}-\mathbf{b}\|_2^2 & =2 \mathbf{A}^T \mathbf{A} \mathbf{x}-\mathbf{A}^T \mathbf{b}-\mathbf{A}^T \mathbf{b}=0 \\
    & \Longrightarrow \mathbf{A}^T \mathbf{A} \mathbf{x}=\mathbf{A}^T \mathbf{b} \\
    & \Longrightarrow \mathbf{x}=\left(\mathbf{A}^T \mathbf{A}\right)^{-1} \mathbf{A}^T \mathbf{b} .
    \end{aligned}
    \end{equation}

    La matriz $\mathbf{A}^{\dagger}=\left(\mathbf{A}^T \mathbf{A}\right)^{-1} \mathbf{A}^T$ es conocida como la pseudo-inversa de Moore-Penrose, y esta está definida incluso para matrices no cuadradas $\mathbf{A}$. Sin embargo, esto es sólo válido cuando $\left(\mathbf{A}^T \mathbf{A}\right)$ es invertible. La matriz $\mathbf{A}^T \mathbf{A}$ es invertible cuando $\mathbf{A}$ tiene $n$ columnas linealmente independientes y $m \geq n$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Descenso del Gradiente Estocástico y ADAM

    Descenso de gradiente estocástico (SGD-Stochastic gradient descent) es una variante del algoritmo de descenso de gradiente que escala a problemas de optimización de altas dimensiones al aproximar el gradiente en cada paso, haciéndolo adecuado para el entrenamiento de modelos sofisticados, como el entrenamiento de redes neuronales de gran escala. ADAM (adaptive momentum estimation) es un algoritmo de optimización que combina SGD con inercia (momentum) y tasa de aprendizaje adaptativa, via AdaGrad y RMSProp, generando actualizaciones suaves y más responsivas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    SGD tiene más sentido en el contexto de machine learning, donde estamos optimizando los parámetros $\boldsymbol{\theta}$ de una función $\mathrm{f}_\theta(\mathrm{x})$, tal como en un problema de regresión o una red neuronal, para ajustarse mejor a los datos observados.

    Dados pares de datos de entrada y salida $\left\{\mathbf{x}_j, \mathbf{y}_j\right\}_{j=1}^N$, buscamos encontrar los parámetros $\boldsymbol{\theta}$ de manera que $\mathbf{f}_{\boldsymbol{\theta}}\left(\mathbf{x}_j\right)$ mejor aproxime $\mathbf{y}_j$ promediado sobre los datos.

    \begin{equation}
    \min _{\boldsymbol{\theta}} \sum_{j=1}^N\left\|\mathbf{f}_{\boldsymbol{\theta}}\left(\mathbf{x}_j\right)-\mathbf{y}_j\right\|^2
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    En machine learning se denomina al objetivo una *función de pérdida* $L(\mathbf{x}, \mathbf{y} ; \boldsymbol{\theta})=\left\|\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x})-\mathbf{y}\right\|^2$ la cual minimizamos sobre $\theta$. Es posible calcular el gradiente de la función de pérdida con respecto a $\boldsymbol{\theta}$. Es posible calcular el gradiente de la función de pérdida con respecto a $\boldsymbol{\theta}$:

    \begin{equation}
    \nabla L=2 \nabla \mathbf{f}_{\boldsymbol{\theta}}^T \cdot\left(\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x})-\mathbf{y}\right)
    \end{equation}


    NOTA: el gradiente de puede ser calculado analiticamente o aproximado usando el algoritmo de backpropagation si se trata de una red neuronal. Backpropagation es esencialmente la regla de la cadena aplicada a capas de una red neuronal usando diferenciación automática. Diferenciación automática y backpropagation son la columna vertebral del entrenamiento de machine learning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    SGD aproxima este gradiente en sólo un subconjunto de los datos, llamado *batch*, en lugar de tomar el conjunto completo de datos. SGD evalúa la función de pérdida y los gradientes en batches aleatorios de datos, añadiendo aleatoriedad al proceso. Esto tiene varios beneficios clave como :
    - Aceleraciones computacionales,
    - Robustez frente al ruido y al estancamiento en mínimos locales,
    - Mejor exploración del landscape de optimización, etc.

    Por lo tanto, SGD introduce un nuevo hiperparámetro : el tamaño del batch.

    El algoritmo de descenso del gradiente estocástico es nombrado *estocástico* dado que elige una **una sóla** muestra aleatoria a la vez, ajustando los pesos con el fin de mejorar el desempeño de **sola** esa muestra. Esto puede generar movimientos muy irregulares, por lo que es habitual calcular el gradiente sobre lotes (*batches*) de muestras de entrenamiento en lugar de sobre una sola muestra.

    En el **entrenamiento por lote** (**batch training**) calculamos el gradiente sobre el conjunto de entrenamiento completo. Al observar tantas muestras, el entrenamiento por lotes ofrece una estimación muy buena de la dirección en la que deben moverse los pesos, pero con el costo de gastar tiempo de procesamiento para cada muestra única en el conjunto de entrenamiento para calcular la dirección de ajuste.

    Una alternativa es el **entrenamiento por mini-lote** (**mini-batch training**) : entrenamos para un grupo de $m$ muestras (512 o 1024, por ejemplo) menor a la cantidad total de las muestras del conjunto de datos completo. Si $m$ es del tamaño del conjunto de datos completo, usamos el descenso de gradiente por **lote**; si $m=1$, usamos el algoritmo de descenso del gradiente estocástico.

    El entrenamiento por mini-batch tiene la ventaja de ser computacionalmente eficiente. Los mini-batches pueden ser facilmente vectorizados, lo cual permite procesar todos los batches en paralelo para posteriormente acumular la pérdida, algo que no es posible con el entrenamiento por batch o por actualización individual.

    La regla de actualización del algoritmo puede ser escrita como :

    \begin{equation}
    \boldsymbol{\theta}_{k+1}=\boldsymbol{\theta}_k-\gamma \nabla_{\boldsymbol{\theta}} L
    \end{equation}

    En principio, nada parece que haya cambiado del algoritmo de descenso de gradiente. Toda la aleatoriedad está escondida dentro del gradiente $\nabla_{\boldsymbol{\theta}} L$. Esta es una característica, no un bug, y significa que podemos extender de forma sencilla SGD con otras reglas de actualización basadas en gradientes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SGD para resolver un problema de Regresión Lineal

    Estamos listos para probar el SGD para resolver un problema de regresión lineal simple.

    Primero, generamos datos a partir de un modelo lineal con dos estados dimensionales de $\boldsymbol{x}$ y agregamos una cantidad relativamente grande de ruido.
    """)
    return


@app.cell
def _(np):
    N = 2000 # number of samples
    A = np.random.randn(N, 2) # design matrix
    true_x = np.array([2.0, -3.0]) # true slopes
    true_b = 5.0
    # True y = A * x + b plus some noise
    noise_std = 0.5
    noise = noise_std * np.random.randn(N)
    y = A.dot(true_x) + true_b + noise
    return A, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Definimos funciones que calculan la predicción del modelo y el error cuadrático medio sobre el conjunto de datos.
    """)
    return


@app.cell
def _(np):
    def predict(A, x, b):
        # A: shape (batch_size, 2)
        # x: shape (2,)
        # b: scalar
        return A.dot(x) + b # shape (batch_size,)

    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    return mean_squared_error, predict


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Definimos la función que calcula el gradiente aproximado sobre un lote de datos.

    El gradiente para los pesos $\mathbf{x}$ es igual a :

    \begin{equation}
    \dfrac{2}{m} \mathbf{A}^T(\mathbf{y}-\boldsymbol{A} \mathbf{x} - b)
    \end{equation}

    y para el intercepto $b$:

    \begin{equation}
    \dfrac{2}{m} \mathbf{1}^T (\mathbf{y}-\boldsymbol{A} \mathbf{x} - b)
    \end{equation}

    donde $\mathbf{1}$ es el vector de unos :

    \begin{equation}
    \mathbf{1}=\left[\begin{array}{l}
    1 \\
    1 \\
    1
    \end{array}\right]
    \end{equation}

    y $m$ es el tamaño de batch.
    """)
    return


@app.cell
def _(np, predict):
    def batch_gradients(A_batch, y_batch, x, b):
        # A_batch, x, b: same shape
        # y_batch: shape (batch_size,)
        batch_size = A_batch.shape[0]
        y_pred = predict(A_batch, x, b)
        residuals = (y_pred - y_batch)
        grad_x = (2.0 / batch_size) * A_batch.T.dot(residuals)
        grad_b = (2.0 / batch_size) * np.sum(residuals)
        return grad_x, grad_b

    return (batch_gradients,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ahora definimos los tres métodos a consideración :
    - Descenso de gradiente estándar (full-batch).
    - Descenso de gradiente estocástico con mini-batch.
    - Descenso de gradiente estocástico con un tamaño de batch de $n=1$. La tasa de aprendizaje es $\gamma$.
    """)
    return


@app.cell
def _(batch_gradients, mean_squared_error, np, predict):
    # Descenso por gradiente estándar
    def full_batch_gd(A, y, gamma=0.01, n_epochs=50):
        print("\nDescenso por gradiente estándar\n")
        x = np.zeros(A.shape[1]) # unknown parameter vector
        b = 0.0
        mse_history = []
        for epoch in range(n_epochs):
            grad_x, grad_b = batch_gradients(A, y, x, b)
            x -= gamma * grad_x
            b -= gamma * grad_b

            y_pred = predict(A, x, b)
            mse = mean_squared_error(y, y_pred)
            mse_history.append(mse)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss (MSE) = {mse:.4f}")

        results = {
            "x" : x, 
            "b" : b, 
            "mse_history" : mse_history
        }

        return results

    return (full_batch_gd,)


@app.cell
def _(batch_gradients, mean_squared_error, np, predict):
    # SGD con mini-batch (SGD con un tamaño de batch pequeño, n = 50)
    def mini_batch_sgd(A, y, batch_size=50, gamma=0.01, n_epochs=50):
        print("\nSGD con mini-batch (SGD con un tamaño de batch pequeño, n = 50)\n")
        x = np.zeros(A.shape[1])
        b = 0.0
        N = A.shape[0]
        mse_history = []

        for epoch in range(n_epochs):
            indices = np.random.permutation(N) # Shuffle data
            for start in range(0, N, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                A_batch = A[batch_idx]
                y_batch = y[batch_idx]
                grad_x, grad_b = batch_gradients(A_batch, y_batch, x, b)
                x -= gamma * grad_x
                b -= gamma * grad_b
            # Check MSE after each epoch
            y_pred = predict(A, x, b)
            mse = mean_squared_error(y, y_pred)
            mse_history.append(mse)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss (MSE) = {mse:.4f}")

        results = {
            "x" : x, 
            "b" : b, 
            "mse_history" : mse_history
        }

        return results

    return (mini_batch_sgd,)


@app.cell
def sgd_batch1(batch_gradients, mean_squared_error, np, predict):
    # SGD con tamaño de batch igual a 1
    def sgd_batch1(A, y, gamma=0.001, n_epochs=50):
        print("\nSGD con tamaño de batch igual a 1\n")
        x = np.zeros(A.shape[1])
        b = 0.0
        N = A.shape[0]
        mse_history = []

        for epoch in range(n_epochs):
            indices = np.random.permutation(N)
            for i in indices:
                A_i = A[i:i+1] # shape (1,2)
                y_i = y[i:i+1] # shape (1,)
                grad_x, grad_b = batch_gradients(A_i, y_i, x, b)
                x -= gamma * grad_x
                b -= gamma * grad_b
            # MSE after epoch
            y_pred = predict(A, x, b)
            mse = mean_squared_error(y, y_pred)
            mse_history.append(mse)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss (MSE) = {mse:.4f}")

        results = {
            "x" : x, 
            "b" : b, 
            "mse_history" : mse_history
        }

        return results

    return (sgd_batch1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Históricamente SGD "verdadero" solía usar un tamaño de batch de 1, aunque ahora en la práctica casi siempre se usa mini-batch. Para SGD con tamaño de batch de 1, se reduce la tasa de aprendizaje para disminuir la sensibilidad al ruido en los datos y el riesgo de divergencia.

    SGD y descenso por gradiente son los extremos de SGD por mini-batch.

    Ejecutemos los métodos:
    """)
    return


@app.cell
def _(A, full_batch_gd, mini_batch_sgd, sgd_batch1, y):
    # Guardemos los resultados de los métodos en un diccionario
    n_epochs = 50
    resultados_metodos = {
        "GD (full-batch)" : full_batch_gd(A, y, gamma=0.01, n_epochs=n_epochs), 
        "Mini-Batch GD" : mini_batch_sgd(A, y, batch_size=50, gamma=0.01, n_epochs=n_epochs), 
        "SGD (batch = 1)" : sgd_batch1(A, y, gamma=0.001, n_epochs=n_epochs)
    } 
    return n_epochs, resultados_metodos


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Comparemos los resultados de los tres métodos. Mini-batch y SGD (batch = 1) convergen en muchos menos épocas que el descenso de gradiente con batch completo.

    Sin embargo, una época denota una única pasada por todos los datos, de manera que una época con tamaño de batch igual a 1 involucra $n = 2000$ pasos individuales, cada uno utilizando un único punto de datos.
    """)
    return


@app.cell
def _(n_epochs, plt, resultados_metodos):
    ## Graficamos los resultados
    fig, ax = plt.subplots()

    for metodo, resultados in resultados_metodos.items():
        ax.semilogy(
            range(n_epochs), 
            resultados["mse_history"], 
            label = metodo, 
            marker='o', 
            linestyle='dotted'
        )
    ax.grid()
    ax.grid(which="minor", color="0.9")
    ax.legend()
    plt.ylabel("MSE (log scale)")
    plt.show()
    return


@app.cell
def _(resultados_metodos):
    resultados_metodos
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verificación de Resultados con [`statsmodels`](https://www.statsmodels.org/stable/index.html)
    """)
    return


@app.cell
def _(A, y):
    import statsmodels.api as sm

    X = sm.add_constant(A)
    model = sm.OLS(y, X)
    results = model.fit()

    print(results.summary())
    return (sm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regresión Logística
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Componentes de un clasificador probabilístico de Aprendizaje de Máquina
    La regresión logística es un clasificador probabilístico que hace uso de aprendizaje de máquina supervisado. Un sistema de aprendizaje de máquina que resuelve un problema de clasificación tiene cuatro componentes:

    - Una **representación de carácterísticas** de los inputs. Para cada observación de entrada $x^{(i)}$, se representará como un vector de carácterísticas $[x_1, x_2, \dots, x_n]$.
    -  Una función de clasificación que calcula $\hat{y}$, la clase estimada, via $p(y|x)$. Usaremos la función **sigmoide** para este fin.
    -  Una función objetivo para el aprendizaje, la cual es una medida de desempeño del modelo la cual minimiza el error para los datos de entrenamiento.
    -  Un algoritmo para optimizar la función objetivo. Usaremos SGD para este fin.

    Una regresión logística involucra dos etapas :
    - **Training** : Entrenar el sistema (específicamente los pesos $w$ y sesgo $b$) mediante el algoritmo de descenso de gradiente estocástico y la función de pérdida de *entropía cruzada* (*cross-entropy*).
    - **Test** : Dada una muestra de prueba $x$ calculamos $p(y|x)$ para calcular la probabilidad más alta para las etiquetas $y = 1$ o $y=0$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Clasificación : la función sigmoide

    El objetivo de una regresión logística binaria es entrenar un clasificador que pueda generar decisiones binarias acerca de la clase o etiqueta de una nueva observación de entrada.

    Consideremos una observación $x$ que representamos como un vector de carácterísticas $[x_1, x_2, \dots, x_n]$. La salida del clasificador $y$ puede ser 1 o 0. Queremos saber la probabilidad $P(y=1|x)$ que la observación sea miembra de la clase 1.

    La regresión logística resuelve esta tarea al aprender, a partir de los datos de entrenamiento, un vector de **pesos** y un **término de sesgo**. Los pesos $w_i$ nos dicen cuán importante es la característica en la decisión de clasificación. Los peros pueden ser positivo (proporcionando evidencia de que el caso que se está clasificando pertenece a la clase positiva) o negativos (proporcionando evidencia de que el caso que se está clasificando pertenece a la clase negativa).

    Para hacer decisiones sobre caso (porterior a que hemos aprendido los pesos en el entrenamiento) el clasificador multiplica cada característica $x_i$ por su peso $w_i$ (resumiendo así las características ponderadas) y agrega el término de sesgo $b$. El resultado es un valor numérico $z$ que expresa la suma ponderada de la evidencia para la clase.

    \begin{equation}
    z=\left(\sum_{i=1}^n w_i x_i\right)+b
    \end{equation}

    Representaremos esta suma como un **producto punto**.

    \begin{equation}
    z=w \cdot x+b
    \end{equation}

    La ecuación anterior no forza a $z$ a ser una probabilidad, esto es, su rango no se encuentra entre 0 y 1.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Para construir la probabilidad, pasaremos $z$ a la función **sigmoide** $\sigma (z)$. La función sigmoide (llamada así porque su forma es parecida a la letra $s$) es también llamada **función logística**. La función tiene la siguiente forma :

    \begin{equation}
    y=\sigma(z)=\frac{1}{1+e^{-z}}=\frac{1}{1+\exp (-z)}
    \end{equation}

    La función sigmoide tiene varias ventajas como :
    - Toma valores reales y los mapea en un rango de $[0,1]$.
    - Dado que es casi lineal cerca de 0 pero se aplana hacia los extremos, tiende a comprimir los valores atípicos hacia 0 o 1.
    - Es diferenciable, lo cual será útil para el aprendizaje.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Si aplicamos la sigmoide a la suma ponderada de características, obtenemos un número entre 0 y 1. Para que sea una probabilidad, solo necesitamos asegurarnos de que los dos caso, $p(y=1)$ y $p(y=0)$, sumen 1. Probaremos esto a continuación :

    \begin{equation}
    \begin{aligned}
    P(y=1) & =\sigma(w \cdot x+b) \\
    & =\frac{1}{1+\exp (-(w \cdot x+b))} \\
    P(y=0) & =1-\sigma(w \cdot x+b) \\
    & =1-\frac{1}{1+\exp (-(w \cdot x+b))} \\
    & =\frac{\exp (-(w \cdot x+b))}{1+\exp (-(w \cdot x+b))}
    \end{aligned}
    \end{equation}

    La función sigmoide tiene la propiedad :

    \begin{equation}
    1-\sigma(x)=\sigma(-x)
    \end{equation}

    De manera que podemos expresar $P(y=0)$ como $\sigma(-(w \cdot x+b))$

    Ahora disponemos de un modelo que, dada una instancia x, calcula la probabilidad. Cómo tomamos una decisión? para instancia de prueba $x$, decimos **sí** si la probabilidad $P(y = 1 | x)$ es mayor que 0.5, y **no** en el caso contrario. Llamamos 0.5 la **frontera de decisión** :

    \begin{equation}
    \hat{y}= \begin{cases}1 & \text { if } P(y=1 \mid x)>0.5 \\ 0 & \text { otherwise }\end{cases}
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Aprendizaje

    Cómo son aprendidos los parámetros, $w$ y $b$, del modelo? La regresión logística es un caso de clasificación supervisada en el que conocemos la etiqueta correcta $y$ (0 o 1) para cada observación $x$. Lo que el modelo produce es $\hat{y}$, la estimación del sistema del valor real de $y$. Queremos aprender los parámetros  que hacen que $\hat{y}$ para cada observación de entrenamiento sea lo más cercana posible al valor verdadero de $y$.

    Para esto requerimos dos componentes. El primero es una métrica que cuantifique cuán cercana es la etiqueta estimada por el sistema ($\hat{y}$) de su etiqueta verdadera $y$. Está distancia es cuantificada por la **función de pérdida**. Usaremos la **función de pérdida entropía cruzada** para este objetivo. El segundo componente es el algoritmo de optimización el cual se utiliza como mecanismo iterativo de actualización de los pesos y sesgo para minimizar la función de pérdida. Usaremos SGD para esta tarea.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Función de pérdida entropía cruzada

    Necesitamos una función de pérdida que exprese, para una observación x, qué tan cerca está la salida del clasificador ($\hat{y} = \sigma (w \cdot x + b)$) de su valor real ($y$ que toma valor 1 o 0). Llamemos a esta función:

    \begin{equation}
    L(\hat{y}, y)=\text { How much } \hat{y} \text { differs from the true } y
    \end{equation}

    Esto se realiza mediante una función de pérdida que favorece una mayor probabilidad para las etiquetas de clase correctas de los ejemplos de entrenamiento. Esta función se obtiene mediente una **estimación por máxima verosimilitud condicional** : escogemos los parámetros $w$ y $b$ que **maximizan la log probabilidad de las etiquetas verdaderas $y$ en el conjunto de entrenamiento** dadas las observaciones $x$. La función de pérdida resultante es el *negativo del logaritmo de la función de verosimilitud*, la cuál se denomina **entropía cruzada**.

    Derivemos esta función de pérdida para una sola observación $x$. Nuesto objetivo es aprender los pesos que maximizan la probabilidad de la etiqueta correcta $p(y|x)$. Dado que existen sólo dos salidas discretas (1 o 0), esta se modela como una distribución Bernoulli, y podemos expresar la probabilidad $p(y|x)$ que nuestro clasificador produce para una observación como

    \begin{equation}
    p(y \mid x)=\hat{y}^y(1-\hat{y})^{1-y}
    \end{equation}

    /// admonition | Nota

    Sí $y=1$, la ecuación simplifica a $\hat{y}$; si $y=0$, la ecuación simplifica a $1-\hat{y}$
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ahora tomemos logaritmos en ambos lados de la expresión.

    \begin{equation}
    \begin{aligned}
    \log p(y \mid x) & =\log \left[\hat{y}^y(1-\hat{y})^{1-y}\right] \\
    & =y \log \hat{y}+(1-y) \log (1-\hat{y})
    \end{aligned}
    \end{equation}

    Esta última expresión describe la log verosimilitud que debemos maximizar. Para convertirla en una función de pérdida (que necesitamos minimizar), simplemente invertiremos el signo. El resultado es la pérdida de entropía cruzada $L_{CE}$

    \begin{equation}
    L_{\mathrm{CE}}(\hat{y}, y)=-\log p(y \mid x)=-[y \log \hat{y}+(1-y) \log (1-\hat{y})]
    \end{equation}

    Finalmente, podemos introducir la definición de $\hat{y} = \sigma (w \cdot x + b)$

    \begin{equation}
    L_{\mathrm{CE}}(\hat{y}, y)=-[y \log \sigma(w \cdot x+b)+(1-y) \log (1-\sigma(w \cdot x+b))]
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gradiente de la entropía cruzada

    Nuestro objetivo con el algoritmo de descenso del gradiente es encontrar los pesos óptimos que minimizan la función de pérdida que hemos definido para el modelo ($L_{CE}$). Nuestro objetivo es encontrar el conjunto de pesos que minimizan la función de pérdid,  promediada sobre todas las muestras de entrenamiento :

    \begin{equation}
    \hat{\theta}=\underset{\theta}{\operatorname{argmin}} \frac{1}{m} \sum_{i=1}^m L_{\mathrm{CE}}\left(f\left(x^{(i)} ; \theta\right), y^{(i)}\right)
    \end{equation}

    Con el objetivo de actualizar $\theta$, necesitamos una definición del gradiente $\nabla L(f(x ; \theta), y)$. El gradiente de la función de pérdida de la entropía cruzada para un peso $w_j$ es :

    \begin{equation}
    \frac{\partial L_{\mathrm{CE}}(\hat{y}, y)}{\partial w_j}=[\sigma(w \cdot x+b)-y] x_j
    \end{equation}

    Nota que la expresión del gradiente corresponde a un solo peso $w_j$ representa un valor muy intuitivo : la diferencia entre $y$ y nuestra estimación $\hat{y} = \sigma (w \cdot x +b)$ multiplicado por el el vector de características $x_j$.

    El gradiente de la función de pérdida de la entropía cruzada para el sesgo b$ es :

    \begin{equation}
    \frac{\partial L_{\mathrm{CE}}(\hat{y}, y)}{\partial b}=[\sigma(w \cdot x+b)-y]
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Derivación del Gradiente
        type: info

    Tengamos en cuenta tres elementos. El primero, la derivada de $\ln (x)$ :

    \begin{equation}
    \frac{d}{d x} \ln (x)=\frac{1}{x}
    \end{equation}

    Segundo, la derivada de la función sigmoide :

    \begin{equation}
    \frac{d \sigma(z)}{d z}=\sigma(z)(1-\sigma(z))
    \end{equation}

    Tercero, la **regla de la cadena**. Supongamos que estamos calculando la derivada de una composición de funciones $f(x) = u (v(x))$. La derivada de $f(x)$ es la derivada de $u(x)$ con respecto a $v(x)$ con respecto a $x$ :

    \begin{equation}
    \frac{d f}{d x}=\frac{d u}{d v} \cdot \frac{d v}{d x}
    \end{equation}

    Queremos calcular la derivada de la función de pérdida con respecto a un peso $w_j$ :

    \begin{equation}
    \begin{aligned}
    \frac{\partial L_{\mathrm{CE}}}{\partial w_j} & =\frac{\partial}{\partial w_j}-[y \log \sigma(w \cdot x+b)+(1-y) \log (1-\sigma(w \cdot x+b))] \\
    & =-\left[\frac{\partial}{\partial w_j} y \log \sigma(w \cdot x+b)+\frac{\partial}{\partial w_j}(1-y) \log [1-\sigma(w \cdot x+b)]\right]
    \end{aligned}
    \end{equation}

    Posteriormente, usando la regla de la cadena, y basándonos en la derivada del logaritmo:

    \begin{equation}
    \frac{\partial L_{\mathrm{CE}}}{\partial w_j}=-\frac{y}{\sigma(w \cdot x+b)} \frac{\partial}{\partial w_j} \sigma(w \cdot x+b)-\frac{1-y}{1-\sigma(w \cdot x+b)} \frac{\partial}{\partial w_j} 1-\sigma(w \cdot x+b)
    \end{equation}

    Reordenando términos :

    \begin{equation}
    \frac{\partial L_{\mathrm{CE}}}{\partial w_j}=-\left[\frac{y}{\sigma(w \cdot x+b)}-\frac{1-y}{1-\sigma(w \cdot x+b)}\right] \frac{\partial}{\partial w_j} \sigma(w \cdot x+b)
    \end{equation}

    Sustituyendo la derivada de la sigmoide y aplicando la regla de la cadena una vez más, obtenemos:

    \begin{equation}
    \begin{aligned}
    \frac{\partial L_{\mathrm{CE}}}{\partial w_j} & =-\left[\frac{y-\sigma(w \cdot x+b)}{\sigma(w \cdot x+b)[1-\sigma(w \cdot x+b)]}\right] \sigma(w \cdot x+b)[1-\sigma(w \cdot x+b)] \frac{\partial(w \cdot x+b)}{\partial w_j} \\
    & =-\left[\frac{y-\sigma(w \cdot x+b)}{\sigma(w \cdot x+b)[1-\sigma(w \cdot x+b)]}\right] \sigma(w \cdot x+b)[1-\sigma(w \cdot x+b)] x_j \\
    & =-[y-\sigma(w \cdot x+b)] x_j \\
    & =[\sigma(w \cdot x+b)-y] x_j
    \end{aligned}
    \end{equation}

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrenamiento por Mini-batch

    El algoritmo de descenso del gradiente estocástico es nombrado *estocástico* dado que elige una **una sóla** muestra aleatoria a la vez, ajustando los pesos con el fin de mejorar el desempeño de **sola** esa muestra. Esto puede generar movimientos muy irregulares, por lo que es habitual calcular el gradiente sobre lotes (*batches*) de muestras de entrenamiento en lugar de sobre una sola muestra.

    En el **entrenamiento por lote** (**batch training**) calculamos el gradiente sobre el conjunto de entrenamiento completo. Al observar tantas muestras, el entrenamiento por lotes ofrece una estimación muy buena de la dirección en la que deben moverse los pesos, pero con el costo de gastar tiempo de procesamiento para cada muestra única en el conjunto de entrenamiento para calcular la dirección de ajuste.

    Una alternativa es el **entrenamiento por mini-lote** (**mini-batch training**) : entrenamos para un grupo de $m$ muestras (512 o 1024, por ejemplo) menor a la cantidad total de las muestras del conjunto de datos completo. Si $m$ es del tamaño del conjunto de datos completo, usamos el descenso de gradiente por **lote**; si $m=1$, usamos el algoritmo de descenso del gradiente estocástico.

    El entrenamiento por mini-batch tiene la ventaja de ser computacionalmente eficiente. Los mini-batches pueden ser facilmente vectorizados, lo cual permite procesar todos los batches en paralelo para posteriormente acumular la pérdida, algo que no es posible con el entrenamiento por batch o por actualización individual.

    Definamos la versión minibatch de la función de pérdida de la entropía cruzada. Extenderemos la entropía cruzada de una sola muestra a un tamaño de muestra de tamaño $m$. Usaremos la notación que $x^{(i)}$ y $y^{(i)}$ significan la $i$-ésima característica y etiqueta, respectivamente. Supongamos que las muestras de entrenamiento son independientes :

    \begin{equation}
    \begin{aligned}
    \log p(\text { training labels }) & =\log \prod_{i=1}^m p\left(y^{(i)} \mid x^{(i)}\right) \\
    & =\sum_{i=1}^m \log p\left(y^{(i)} \mid x^{(i)}\right) \\
    & =-\sum_{i=1}^m L_{\mathrm{CE}}\left(\hat{y}^{(i)}, y^{(i)}\right)
    \end{aligned}
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ahora la función de costo para el mini-batch de $m$ muestras es la pérdida promedio para cada lote:

    \begin{equation}
    \begin{aligned}
    \operatorname{Cost}(\hat{y}, y) & =\frac{1}{m} \sum_{i=1}^m L_{\mathrm{CE}}\left(\hat{y}^{(i)}, y^{(i)}\right) \\
    & =-\frac{1}{m} \sum_{i=1}^m y^{(i)} \log \sigma\left(w \cdot x^{(i)}+b\right)+\left(1-y^{(i)}\right) \log \left(1-\sigma\left(w \cdot x^{(i)}+b\right)\right)
    \end{aligned}
    \end{equation}

    El gradiente de mini-batch es el promedio de los gradientes individuales :

    \begin{equation}
    \frac{\partial \operatorname{Cost}(\hat{y}, y)}{\partial w_j}=\frac{1}{m} \sum_{i=1}^m\left[\sigma\left(w \cdot x^{(i)}+b\right)-y^{(i)}\right] x_j^{(i)}
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Datos : Spector and Mazzeo (1980) - Program Effectiveness Data
    ======================================================

    Description
    -----------

    Experimental data on the effectiveness of the personalized system of instruction (PSI) program

    Notes
    -----
    ::

        Number of Observations - 32

        Number of Variables - 4

        Variable name definitions::

            Grade - binary variable indicating whether or not a student's grade
                    improved.  1 indicates an improvement.
            TUCE  - Test score on economics test
            PSI   - participation in program
            GPA   - Student's grade point average


    Source
    ------

    http://pages.stern.nyu.edu/~wgreene/Text/econometricanalysis.htm

    The raw data was downloaded from Bill Greene's Econometric Analysis web site,
    though permission was obtained from the original researcher, Dr. Lee Spector,
    Professor of Economics, Ball State University.

    Copyright
    ---------

    Used with express permission of the original author, who
    retains all rights.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regresión Logística con [`statsmodels`](https://www.statsmodels.org/stable/index.html)
    """)
    return


@app.cell
def _(sm):
    ## Cargamos datos
    spector_data = sm.datasets.spector.load_pandas()

    ## Extraemos catacterísticas
    spector_data.exog = sm.add_constant(spector_data.exog)

    ## Definimos el modelo y lo estimamos
    logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
    logit_res = logit_mod.fit()

    ## Imprimimos los resultados
    print(logit_res.summary())
    return (logit_res,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regresión Logística desde cero
    """)
    return


@app.cell
def _(np):
    def sigmoid(X):
        return 1.0 / (1.0 + np.exp(-X))

    def predict_log(X, w, b):
        return sigmoid(X.dot(w) + b)

    def loss_cross_entropy(y_true, y_pred):
        # Evitar log(0) con un pequeño épsilon numérico
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


    return loss_cross_entropy, predict_log


@app.cell
def _(np, predict_log):
    ### Función que calcula el gradiente para el lote de dato
    def batch_gradients_log(X_batch, y_batch, w, b):
        # X_batch, w, b: same shape
        # y_batch: shape (batch_size,)
        batch_size = X_batch.shape[0]
        y_pred = predict_log(X_batch, w, b)
        residuals = (y_pred - y_batch)
        grad_w = (1.0 / batch_size) * X_batch.T.dot(residuals)
        grad_b = (1.0 / batch_size) * np.sum(residuals)
        return grad_w, grad_b



    return (batch_gradients_log,)


@app.cell
def _(batch_gradients_log, loss_cross_entropy, mo, np, predict_log):
    # Descenso por gradiente estándar
    def full_batch_gd_log(X, y, gamma=0.00999, n_epochs=500_000):
        print("\nDescenso por gradiente estándar\n")
        w = np.zeros(X.shape[1]) # unknown parameter vector
        b = 0.0
        loss_history = []
        for epoch in mo.status.progress_bar(range(n_epochs), title = "Entrenando SGD (full-batch)" ) :
            grad_w, grad_b = batch_gradients_log(X, y, w, b)
            w -= gamma * grad_w
            b -= gamma * grad_b

            y_pred = predict_log(X, w, b)
            loss = loss_cross_entropy(y, y_pred)
            loss_history.append(loss)

            if epoch % 50_000 == 0:
                print(f"Epoch {epoch}: Loss (Cross-Entropy) = {loss:.8f}")

        results = {
            "w" : w, 
            "b" : b, 
            "loss_history" : loss_history
        }

        return results

    return (full_batch_gd_log,)


@app.cell
def _(batch_gradients_log, loss_cross_entropy, mo, np, predict_log):
    # SGD con mini-batch (SGD con un tamaño de batch pequeño, n = 50)
    def mini_batch_sgd_log(X, y, batch_size=50, gamma=0.01, n_epochs=50):
        print("\nSGD con mini-batch (SGD con un tamaño de batch pequeño, n = 50)\n")
        w = np.zeros(X.shape[1])
        b = 0.0
        N = X.shape[0]
        loss_history = []

        for epoch in mo.status.progress_bar(range(n_epochs), title = "Entrenando SGD con mini-batch" ):
            indices = np.random.permutation(N) # Shuffle data
            for start in range(0, N, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                grad_w, grad_b = batch_gradients_log(X_batch, y_batch, w, b)
                w -= gamma * grad_w
                b -= gamma * grad_b
            # Check MSE after each epoch
            y_pred = predict_log(X, w, b)
            loss = loss_cross_entropy(y, y_pred)
            loss_history.append(loss)

            if epoch % 50_000 == 0:
                print(f"Epoch {epoch}: Loss (Cross-Entropy) = {loss:.8f}")

        results = {
            "w" : w, 
            "b" : b, 
            "loss_history" : loss_history
        }

        return results

    return (mini_batch_sgd_log,)


@app.cell
def _():
    return


@app.cell
def _(batch_gradients_log, loss_cross_entropy, mo, np, predict_log, x):
    # SGD con tamaño de batch igual a 1
    def sgd_batch1_log(X, y, gamma=0.001, n_epochs=50):
        print("\nSGD con tamaño de batch igual a 1\n")
        w = np.zeros(X.shape[1])
        b = 0.0
        N = X.shape[0]
        loss_history = []

        for epoch in mo.status.progress_bar(range(n_epochs), title = "SGD con tamaño de batch igual a 1" ):
            indices = np.random.permutation(N)
            for i in indices:
                X_i = X[i:i+1] # shape (1,2)
                y_i = y[i:i+1] # shape (1,)
                grad_w, grad_b = batch_gradients_log(X_i, y_i, w, b)
                w -= gamma * grad_w
                b -= gamma * grad_b
            # MSE after epoch
            y_pred = predict_log(X, w, b)
            loss = loss_cross_entropy(y, y_pred)
            loss_history.append(loss)

            if epoch % 50_000 == 0:
                print(f"Epoch {epoch}: Loss (Cross-Entropy) = {loss:.8f}")

        results = {
            "w" : x, 
            "b" : b, 
            "loss_history" : loss_history
        }

        return results

    return


@app.cell
def _(full_batch_gd_log, mini_batch_sgd_log, sm):
    # Obtengamos los datos de StatsModels
    X_log = sm.datasets.spector.load_pandas().exog.to_numpy()
    y_log = sm.datasets.spector.load_pandas().endog.to_numpy()

    # Guardemos los resultados de los métodos en un diccionario
    resultados_metodos_log = {
        "GD (full-batch)" : full_batch_gd_log(X_log, y_log, gamma=0.00999, n_epochs=700_000), 
        "Mini-Batch GD" : mini_batch_sgd_log(X_log, y_log, batch_size=50, gamma=0.014, n_epochs=700_000), 
        #"SGD (batch = 1)" : sgd_batch1_log(X_log, y_log, gamma=0.0004, n_epochs=500_000)
    } 
    return (resultados_metodos_log,)


@app.cell
def _(resultados_metodos_log):
    ### Pesos y sesgos calculados con 
    resultados_metodos_log["GD (full-batch)"]["w"],resultados_metodos_log["GD (full-batch)"]["b"], 
    return


@app.cell
def _(resultados_metodos_log):
    ### Pesos y sesgos calculados con 
    resultados_metodos_log["Mini-Batch GD"]["w"],resultados_metodos_log["Mini-Batch GD"]["b"], 
    return


@app.cell
def _(logit_res):
    ## Imprimimos los resultados
    print(logit_res.summary())
    return


if __name__ == "__main__":
    app.run()
