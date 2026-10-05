import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cross-Entropy Loss
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    En Machine Learning se necesita de medidas matemáticas que puedan comparar la distribución de probabilidad generala por un modelo contra la distribución de probabilidad observada en los datos. Estas medidas proporcionan un valor escalar que indica qué tan bien se ajustan nuestras predicciones con los valores observados.

    La funnción de pérdida **entropía cruzada** (*cross-entropy*) nos proporciona un principio matemático de esta medida y nos proporciona una métrica que nos dice qué tan bien nuestro modelo entiende los patrones estadísticos observados en los datos.

    La entropía cruzada nos proporciona una manera de medir en qué medida la distribución predicha por el modelo difiere de las distribuciones reales que observamos en los datos de entrenamiento.

    Una característica importante de la entropía cruzada es su conexión natural con la estimación por máxima verosimilitud. Cuando minimizamos la entropía cruzada en un conjunto de datos, simultaneamente maximizamos la probabilidad que el modelo asigne las salidas de los datos de entrenamiento a la verdadera distribución de probabilidad. Esta propiedad es el fundamento teórico para responder a la pregunta de cómo medimos la calidad de un modelo probabilístico.

    Dado que la entropía cruzada se basa en conceptos de teoría de la información, abordaremos el concepto de **entropía**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Teoría de la Información :  Entropía

    La Entropía es una medida de *incertidumbre*. También podemos pensarla como una medida de la información contenida en un evento aleatorio. El concepto de entropía fue propuesto por Claude Shannon y tiene sus orígenes en la termodinámica y en la mecánica estadística. Shannon propuso el concepto en su tesis de maestría en 1948. Para Shannon, la información y la incertidumbre son lados de la misma moneda.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    shannon = mo.image(
        src="images/shannon.png",
        #alt="Logo de Marimo",
        width=450,
        height=550,
        rounded=True,
        #caption="Logo oficial",
    )
    shannon_1948 = mo.image(
        src="images/shannon_1948.png",
        #alt="Logo de Marimo",
        width=450,
        height=550,
        rounded=True,
        #caption="Logo oficial",
    )

    mo.hstack([shannon, shannon_1948], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entropía

    La entropía mide la incertidumbre inherente en una distribución de probabilidad. Responde a la pregunta : en promedio, ¿Qué tan sorprendidos estaremos cuando observemos un resultado de esta distribución? Para una distribución de probabilidad discreta $P$ sobre $x$, la entropía se define como :

    \begin{equation}
    H(P)=-\sum_x P(x) \log P(x)
    \end{equation}

    donde :
    - $H(P)$ es la entropía de la distribución de probabilidad $P$, medida en bits (logaritmo base 2).
    - $x$ es la posible realización del conjunto de eventos.
    - $P(x)$ es la probabilidad que ocurra el evento $x$.
    - $\log$ es la función logaritmo, típicamente en base 2 o logaritmo natural.

    El logaritmo asegura que eventos raros contribuyan más a la incertidumbre total que eventos comunes. Esta propiedad matemática se alinea con nuestra intuición : cuando ocurre un evento improbable, experimentamos más sorpresa y recibimos más información que cuando ocurre un evento común.

    Cuando se tiene plena certeza que ocurrirá un evento, $P(x) = 1$, su contribución es cero dado que no ganamos más información de observar algo que sabemos que ocurrirá. En el caso contrario, $P(x)=0$ indica que eventos que nunca ocurrirán no contribuyen a la entropía.

    La entropía representa el contenido de información esperada o "sorpresa" de observar una variable aleatoria. Para una distribución uniforme sobre $N$ posibles realizaciones, se tiene la máxima entropía $\log N$, mientras que para una distribución determinística tenemos cero entropía. Esto significa que cuando todas las realizaciones son equi-probables, estamos maximizando la incertidumbre acerca de lo que ocurrirá en el futuro, miestras que un evento que se tiene garantizada su ocurrencia, tenemos perfecta certeza y la incertidumbre es cero.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ejemplo :  Dado justo y dado cargado
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Dado justo

    - Un dado justo produce las salidas : 1, 2, 3, 4, 5 y 6.
    - La probabilidad de cada una de las salidas es : $\dfrac{1}{6}$, $\dfrac{1}{6}$, $\dfrac{1}{6}$, $\dfrac{1}{6}$, $\dfrac{1}{6}$, $\dfrac{1}{6}$.
    - La información asociada a cada salida es $Q=-\log _2 \frac{1}{6}=\log _2 6$.
    - La entropía de Shannon del sistema es:

    \begin{equation}
    H=\sum_{i=1}^6 \frac{1}{6} \log _2 6=2.58
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Dado cargado

    - Un dado cargado produce las salida : 1, 2, 3, 4, 5 y 6.
    - La probabilidad de cada una de las salidas es : $\dfrac{1}{10}$, $\dfrac{1}{10}$, $\dfrac{1}{10}$, $\dfrac{1}{10}$, $\dfrac{1}{10}$ y $\dfrac{1}{2}$,  respectivamente.
    - La información asociada a cada salida es :

    \begin{equation}
    \log _2 10, \log _2 10, \log _2 10, \log _2 10, \log _2 10, \log _2 2
    \end{equation}

    - La entropía de Shannon del sistema es :

    \begin{equation}
    H=5 \times \frac{1}{10} \log _2 10+\frac{1}{2} \log _2 2=\log _2 \sqrt[2]{20}=1.16
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ejemplo : Tirada de una moneda
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Para el caso de una tirada de moneda, donde la realización de una cara tiene probabilidad $P_1 = P$ y la probabilidad de la otra moneda es $P_2 = 1 - P$.

    La entropía del sistema es :

    \begin{equation}
    H(P)=-\sum P_i \log _2 P_i=-P \log _2 P-(1-P) \log _2(1-P)
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src="images/entropia_binaria.png",
        #alt="Logo de Marimo",
        width=350,
        height=350,
        rounded=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - La entropía se maximiza cuando $P= \dfrac{1}{2}$. Mucha incertidumbre sobre la salida, mayor información o sorpresa ganada.
    - La entropía mínima cuando $P=1$ o $P=0$. Poca incertidumbre acerca de la salida, menor información o sorpresa ganada.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Divergencia Kullback-Leibler

    Mientras que la entropía es una medida de incertidumbre para una distribución de probabilidad,  la Divergencia Kullback-Leibler (KL) contabiliza en qué medida una distribución de probabilidad $Q$ diverge de una distribución de referencia $P$. KL cuantifica la ineficiencia de asumir que la distribución is $Q$ cuando la verdadera distribución es $P$ :

    \begin{equation}
    D_{K L}(P \| Q)=\sum_x P(x) \log \frac{P(x)}{Q(x)}
    \end{equation}

    donde :

    - D_{K L}(P \| Q) es la divergencia KL de la distribución $Q$ a la distribución de referencia $P$.
    - $P(x)$ es la probabilidad de la realización $x$ bajo la distribución $P$.
    - $Q(x)$ es la probabilidad de la realización $x$ bajo la distribución $A$
    - $x$ es una realización de $P$.

    La divergencia KL es siempre no negativa e igual a cero sí y solo si $P=Q$. La divergencia mide la penalidad que pagamos de usar un mal modelo: si usamos un mecanismo de compresión basado en nuestra distribución predicha $Q$, pero en realidad se sigue la distribución $P$, gastaremos una cantidad de bits adicional proporcional a la divergencia.

    La divergencia tiene una asimetría que importa en machine learning : Esta requiere conocer la verdadera distribución $P$ sobre todas las realizaciones. En la práctica sólo obervamos realizaciones específicas de $P$, no conocemos la distribución completa. Esta limitación nos previene de calcular directamente la divergencia KL durantge el entrenamiento, por lo que necesitamos usar la **entropía cruzada** en su lugar.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Definición de la Entropía Cruzada
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La entropía cruzada combina la entropía de la distribución verdadera y la divergencia KL entre las distribuciones verdadera y predicha por el modelo en una sola cantidad que podemos cuantificar de los datos. Para las distribuciones $P$ (verdadera) y $Q$ (predicción del modelo), la entropía cruzada es :

    \begin{equation}
    H(P, Q)=-\sum_x P(x) \log Q(x)
    \end{equation}

    donde :
    - $H(P, Q)$ es la entropía cruzada entre la distribución verdadera $P$ y la distribución predica $Q$.
    - $P(x)$ es la probabilidad de la realización $x$ bajo la distribución verdadera (distribución empírica de los datos).
    - $Q(x)$ la probabilidad de la realización $x$ asignada por el modelo.
    - $x$ es la realización del conjunto de eventos posibles.

    Cuando $P$ representa la distribución empírica de los datos de entrenamiento, $P(x)$ es 1 para la etiqueta o clase observada y 0 para el resto de etiquetas o clases. Esto simplifica dramáticamente la entropía cruizada dado que la suma se reduce a un sólo término que involucra sólo a la verdadera etiqueta:

    \begin{equation}
    \begin{aligned}
    H(P, Q) & =-\sum_x P(x) \log Q(x) \\
    & =-\log Q\left(x_{\text {true }}\right)
    \end{aligned}
    \end{equation}


    donde :
    - $x_{\text {true }}$ es la etiqueta observada ( para la cual $P(x) = 1$).
    - $Q\left(x_{\text {true }}\right)$ es la probabilidad asignada por el modelo a la verdadera etiqueta.

    La entropía cruzada es el negativo de la log verosimilitud (log-likelihood) de la verdadera etiqueta bajo la distribución predicha por el modelo. Penaliza al modelo en proporción a cuánto sopresa es la probabilidad estimada versus la observación real: si el modelo asigna una probabilidad alta a la etiqueta correcta, la pérdida es pequeña; si asigna una probabilidad baja, la pérdida es grande, acercándose al infinito a medida que la probabilidad predicha se aproxima a cero.
    """)
    return


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt 
    import numpy as np 

    predicted_probability = np.linspace(0.1, 1, 100)
    bits = -1*np.log2(predicted_probability)
    nats = -1*np.log(predicted_probability)

    plt.plot(predicted_probability, nats, label = "Nat (natural log 2)" )
    plt.plot(predicted_probability, bits, label = "Bits (log base 2)" )
    plt.axhline(y=0.5, color='red', linestyle='--', label='Nat loss at p = 0.5 (ln = 2)')

    plt.title("Loss vs Predicted Probability")
    plt.ylabel("Cross-Entropy Loss")
    plt.xlabel("Predicted Probability of True Class")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    >Cross-entropy loss as a function of predicted probability for the true class. The loss decreases rapidly as model confidence increases, approaching zero for perfect prediction, and rises steeply to infinity as the predicted probability approaches zero. The steep left tail creates strong gradient signals that push the model to avoid confidently wrong predictions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Descomposición de la entropía cruzada

    La entropía cruzada se descompone en la suma de la entropía y la divergencia KL:

    \begin{equation}
    H(P, Q)=H(P)+D_{K L}(P \| Q)
    \end{equation}

    donde:

    - $H(P, Q)$ entropía cruzada entre la distribución verdadera $P$ y la distribución predicha $Q$.
    - $H(P)$ la entropía de la distribución verdadera (constante con respecto a los parámetros del modelo).
    - $D_{K L}(P \| Q)$ la divergencia KL de la distribución del modelo a la verdadera distribución.

    Dado que $H(P)$ es constante con respecto a los parámetros del modelo, minimizar la entropía cruzada es equivalente a minimizar la divergencia KL. Esta descomposición revela que la entropía cruzada captura tanto la incertidumbre inherente en los datos (el término de entropía) y la penalidad adicional de usar un modelo subóptimo (el término de la divergencia KL). Durante la optimización, el término de entropía actúa como constante, de manera que el descenso por gradiente en la entropía cruzada efectivamente minimiza la divergencia KL entre nuestro modelo y los datos de entrenamiento.

    La entropía cruzada más baja alcanzable en cualquier conjunto de datos es igual a la entropía verdadera de los datos, que representa la incertidumbre irreducible de los datos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entropía cruzada como estimación por máxima verosimilitud

    La conexión entre la entropía cruzada y la estimación por máxima verosimilitud (maximum likelihood estimation-MLE) explica los principios estadísticos para usar la entropía cruzada para entrenar modelos probabilísticos de lenguaje. Dado un conjunto de $N$ tokens observados $\{
    x_1, x_2, \dots, x_N\}$, MLE busca los parámetros del modelo $\theta$ que maximizan la probabilidad de los datos observados :

    \begin{equation}
    \widehat{\theta}=\arg \max _\theta \prod_{i=1}^N P\left(x_i \mid \text { context }_i ; \theta\right)
    \end{equation}

    donde:
    - $\widehat{\theta}$ es la estimación por máxima verosimilitud de los parámetros del modelo.
    - $P\left(x_i \mid \text { context }_i ; \theta\right)$ es la probabilidad calculada por el modelo para el $i$-ésimo token observado dado su contexto.
    - $\prod$ el producto total para todos los tokens en el conjunto de datos.

    Dado que los productos de muchas probabilidades pequeñas sufren rápidamente un subdesbordamiento numérico hacia cero, tomamos el logaritmo de ambos lados. Esto transforma el producto en una suma, la cual es numéricamente más estable:

    \begin{equation}
    \begin{aligned}
    \widehat{\theta} & =\arg \max _\theta \sum_{i=1}^N \log P\left(x_i \mid \text { context }_i ; \theta\right) \\
    & =\arg \min _\theta-\frac{1}{N} \sum_{i=1}^N \log P\left(x_i \mid \text { context }_i ; \theta\right)
    \end{aligned}
    \end{equation}

    donde :
    - $\sum_{i=1}^N \log P\left(x_i \mid\right.$ context $\left._i ; \theta\right)$ es la log verosimilitud de los tokens observados.
    - $-\frac{1}{N} \sum_{i=1}^N \log P\left(x_i \mid\right.$ context $\left._i ; \theta\right)$ es el negativo de la log verosimilitud, normalizado por el tamaño del conjunto de entrenamiento.

    La última expresión es exactamente la pérdida por entropía cruzada promediada sobre el conjunto de entrenamiento.

    Minimizar la entropía cruzada es matematicamente equivalente que realizar la estimación por máxima verosimilitud. Esto significa que tenemos una justificación estadística de usar la entropía cruzada : en la medida que $N$ crece, nuestro modelo converge a la distribución subyacente verdadera de los datos.

    Cualquier otra función de pérdida no tendría esta garantía de convergencia.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entropía cruzada binaria

    Para un problema de clasificación binaria, la pérdida de entropía cruzada es :

    \begin{equation}
    \mathcal{L}=-[y \log (\widehat{y})+(1-y) \log (1-\widehat{y})]
    \end{equation}

    donde:
    - \mathcal{L} es la pérdida de la entropía cruzada binaria.
    - $y$ es la etiqueta binaria verdadera (0 o 1)
    - $\hat{y}$ es la probabilidad predicha del modelo que la etiqueta es 1.
    - $1-y$ es la probabilidad que la etiqueta sea 0.
    - $1 - \hat{y}$ es la probabilidad predicha por el modelo que la etiqueta es 0.
    """)
    return


if __name__ == "__main__":
    app.run()
