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
    # Mecanismos de Atención y Transformers
    Tomado de *Processing language and classifying images with attention and transformers* de Volpe, G., Midtvedt, B., Pineda, J., Moberg, H. K., & Bachimanchi, H. (2026). Deep Learning Crash Course. No Starch Press.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Atención** (*Attention*) es un mecanismo que permite a los modelos hacer predicciones al enfocarse en partes específicas de los datos de insumo, basándose en su reelevancia. Los mecanismos de atención fueron introducidos por primera vez en arquitecturas codificador-decodificador para mejorar el desempeño de tareas de traducción al permitirle a los modelos enfocarse de forma selectiva en las partes más reelevantes de la oración.

    El concepto de atención se ha vuelto fundamental para los **transformers**, una arquitectura de redes neuronales que ha transformado el tratamiento de datos secuenciales, tales como oraciones y series temporales.

    El mecanismo de **autoatención** (*self-attention*) de los Transformers les permite capturar dependencias de largo alcance, lo que conlleva un mejor rendimiento en comparación con modelos anteriores, como las redes neuronales recurrentes.}

    Desde que Google introdujo la arquitectura transformer en 2017, los transformers se han convertido en el modelo de referencia para muchas tareas de procesamiento del lenguaje, entre ellas la traducción de textos (traducción automática), la elaboración de resúmenes de documentos extensos (resumen de textos) y la determinación de si un texto tiene un tono positivo o negativo (análisis de sentimiento).

    Los transformadores también se han utilizado para predecir la estructura de las proteínas a partir de secuencias genéticas (AlphaFold), analizar imágenes (transformadores de visión, o ViTs) y vincular el lenguaje con las imágenes mediante el aprendizaje de conceptos visuales a partir de la supervisión basada en el lenguaje natural (Contrastive Language-Image Pre-training, o CLIP).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ¿Cómo trabajan los mecanismos de Atención?

    Consideremos la oración :

    <center>The teacher praised a student because she improved.</center>


    A los humanos les es sencillo obtener información semática de esta oración.

    Podemos responder inmediatamente a preguntas como:
    <center>Does improved refer to the teacher or the student?</center>

    Sin embargo, esta tarea plantea serios desafíos para un algoritmo, ya que este carece de la capacidad inherente de comprender el contexto y resolver referencias ambiguas, tal como lo hacen los seres humanos. Determinar si *improved* se refiere al profesor o al estudiante requiere analizar las relaciones y dependencias entre las palabras de la oración.


    El mecanismo de atención permite a los modelos evaluar la importancia de cada palabra en una oración en relación con otras palabras, ya sea dentro de la misma oración (autoatención-**self-attention**) o entre oraciones distintas (atención cruzada-**cross-attention**).

    En la autoatención, una oración presta atención a sí misma para captar dependencias internas, mientras que, en la atención cruzada, una oración presta atención a otra para captar relaciones entre oraciones. Por motivos de simplicidad, en el siguiente ejemplo nos centraremos en la atención cruzada.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Implementación de la atención por producto escalar

    Contruyamos un ejemplo de atención cruzada. Definamos dos oraciones:
    """)
    return


@app.cell
def _():
    query_sentence = "The teacher praised a student because she improved"
    key_sentence = "A student asked the teacher to help her improve"
    return key_sentence, query_sentence


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dado que las redes neuronales no pueden procesar directamente texto crudo, necesitamos tokenizar las oraciones y convertir cada palabra en vector de *embedding*. Un *embedding* es una representación numérica de una palabra en un espacio vectorial continuo,
    el cual capta su significado semántico y sus relaciones con otras palabras.

    Al utilizar mecanismo de atención seremos capaces de calcular *pesos* de atención que resaltan la relevancia de cada palabra en la oración clave respecto a cada palabra en la oración de consulta. Los pesos son almacenados en una *matriz de atención*
    que representa el grado en que cada palabra de la oración de consulta presta atención a cada palabra de la oración clave.

    Con la matriz de pesos actualizaremos los embeddings de la oración de consulta al calcular una suma ponderada de los embeddings de la oración clave.

    El embedding actualizado, conocido como *attention output*, proporciona una representación enriquecida de los tokens de consulta los cuales ahora incorporan información contextual reelevante de la oración clave. Este proceso le permite al modelo enfocarse en relaciones importantes entre dos oraciones.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Tokenización y construcción de embeddings

    Tokenizaremos las oraciones y convertiremos las palabras a vectores de embeddings.
    """)
    return


@app.cell
def _(key_sentence, query_sentence):
    query_tokens = query_sentence.split()
    key_tokens = key_sentence.split()
    return key_tokens, query_tokens


@app.cell
def _(query_tokens):
    query_tokens
    return


@app.cell
def _(key_tokens):
    key_tokens
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nota que ambas oraciones contienen palabras como *she* o *her* que pueden ser ambiguas sin contexto. El mecanismo de atención pude ayudar al modelo a resolver dichas ambiguedades al enfocarse en las palabras reelevantes de la oración clave.

    Ahora convertiremos cada palabra a embeddings. Cada palabra se convertirá en un vector numérico y cada oración se convertirá en una matriz, como la figura siguiente:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src="images/atencion/atencion_01.png",
        alt="Marimo logo",
        width=750,
        height=350,
        rounded=True,
        caption="Representación matricial de oraciones de consulta y llave",
    ).center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Estas matrices representan las oraciones de consulta y clave (query and key sentences) en forma de embeddings; cada fila corresponde a una palabra de la oración, y cada columna representa las características del embeddings. Formalmente esas matrices pertenecen al espacio vectorial $\R^{n \times d}$ donde $n$ es el tamaño de la secuencia y $d$ es la dimensión del embedding.


    Para obtener representaciones vectoriales (embeddings) significativas que capturen las relaciones semánticas entre las palabras, utilizaremos GloVe (Global Vectors for Word Representation), un popular modelo de representaciones vectoriales preentrenado. El uso de embeddings preentrenados, como GloVe, permite que el modelo obtenga información lingüística sin necesidad de entrenar los embeddings desde cero.

    Tenemos que descargar los embedding de GloVe con la siguiente instrucción:
    """)
    return


@app.cell
def _():
    import os
    import zipfile
    import urllib.request


    glove_folder = ".glove_cache"

    if not os.path.exists(glove_folder):
        os.makedirs(glove_folder, exist_ok=True)
        url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.42B.300d.zip"
        zip_filepath = os.path.join(glove_folder, "glove.42B.300d.zip")
        urllib.request.urlretrieve(url, zip_filepath)

        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(glove_folder)
        os.remove(zip_filepath)
    return glove_folder, os


@app.cell
def _():
    from tqdm import tqdm 
    import numpy as np
    import torch

    """
    This function loads the GloVe embeddings from the specified file.
    """
    def load_glove_embeddings(glove_file):
        """Load GloVe embeddings."""
        glove_embeddings = {}
        with open(glove_file, "r", encoding="utf-8") as file:
            for line in tqdm(file):
                values = line.split()
                word = values[0]
                glove_embeddings[word] = np.round(
                    np.asarray(values[1:], dtype="float32"), decimals=6,
                )
        return glove_embeddings

    """
    This function constructs an embeddings matrix for the given vocabulary by mapping each token to its corresponding GloVe embedding. For each token in the vocabulary, the function attempts to retrieve its embedding from the glove_embeddings dictionary. If the exact token is not found, the function tries the lowercase version. If an embedding is found, the function replaces the corresponding row in the embeddings matrix with the GloVe embedding. Tokens without a matching GloVe embedding remain as zero vectors in the matrix. The resulting embeddings matrix ensures that each token’s index in the vocabulary aligns with the same index in the embeddings matrix.
    """
    def get_glove_embeddings(vocab, glove_embeddings, embed_dim):
        """Get GloVe embeddings for a vocabulary."""
        embeddings = torch.zeros((len(vocab), embed_dim), dtype=torch.float32)
        for i, token in enumerate(vocab):
            embedding = glove_embeddings.get(token)
            if embedding is None:
                embedding = glove_embeddings.get(token.lower())
            if embedding is not None:
                embeddings[i] = torch.tensor(embedding, dtype=torch.float32)
        return embeddings

    return get_glove_embeddings, load_glove_embeddings, torch


@app.cell
def _(
    get_glove_embeddings,
    glove_folder,
    key_tokens,
    load_glove_embeddings,
    os,
    query_tokens,
):
    glove_file = os.path.join(glove_folder, "glove.42B.300d.txt")
    glove_embed, embed_dim = load_glove_embeddings(glove_file), 300

    query_embeddings = get_glove_embeddings(query_tokens, glove_embed, embed_dim)
    key_embeddings = get_glove_embeddings(key_tokens, glove_embed, embed_dim)
    return key_embeddings, query_embeddings


@app.cell
def _(query_embeddings):
    query_embeddings
    return


@app.cell
def _(key_embeddings):
    key_embeddings
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cada fila de estas matrices representa el embedding de una palabra en un espacio de 300 dimensiones. Estos embeddings transforman las  oraciones tokenizadas en formatos numéricos que el mecanismo de atención puede procesar, lo que permite al modelo capturar las relaciones semánticas entre las palabras de tu consulta y las oraciones clave.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Mecanismo de Atención

    Una forma simple de calcular la matriz de atención es con el **producto punto** entre las matrices que contienen los embeddings de las oraciones de consulta y llave. Esto implica calcular la similitud entre cada palabra de la oración de consulta y cada palabra de la oración clave, mediante el cálculo del producto punto de sus respectivos embeddings.

    Esta operación da lugar a una matriz de atención en la que cada elemento representa el grado de relevancia o similitud entre una palabra de consulta y una palabra clave. Cuanto mayor sea el valor del producto punto, más similares son las palabras en términos de sus representaciones numéricas. Esta matriz de atención cuantifica de manera efectiva en qué medida cada palabra de la oración de consulta **presta atención** (o se centra en) cada palabra de la oración clave, lo que permite al modelo priorizar la información más relevante al actualizar los embeddings de la consulta con información contextual proveniente de la oración clave.

    Después de calcular el producto punto entre los embeddings de consulta y clave, usualmente se divide la matriz de atención por la raíz cuadrada de la dimensión del embedding ($d$) para ayudar a la estabilización del gradiente cuando se usan vectores de dimensiones altas. Esta división calcula el **producto punto escalado**. Finalmente, es necesario normalizar los scores para convertirlaos en pesos mediante la función softmax.

    El proceso completo se ilustra en la siguiente imagen:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src="images/atencion/atencion_02.png",
        alt="Marimo logo",
        width=800,
        height=450,
        rounded=True,
        caption="Cálculo de la matriz de atención y salida",
    ).center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Para calcular la salida de la atención (*attention output*) necesitamos multiplicar la matriz de atención por el valor de los embeddings dando lugar a una representación actualizada de la oración de consulta. En este caso, también se usarán los embeddings de clave como embeddings de valor, de modo que los valores sean iguales a las claves. Esta operación agrega de manera efectiva los embeddings de valor, ponderados por las scores de atención, para generar nuevos embeddings para las palabras de consulta que incorporan información relevante proveniente de la oración clave.

    La salida de la atención proporciona embeddings enriquecidos de los tokens de consulta que reflejan las relaciones contextuales capturadas por el mecanismo de atención.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Implementamos la atención de producto punto con la clase `DotProductAttention`:
    """)
    return


@app.cell
def _(torch):
    import deeplay as dl

    class DotProductAttention(dl.DeeplayModule):
        """Dot-product attention."""

        def __init__(self):
            """Initialize dot-product attention."""
            super().__init__()

        def forward(self, queries, keys, values):
            """Calculate dot-product attention."""
            attn_scores = (torch.matmul(queries, keys.transpose(-2, -1))
                           / (keys.size(-1) ** 0.5))
            attn_matrix = torch.nn.functional.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_matrix, values)
            return attn_output, attn_matrix

    return DotProductAttention, dl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    El método `forward()` calcula los scores de atención (`attn_scores`) al calcular el producto punto escalado entre las consultas y las llaves transpuestas. El método normaliza los scores de atención con la función softmax para obtener la matriz de atención (`attn_matrix`), que representa la reelevancia normalizada de cada llave a cada consulta. Por último, el método calcula la salida de atención (`attn_output`) al multiplicar la matriz de atención por los valores de los embeddings, dando lugar embeddings de consulta actualizados que incorporan información de la oración clave.

    Finalmente calculamos la matriz de atención y su salida:
    """)
    return


@app.cell
def _(DotProductAttention, key_embeddings, query_embeddings):
    attention = DotProductAttention() 
    attn_output, attn_matrix = attention(
        queries=query_embeddings, keys=key_embeddings, values=key_embeddings,
    )
    return (attn_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Con la matriz de atención y la salida de atención hemos implementado completamente el mecanismo de atención de producto punto, permitiéndole a nuestro modelo efectivamente capturar y utilizar relaciones contextuales entre las oraciones de consulta y llave.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Visualización de la atención
    """)
    return


@app.cell
def _():
    from matplotlib import pyplot as plt 
    from matplotlib.ticker import FixedLocator

    def plot_attention(query_tokens, key_tokens, attn_matrix):
        """Plot attention."""
        fig, ax = plt.subplots()
        cax = ax.matshow(attn_matrix, cmap="Greens")
        fig.colorbar(cax)
        ax.xaxis.set_major_locator(FixedLocator(range(len(key_tokens))))
        ax.yaxis.set_major_locator(FixedLocator(range(len(query_tokens))))
        ax.set_xticklabels(key_tokens, rotation=90)
        ax.set_yticklabels(query_tokens)
        plt.show()

    return (plot_attention,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Esta función genera un mapa de calor que ofrece información sobre cuánto énfasis pone el modelo en cada palabra de la oración clave al procesar la oración de consulta.
    """)
    return


@app.cell
def _(attn_matrix, key_tokens, plot_attention, query_tokens):
    plot_attention(query_tokens, key_tokens, attn_matrix.detach().squeeze())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Las filas representan los tokens de la oración de consulta, mientras que las columnas representan los tokens de la oración clave. El mapa de calor muestra los pesos de atención; las zonas más oscuras indican scores de atención más altos.

    Lo que resulta especialmente interesante en este ejemplo de atención cruzada es la forma en que el modelo destaca palabras que están relacionadas semánticamente, aunque no sean idénticas. Por ejemplo, existe una relación de atención entre «she» (en la consulta) y «her» (en la oración clave). Esto sugiere que el mecanismo de atención es capaz de reconocer que dichas palabras están relacionadas, a pesar de no ser exactamente iguales. Esto pone de manifiesto la eficacia de la atención cruzada al utilizar representaciones vectoriales («embeddings») preentrenadas, como GloVe, ya que palabras relacionadas semánticamente —que no tienen por qué coincidir de forma exacta— pueden, aun así, compartir atención.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Es importante señalar que, en este ejemplo, no existen pesos que puedan aprenderse, ni en las codificaciones de las palabras ni en el propio mecanismo de atención. Por tanto, los resultados reflejan simplemente las relaciones básicas entre los vectores de inmersión (*embeddings*). Estos vectores han sido preentrenados en un corpus extenso y capturan similitudes semánticas entre palabras basándose en sus estadísticas de coocurrencia dentro de dicho corpus. En consecuencia, el mecanismo de atención utiliza aquí estas relaciones fijas, asignando pesos de atención más elevados a aquellas palabras que están semánticamente relacionadas o que aparecen juntas con frecuencia (como «she» y «her»). No obstante, dado que tanto los vectores de inmersión como el mecanismo de atención no han sido ajustados específicamente para la tarea o el conjunto de datos en cuestión, el rendimiento del modelo se ve limitado por las relaciones generales codificadas en GloVe, en lugar de estar adaptado para optimizar el resultado en este caso concreto (por ejemplo, generando un peso de atención bajo entre «her» y «student»).

    Si bien esto resulta razonable para tareas de coincidencia exacta, la verdadera fortaleza del mecanismo de atención reside en modelos capaces de aprender a asignar mayores pesos de atención a aquellas palabras que, según el contexto, resultan relevantes para la consulta. En tales situaciones, la atención ayuda al modelo a centrarse en la información más pertinente, mejorando así su capacidad para comprender y procesar datos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Hagamos entrenable el mecanismo de atención

    La atención de producto punto que acabamos de estudiar es una forma de **atención multiplicativa** : las consultas y llaves son multiplicadas directamente. Formas más complejas de atención multiplicativa pueden incluir **pesos entrenables** o transformaciones que permiten al modelo ajustar su atención dinamicamente basado en en los datos de insumo, ofreciendo así más flexibilidad al capturar información reelevante de diferentes partes de la secuencia.

    Un ejemplo notable es la **atención de producto punto escalada estándar** en la que las consultas, llaves y valores son transformadas al aplicar una transformación lineal con los pesos aprendidos. Esto le permite al modelo aprender interacciones más sofisticadas entre las consultas y las llaves al modificar su representación, como se muestra en las figuras:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src="images/atencion/atencion_03.png",
        alt="Marimo logo",
        width=750,
        height=350,
        rounded=True,
        caption="Atención entrenable de producto punto",
    ).center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La transformación lineal entrenable que genera las matrices de consulta, llave y valor, en esta forma de atención, permiten al modelo ser entrenado al optimizar el mecanismo de atención para una tarea específica.

    Esta forma de atención es implementada por la clase  `TrainableAttention`
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    En este caso, las matrices de pesos (`Wq`, `Wk`, `Wv`) son inicializadas aleatoriamente y el proceso de entrenamiento determinará los valores finales.

    En el método `forward()` las matrices de insumo de las consultas, llaves y valores son primeramente transformadas al aplicarles las transformaciones lineales entrenables `Wq`, `Wk` y `Wv`. Estas transformaciones ajustan las representaciones originales de las consultas, llaves y valores, permitiendo al modelo aprender representaciones específicas a la tarea.

    El parámetro `num_in_features` representa la dimensionalidad del embedding de entrada (tipicamente el tamaño del embedding de palabras preentrenado) mientras que `num_out_features` refiere a la dimensionalidad de las matrices transformadas de consulta, llaves y valores. Al ajustar el parámetro `num_out_features` podemos controlar el tamaño del espacio latente donde operan los mecanismos de atención, incrementando o disminuyendo con ello la capacidad del modelo de capturar relaciones complejas.

    A continuación, se calculan las puntuaciones de atención mediante el producto escalar escalado entre las matrices de consulta y de clave transformadas. Al igual que en el caso anterior, dichas puntuaciones se normalizan utilizando la función softmax para obtener los pesos de atención, los cuales representan la importancia relativa de cada clave con respecto a la consulta. Finalmente, la salida de la atención se calcula aplicando estos pesos a la matriz de valores.

    Este enfoque aporta una flexibilidad considerable, ya que el modelo puede aprender la forma óptima de representar y relacionar las consultas, las claves y los valores para la tarea en cuestión. Por ejemplo, en una tarea de traducción automática, el modelo puede aprender que ciertas transformaciones de la consulta (como poner mayor énfasis en estructuras sintácticas específicas) mejoran la precisión de la traducción, o que ciertas representaciones de las claves deben recibir menos peso en favor de otras.

    Esta atención basada en el producto escalar, que es entrenable, constituye la columna vertebral del mecanismo de **autoatención (self-attention)** en los modelos **transformer** modernos, ya que permite al modelo centrarse de forma dinámica en distintas partes de la secuencia de entrada, tanto durante el entrenamiento como durante la inferencia.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Autoatención (self-attention) y Transformers

    El poder de la autoatención para capturar dependencias de largo plazo entre palabras o tokens en una oración es completamente explotada por los Transformers. Los **Transformers**, al contrario que los modelos de Redes Neuronales Recurrentes (RNN), no dependen del procesamiento secuencial para modelar las dependencias dentro de una oración 😲. En su lugar, utilizan un mecanismo de atención **multicabezal** para enfocarse simultaneamente en diferentes partes de la oración logrando así procesar la oración en paralelo. Esto hace a los transformers altamente eficientes y efectivos para tareas de NLP.

    Esta atención en paralelo permite a las arquitecturas de transformer amplificar los tokens importantes —como palabras clave que expresan sentimiento—, al tiempo que reduce la influencia de aquellos menos relevantes. Esta atención de ajuste dinámico entre las secuencias de tokens de la oraciones mejora la comprensión y la representación del texto por parte del modelo, especialmente en tareas complejas como el análisis de sentimientos, en las que el sentimiento puede estar vinculado a frases o palabras específicas dentro de una oración más larga. Adicionalmente, las capacidades de los tranformers de procesamiento paralelo los hacen más escalables, permitiéndoles textos largos más eficientemente y con mejor desempeño que modelos basados en redes recurrentes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Atención multicabezal

    La capacidad de los transformers de procesar los datos de insumo de forma paralela es posible por la **atención multicabezal**, un mecanismo que mejora la expresividad de las capas de atención sin aumentar significativamente el número de parámetros. La estructura de la atención multicabezal se muestra en la siguiente figura:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src="images/atencion/atencion_04.png",
        alt="Marimo logo",
        width=750,
        height=350,
        rounded=True,
        caption="Atención multicabezal",
    ).center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La atención multicabezal emplea un enfoque paralelismo *divide y vencerás* al distribuir los procesos de atención en múltiples *cabezales de atención*. Cada cabezal atiende independientemente a diferentes partes de la secuencia de insumos, extrayendo distintas características. Esas salidas individuales son posteriormente concatenadas e integradas a una transformación lineal para generar la salida de atención. Este enfoque multifacético permite al modelo obtener una comprensión más profunda y completa de la secuencia de entrada.

    Este proceso suele compararse con un equipo de expertos que analiza múltiples facetas de un problema complejo y combina sus perspectivas para llegar a una conclusión. De manera análoga, el mecanismo de atención de múltiples cabezales (*multi-head attention*) permite que cada cabezal se centre en distintas partes de la entrada, lo que confiere al modelo una mayor solidez en su interpretación.

    La clase `MultiHeadAttentionLayer` nos permite calcular la autoatención multicabezal:
    """)
    return


@app.cell
def _(dl, torch):

    class MultiHeadAttentionLayer(dl.DeeplayModule):
        """"Multi-head attention layer."""

        def __init__(self, num_features, num_heads):
            """Initialize multi-head attention."""
            super().__init__()
            self.num_features, self.num_heads = num_features, num_heads
            self.head_dim = num_features // num_heads  # Must be integer

            self.Wq = dl.Layer(torch.nn.Linear, num_features, num_features)
            self.Wk = dl.Layer(torch.nn.Linear, num_features, num_features)
            self.Wv = dl.Layer(torch.nn.Linear, num_features, num_features)
            self.Wout = dl.Layer(torch.nn.Linear, num_features, num_features)

        def forward(self, in_sequence):
            """Apply the multi-head attention mechanism to the input sequence."""
            seq_len, embed_dim = in_sequence.shape
            Q = self.Wq(in_sequence)
            Q = Q.view(seq_len, self.num_heads, self.head_dim).permute(1, 0, 2)
            K = self.Wk(in_sequence)
            K = K.view(seq_len, self.num_heads, self.head_dim).permute(1, 0, 2)
            V = self.Wv(in_sequence)
            V = V.view(seq_len, self.num_heads, self.head_dim).permute(1, 0, 2)

            attn_scores = (torch.matmul(Q, K.transpose(-2, -1)) 
                           / (self.head_dim ** 0.5))
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)
            attn_output = attn_output.permute(1, 0, 2).contiguous()
            attn_output = attn_output.view(seq_len, self.num_features)

            return self.Wout(attn_output)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La implementación es una variación avanzada de la atención por producto punto escalado estandar. La clase toma como argumentos `num_features`, que define la dimensionalidad de la entrada, y `num_heads`, que define el número de cabezales de atención (`num_features` debe ser divisible por `num_heads`, dado que cada cabeza de atención procesa un subconjunto de las características).

    En el método `forward()`, la secuencia de entrada es transformada usando capas lineales (`Wq`, `Wk` y `Wv`). Después que se calculan las matrices de consulta, llave y valor, la rutina *amolda* (reshape) los tensores para ajustarlos a las múltiples cabezales de atención mediante el método `view()`, que divide la entrada en `num_head` partes, cada una con dimensiones iguales a `head_dim`. El método `permute(1, 0, 2)` reordena las dimensiones del tensor al intercambiar la longitud de la secuencia y el número de cabezales, permitiéndole a cada cabezal procesar su porción de la secuencia de forma independiente y en paralelo.

    Posteriormente se calculan los productos punto y los pesos de atención tal como en el mecanismo de atención por producto punto escalado estándar. Tras el procesamiento, las salidas de cada cabezal se concatenan de nuevo y se reordenan en el tamaño y dimensiones originales de la secuencia. Se aplica finalmente una transformación lineal via `Wout` para combinar las salidas de todos los cabezales, produciendo así el resultado final.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estructura del Transformer

    Los transformers consisten en una apilación de capas, cada una implementando un mecanismo de atención multicabezal seguido por una red neuronal feed-forward (red neuronal completamente conectada). Adicionalmente estas capas son aumentadas con capas de normalización y conexión a residuales para asegurar la estabilidad del gradiente durante el entrenamiento.

    La arquitectura clásica de un Transformer consiste en un modelo codificador-decodificador (encoder-decoder). Similar a la arquitectura seq2seq, el codificador procesa la secuencia de insumo para generar representaciones contextuales, mientras que el decodificador usa esas representaciones para producir la secuencia de salida; generar esta salida puede usarse para traducir la entrada a otro idioma o transformarla en una forma distinta. El decodificador replica la estructura del codificador, pero incluye una capa de atención cruzada adicional que le permite centrarse en partes relevantes de la entrada codificada mientras genera la secuencia de salida.

    Sin embargo, la arquitectura de Transformer es muy flexible : más allá del modelo clásico encoder-decoder, el modelo puede ser diseñado para variantes como *solo codificación* o *solo decodificación*. Estas variantes sirven para distintos objetivos de aprendizaje dependiendo de las características de la tarea a resolver. Por ejemplo, los Transformer *solo-codificación* son adecuados para tareas como la clasificación, dado que se centran en comprender y representar la secuencia de entrada. Por su parte, los transformer *solo-decodificación* son empleados en tareas que implican generación de texto, esto es, para tareas generativas.
    """)
    return


if __name__ == "__main__":
    app.run()
