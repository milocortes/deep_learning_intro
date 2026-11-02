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
    # Logistic Neural Network Regression
    Tomado de : https://sebastianraschka.com/faq/docs/logisticregr-neuralnet.html
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    La aplicación clásica de un modelo de regresión logística "clásico" es resolver una tarea de clasificación binaria. <sup><a href="#nota1">[1]</a></sup>.

    Aunque existen variantes de la regresión logística, el modelo estándar es un clasificador lineal. Por lo que la regresión logística es útil si estamos trabajando con conjuntos de datos donde las clases son más o menos "linealmente separables". <sup><a href="#nota2">[2]</a></sup>.

    Aún así, la regresión lineal es un modelo robusto para tareas de clasificación simple; el concurso de predicciones de la *March Madness* de este año fue ganado por dos profesores que utilizaron un modelo de regresión logística.

    ---
    <small id="nota1">**[1]** Sin embargo, podemos usar "sabores" de la regresión logística para enfrentar problemas de clasificación multiclase mediante regresiones softmax o regresiones logísticas multiclase.</small>

    <small id="nota2">**[2]** For “relatively” very small dataset sizes, I’d recommend comparing the performance of a discriminative Logistic Regression model to a related Naive Bayes classifier (a generative model) or SVMs, which may be less susceptible to noise and outlier points.</small>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Las redes neuronales están de alguna forma relacionadas con la regresión logística. Podemos pensar una regresión logística como una red neuronal con una capa:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Render an image from a URL
    mo.image(
        src="images/logistic_neural_nets.png",
        alt="Marimo logo",
        width=600,
        height=300,
        rounded=True,
        caption="Tomado de https://sebastianraschka.com/faq/docs/logisticregr-neuralnet.html",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    De hecho, es muy común usar funciones sigmoides como funciones de activación en las capas ocultas de una red neuronal profunda, como en el esquema anterior pero sin umbral.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Render an image from a URL
    mo.image(
        src="images/sigmoide.png",
        alt="Marimo logo",
        width=600,
        height=300,
        rounded=True,
        caption="Tomado de https://sebastianraschka.com/faq/docs/logisticregr-neuralnet.html",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Es aceptable utilizar la función de umbral en la capa de salida si se trata de una tarea de clasificación binaria (en este caso, solo se tendría una neurona con activación sigmoide en la capa de salida).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Una de las propiedades de la regresión logística es que su función de pérdida (entropía cruzada) es convexa, de manera que se garantiza que el mínimo local es también el mínimo global.

    Sin embargo, una vez que apilamos funciones de activación sigmoides en una red neuronal multicapa, se perderá esta convexidad.

    Si nos centramos únicamente en un único peso o coeficiente del modelo, podemos visualizar la función de pérdida en un perceptrón multicapa como una función accidentada con múltiples mínimos locales donde pueden caer los algoritmos de optimización:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Render an image from a URL
    mo.image(
        src="images/landscape_logistic_deep.png",
        alt="Marimo logo",
        width=600,
        height=300,
        rounded=True,
        caption="Tomado de https://sebastianraschka.com/faq/docs/logisticregr-neuralnet.html",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Incluso si el modelo converge a un mínimo local, en muchas ocasiones obtenemos un modelo predictivo potente.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    En resumen, se recomienda primeramente modelar un problema de clasificación usando un modelo *simple* (e.g. una regresión logística).
    En muchos casos, esto puede resolver su problema de manera suficientemente satisfactoria. Sin embargo, si no se está satisfecho con el desempeño del modelo y se tienen suficientes datos de entrenamiento, sería adecuado entrenar un modelo computacionalmente más caro como una red neuronal profunda que tiene la ventaja de aprender funciones no lineales más complejas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrenamiento de Logistic Neural Network Regression con Descenso por Gradiente
    """)
    return


@app.cell
def _():


    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrenamiento de Logistic Neural Network Regression con Adam y [Flax](https://flax.readthedocs.io/en/stable/)
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
