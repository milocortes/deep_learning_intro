import marimo

__generated_with = "0.25.2"
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
    import jax 
    import jax.numpy as jnp 
    import numpy as np 


    return jax, jnp, np


@app.cell
def _(jnp, np):
    def logistic(z):
        return 1 / (1 + jnp.exp(-z))

    def predict(weights, bias, predictors):
        return logistic(jnp.dot(predictors, weights) + bias)

    def loss(weights, bias, predictors, targets):
        preds = predict(weights, bias, predictors)
        term_1 = targets * jnp.log(preds + 1e-15)
        term_2 = (1 - targets) * jnp.log(1 - preds + 1e-15)
        return -jnp.mean(term_1 + term_2)

    def generate_data(num_samples = 100_000, seed = 42):
        rng = np.random.default_rng(seed)
        predictors = rng.standard_normal((num_samples, 2))
        true_weights = np.array([1.5, -0.8])
        true_bias = 0.3
        p = predict(true_weights, true_bias, predictors)
        y = rng.binomial(1, p)
        return predictors, y, true_weights, true_bias

    return generate_data, loss


@app.cell
def _(generate_data, jax, loss):
    grad_fn = jax.grad(loss, argnums=(0, 1))

    @jax.jit
    def update_step(weights, bias, predictors, targets, learning_rate):
        dw, db = grad_fn(weights, bias, predictors, targets)
        new_weights = weights - learning_rate * dw
        new_bias = bias - learning_rate * db
        return new_weights, new_bias

    def train_model(X, y, key):
        w = jax.random.normal(key, shape=(2,))
        b = 0.0

        learning_rate = 0.1
        for i in range(2001):
            w, b = update_step(w, b, X, y, learning_rate)
            if i % 200 == 0:
                current_loss = loss(w, b, X, y)
                print(f"Epoch {i}: Loss = {current_loss:.4f}")

        print(f"Learned Weights: {w}")
        print(f"Learned Bias: {b:.4f}")

    X, y, true_w, true_b = generate_data()
    key = jax.random.PRNGKey(0)
    train_model(X, y, key)
    print(f"\nTrue Weights: {true_w}")
    print(f"True Bias: {true_b}")
    return X, key, true_b, true_w, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrenamiento de Logistic Neural Network Regression con Adam y [Flax](https://flax.readthedocs.io/en/stable/)
    """)
    return


@app.cell
def _(X, jax, jnp, key, true_b, true_w, y):
    import flax.linen as nn
    from flax.training import train_state
    import optax

    class LogisticRegression(nn.Module):
        @nn.compact
        def __call__(self, x):
            # A simple linear layer with 1 output (the logit)
            return nn.Dense(features=1, use_bias=True)(x)

    def loss_fn(params, apply_fn, X, y):
        logits = apply_fn({'params': params}, X).squeeze()
        return optax.sigmoid_binary_cross_entropy(logits, y).mean()

    @jax.jit
    def train_step_adam(state, X, y):
        loss_val, grads = jax.value_and_grad(loss_fn)(state.params, state.apply_fn, X, y)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        return state.replace(params=new_params, opt_state=new_opt_state), loss_val


    def train_model_flax(X, y, key):
        model = LogisticRegression()
        variables = model.init(key, jnp.ones((1, 2)))
        tx_adam = optax.adam(learning_rate=0.1)
        state_adam = train_state.TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=tx_adam
        )
        for i in range(301):
            state_adam, loss = train_step_adam(state_adam, X, y)
            if i % 50 == 0:
                print(f"Epoch {i}: Loss = {loss:.4f}")

        print("Learned Weights: ")
        print(state_adam.params)

    train_model_flax(X, y, key)
    print(f"\nTrue Weights: {true_w}")
    print(f"True Bias: {true_b}")
    return


if __name__ == "__main__":
    app.run()
