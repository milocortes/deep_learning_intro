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
    # Cálculo de medidas de complejidad con el paquete [ecomplexity](https://github.com/harvard-growth-lab/py-ecomplexity)
    """)
    return


@app.cell
def _():
    import pandas as pd 
    import polars as pl
    import altair as alt
    from ecomplexity import ecomplexity
    from ecomplexity import proximity

    return alt, ecomplexity, pd, pl


@app.cell
def _(pd):
    ## Cargamos datos
    datos = pd.read_parquet("datos/complejidad/2023_SAIC_Exporta_202698_037586.parquet")
    ### Limpamos datos 
    #### Al eliminar nan nos quedamos solo con información del municipio
    datos = datos[~datos["Municipio"].isna()]
    datos
    return (datos,)


@app.cell
def _(datos):
    #### Generamos dos dataframes : 1) para clases y 2) ramas
    ramas = datos[datos["Actividad económica"].apply(lambda x : x.startswith("Rama"))]
    clases = datos[datos["Actividad económica"].apply(lambda x : x.startswith("Clase"))]
    ramas
    return clases, ramas


@app.cell
def _(clases, ramas):
    #### Calculamos las medidas de complejidad usando la variable de Unidades Económicas
    ##### Convertimos a tipo entero la columna UE Unidades económicas
    intensidad = "UE Unidades económicas"
    #intensidad = "A111A Producción bruta total (millones de pesos)"
    if intensidad == "UE Unidades económicas":
        ramas[intensidad] = ramas[intensidad].astype(int)
        clases[intensidad] = clases[intensidad].astype(int)
    elif intensidad == "A111A Producción bruta total (millones de pesos)":
        ramas[intensidad] = ramas[intensidad].astype(float)
        clases[intensidad] = clases[intensidad].astype(float)
    return (intensidad,)


@app.cell
def _():
    ### Definimos las variables de tiempo, ubicación, actividad y valor para el cálculo de las medidas de complejidad
    trade_cols = {'time':'Año Censal', 'loc':'Municipio', 'prod':'Actividad económica', 'val':'UE Unidades económicas'}
    return (trade_cols,)


@app.cell
def _(ecomplexity, intensidad, ramas, trade_cols):
    ### Calculamos complejidad para UE de las ramas
    if intensidad == "UE Unidades económicas":
        cdata_ramas = ecomplexity(ramas, trade_cols)

    elif intensidad == "A111A Producción bruta total (millones de pesos)":
        cdata_ramas = ecomplexity(ramas[ramas[intensidad]>0], trade_cols)

    cdata_ramas
    return (cdata_ramas,)


@app.cell
def _(clases, ecomplexity, intensidad, trade_cols):
    ### Calculamos complejidad para UE de las clases
    if intensidad == "UE Unidades económicas":
        cdata_clases = ecomplexity(clases, trade_cols)
    elif intensidad == "A111A Producción bruta total (millones de pesos)":
        cdata_clases = ecomplexity(clases[clases[intensidad]>0], trade_cols)
    cdata_clases
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cuales son los municipios más complejos en cada dataset?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ramas
    """)
    return


@app.cell
def _(cdata_ramas, pl):
    ## Veamos el top 10 de municipios más complejos
    ramas_muni_eci = pl.from_pandas(
        cdata_ramas[["Municipio", "eci"]]
    ).unique().drop_nulls().sort("eci", descending=True)

    ramas_muni_eci.head(10)
    return (ramas_muni_eci,)


@app.cell
def _(cdata_ramas, pl):
    ## Veamos el top 10 de industrias más complejas
    pl.from_pandas(
        cdata_ramas[["Actividad económica", "pci"]]
    ).unique().drop_nulls().sort("pci", descending=True).head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Carguemos datos de GDP per capita PPP a nivel municipal de 1990 a 2022.
    Fuente : [Downscaled gridded global dataset for gross domestic product (GDP) per capita PPP over 1990–2022](https://www.nature.com/articles/s41597-025-04487-x)
    """)
    return


@app.cell
def _(pl):
    gdp = pl.read_csv("datos/complejidad/gdp_perCapita_1990_2022.csv")
    gdp
    return (gdp,)


@app.cell
def _(gdp, pl):
    ### Construyamos la columna ID como identificador único del municipio
    id_gdp = gdp.with_columns(
        cve_mun = pl.col("GID_2").map_elements(lambda x : x.split(".")[2].split("_")[0], return_dtype = pl.String), 
        cve_ent = pl.col("GID_2").map_elements(lambda x : x.split('.')[1], return_dtype = pl.String)
    ).with_columns(
        ID = pl.col("cve_ent").map_elements(lambda x : f'{int(x):02d}', return_dtype = pl.String) + pl.col("cve_mun").map_elements(lambda x : f'{int(x):03d}', return_dtype = pl.String)
    ).with_columns(
        pl.col("ID").replace(
            {
                f"{9000 + i:05d}" : f"{9000 + i + 1:05d}"  for i in range(1,17)
            }
        )
    )
    return (id_gdp,)


@app.cell
def _(pl, ramas):
    municipios = pl.from_pandas(
        ramas.assign(
            ID = lambda d : d["Entidad"].str[:2] + d["Municipio"].str[:3]
        )[["ID", "Municipio"]]
    ).unique()
    municipios
    return (municipios,)


@app.cell
def _(id_gdp, municipios, ramas_muni_eci):
    gdp_ice = id_gdp.join(
        municipios, 
        on = "ID"
    ).join(
        ramas_muni_eci, 
        on = "Municipio"
    )
    gdp_ice
    return (gdp_ice,)


@app.cell
def _(alt, gdp_ice):
    base = alt.Chart(gdp_ice).encode(
        x=alt.X('eci:Q').title("ECI"),
        y=alt.Y('2022:Q', scale=alt.Scale(type='log')).title("GDP per capita PPP")
    )

    points = base.mark_point(color='dodgerblue')

    regression_line = base.mark_line(color='firebrick').transform_regression(
        'eci', '2022'
    )
    chart = points + regression_line
    chart
    return


@app.cell
def _(gdp_ice, np):
    from sklearn.linear_model import LinearRegression

    X = gdp_ice["eci"].to_numpy()[:, np.newaxis]
    y = np.log(gdp_ice["2022"].to_numpy())
    reg = LinearRegression().fit(X, y)

    np.corrcoef(reg.predict(X),y)[0,1]
    return X, reg, y


@app.cell
def _(reg):
    reg.score
    return


@app.cell
def _(X, np, reg):
    np.std(reg.predict(X))
    return


@app.cell
def _(X, reg, y):
    from sklearn.metrics import root_mean_squared_error
    root_mean_squared_error(y, reg.predict(X))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Taylor Diagram
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    import geocat.viz as gv

    return gv, np, plt


@app.cell
def _():
    # Create sample data:

    # Model A
    a_sdev = [1.230, 0.988, 1.092, 1.172, 1.064, 0.966, 1.079]  # normalized standard deviation
    a_ccorr = [0.958, 0.973, 0.740, 0.743, 0.922, 0.982, 0.952]  # correlation coefficient
    a_bias = [2.7, -1.5, 17.31, -20.11, 12.5, 8.341, -4.7]  # bias (%)

    # Model B
    b_sdev = [1.129, 0.996, 1.016, 1.134, 1.023, 0.962, 1.048]  # normalized standard deviation
    b_ccorr = [0.963, 0.975, 0.801, 0.814, 0.946, 0.984, 0.968]  # correlation coefficient
    b_bias = [1.7, 2.5, -17.31, 20.11, 19.5, 7.341, 9.2]

    # Sample Variable List
    var_list = ['Surface Pressure', '2m Temp', 'Dew Point Temp', 'U Wind', 'V Wind', 'Precip', 'Cloud Cov']
    return a_bias, a_ccorr, a_sdev, b_bias, b_ccorr, b_sdev, var_list


@app.cell
def _(a_bias, a_ccorr, a_sdev, b_bias, b_ccorr, b_sdev, gv, np, plt, var_list):
    # Create figure and TaylorDiagram instance
    fig = plt.figure(figsize=(10, 10))
    taylor = gv.TaylorDiagram(fig=fig, label='REF')

    # Draw diagonal dashed lines from origin to correlation values
    # Also enforces proper X-Y ratio
    taylor.add_corr_grid(np.array([0.6, 0.9]))

    # Add models to Taylor diagram
    taylor.add_model_set(a_sdev,
                      a_ccorr,
                      percent_bias_on=True, # indicate marker and size to be plotted based on bias_array
                      bias_array=a_bias, # specify bias array
                      color='red',
                      label='Model A',
                      fontsize=16)

    taylor.add_model_set(b_sdev,
                      b_ccorr,
                      percent_bias_on=True,
                      bias_array=b_bias,
                      color='blue',
                      label='Model B',
                      fontsize=16)

    # Add model name
    taylor.add_model_name(var_list, fontsize=16)

    # Add figure legend
    taylor.add_legend(fontsize=16)

    # Add bias legend
    taylor.add_bias_legend()

    # Add constant centered RMS difference contours.
    taylor.add_contours(levels=np.arange(0, 1.1, 0.25),
                     colors='lightgrey',
                     linewidths=0.5);

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
