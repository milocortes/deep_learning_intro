import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Datos O*NET
    """)
    return


@app.cell
def _(pd):
    ### Cargamos datos de Skills de O*NET
    skills = pd.read_table("datos/ocupaciones/datos/Skills.txt")
    skills
    return (skills,)


@app.cell
def _(skills):
    ### Filtramos por Scale ID == "IM" y por el último levantamiento Date == "08/2023"
    skill_filtradas = skills[
        (skills["Scale ID"] == 'IM') 
    ][
        ["O*NET-SOC Code", "Element ID", "Element Name", "Data Value"]
    ].reset_index(
        drop = True
    ).groupby(
        ["O*NET-SOC Code", "Element ID", "Element Name"]
    ).agg(
        {
            "Data Value" : "mean"
        }
    ).reset_index()

    skill_filtradas
    return (skill_filtradas,)


@app.cell
def _(pd):
    ## Cargamos el archivo de correspondencias ONET-SINCO
    sinco = pd.read_excel(
        "datos/ocupaciones/datos/sinco-onet.xlsx"
    )
    sinco
    return (sinco,)


@app.cell
def _(sinco):
    ### Nos quedamos con las columnas que necesitamos
    sinco_min = sinco[
        ["onetsoccode", "title", "sinco", "nombre_sinco"]
    ]
    sinco_min
    return (sinco_min,)


@app.cell
def _(sinco_min, skill_filtradas):
    ### Reunimos los dataframes
    skill_sinco = skill_filtradas.merge(
        sinco_min, 
        left_on="O*NET-SOC Code", 
        right_on="onetsoccode", 
        how="left"
    )
    skill_sinco
    return (skill_sinco,)


@app.cell
def _(skill_sinco):
    ### Agrupamos por ocupacion SINCO y por las skills
    skill_sinco_agg = skill_sinco.groupby(
        ["sinco", "nombre_sinco", "Element ID", "Element Name"]
    ).agg(
        {
            "Data Value" : "mean"
        }
    ).reset_index()
    skill_sinco_agg
    return (skill_sinco_agg,)


@app.cell
def _(pd):
    ## Carga SINCO
    sinco_cat = pd.read_csv("datos/ocupaciones/datos/sinco_2019_2011.csv")[["cve_sinco_2019", "nombre_sinco_2019"]].dropna()

    ## Nos quedamos sólo con categorías a tres digitos
    sinco_cat = sinco_cat[sinco_cat["cve_sinco_2019"].apply(lambda x : len(x)==3)].reset_index(drop = True)
    sinco_cat
    return (sinco_cat,)


@app.cell
def _(sinco_cat, skill_sinco_agg):
    import polars as pl

    skill_sinco_3d = pl.from_pandas(
        skill_sinco_agg
    ).with_columns(
        sinco_3d = pl.col("sinco").cast(pl.String).map_elements(lambda x : x[:3])
    ).group_by(
        "sinco_3d", "Element ID", "Element Name"
    ).agg(
        pl.col("Data Value").mean()
    ).with_columns(
        year = pl.lit(2026)
    ).to_pandas()

    skill_sinco_3d = skill_sinco_3d.merge(
        sinco_cat, 
        left_on = "sinco_3d", 
        right_on= "cve_sinco_2019", 
        how = "left"
    )

    skill_sinco_3d
    return pl, skill_sinco_3d


@app.cell
def _(skill_sinco_3d):

    from ecomplexity import ecomplexity

    # Calculate complexity
    trade_cols = {'time':'year', 'loc':'sinco_3d', 'prod':'Element Name', 'val':'Data Value'}
    cdata = ecomplexity(skill_sinco_3d, trade_cols)
    cdata
    return (cdata,)


@app.cell
def _(cdata, pl):
    pl.from_pandas(cdata).select("sinco_3d", "nombre_sinco_2019", "eci").unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calculamos ingresos de la ENOE
    """)
    return


@app.cell
def _(pl):
    ## Cargamos datos
    sdemt226 = pl.read_parquet("datos/ocupaciones/datos/ENOE_SDEMT226.parquet")
    coe1t226 = pl.read_parquet("datos/ocupaciones/datos/ENOE_COE1T226.parquet")
    #coe2t226 = pl.read_parquet("datos/ocupaciones/datos/ENOE_COE2T226.parquet")

    ## Función que crea folio
    def crea_folio(df : pl.DataFrame) : 
        return df.with_columns(
            pl.concat_str(
                    [
                        pl.col("cd_a", "cve_ent", "con", "v_sel", "n_hog", "h_mud", "n_ren")
                    ],
                    separator="",
                ).alias("folio"),
        )

    sdemt226 = crea_folio(sdemt226)
    coe1t226 = crea_folio(coe1t226)
    #coe2t226 = crea_folio(coe2t226)
    return coe1t226, sdemt226


@app.cell
def _(coe1t226, pl, sdemt226):
    import numpy as np 

    ## Reune datos
    enoe = sdemt226.join(
        coe1t226, 
        on = "folio"
    )

    enoe = enoe.with_columns(
        sinco  = pl.col("p3").cast(pl.String).map_elements(lambda x : x[:3])
    ).drop_nulls(subset=["ingocup", "sinco"] )

    enoe
    return (enoe,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(enoe, pl, sinco_cat):
    ## Incorporamos el esquema de muestreo
    import svy

    # Define the survey design
    design = svy.Design(stratum="est", psu="upm", wgt="fac_tri")

    # Create a sample object
    sample = svy.Sample(data=enoe, design=design)

    ingreso_sinco = sample.estimation.mean(y="ingocup", by = "sinco").to_polars()
    ingreso_sinco = ingreso_sinco.join(
        pl.from_pandas(sinco_cat), 
        left_on="sinco", 
        right_on="cve_sinco_2019"
    )

    ingreso_sinco
    return (ingreso_sinco,)


@app.cell
def _(pd):
    ## Carga SINCO
    sinco_1d = pd.read_csv("datos/ocupaciones/datos/sinco_2019_2011.csv")[["cve_sinco_2019", "nombre_sinco_2019"]].dropna()

    ## Nos quedamos sólo con categorías a tres digitos
    sinco_1d = sinco_1d[sinco_1d["cve_sinco_2019"].apply(lambda x : len(x)==1)].reset_index(drop = True)
    sinco_1d = sinco_1d.rename( columns=
        {
            "cve_sinco_2019" : "sinco_1d", 
            "nombre_sinco_2019" : "nombre_sinco_1d"
        }
    ).reset_index(drop = True)
    sinco_1d
    return (sinco_1d,)


@app.cell
def _(cdata, ingreso_sinco, pl, sinco_1d):
    sinco_eci_ingresos = pl.from_pandas(
        cdata
    ).select(
        "sinco_3d", "nombre_sinco_2019", "eci"
    ).unique().join(
        ingreso_sinco.filter(pl.col("cv") <= 0.2).select("sinco", "est"), 
        left_on="sinco_3d", 
        right_on="sinco"
    ).with_columns(
        log_ingresos = pl.col("est").log()
    )

    sinco_eci_ingresos = sinco_eci_ingresos.with_columns(
        sinco_1d = pl.col("sinco_3d").map_elements(lambda x : x[0] , return_dtype=pl.String)
    )

    sinco_eci_ingresos = sinco_eci_ingresos.join(
        pl.from_pandas(sinco_1d), 
        on = "sinco_1d", 
        how = "left"
    )
    sinco_eci_ingresos
    return (sinco_eci_ingresos,)


@app.cell
def _(sinco_eci_ingresos):
    import altair as alt

    ## Crea diagrama de dispersion
    scatter = alt.Chart(sinco_eci_ingresos).mark_circle(size=60).encode(
        x=alt.X('eci', scale=alt.Scale(zero=False)),
        y=alt.Y('est', scale=alt.Scale(zero=False)),
        color = "nombre_sinco_1d",
        tooltip=['nombre_sinco_2019', "est", "nombre_sinco_1d"]
    ).interactive()

    # Crear la línea de regresión
    line = alt.Chart(sinco_eci_ingresos).mark_line(color='red').transform_regression(
        'eci', 'est'
    ).encode(
        x='eci:Q',
        y='est:Q'
    )

    scatter + line
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
