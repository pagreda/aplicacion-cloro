import numpy as np
import pandas as pd
import joblib as joblib

# ===============================
# AQUÍ DEBES CARGAR TU MODELO REAL
# ===============================
modelo = joblib.load("modelo.pkl")
escala_X = joblib.load("escala_X.pkl")
escala_y = joblib.load("escala_y.pkl")

columnas_para_escalar = [
    "DOSIS_CL",
    "CAUDAL",
    "ORP_IN",
    "TURBIEDAD_IN",
    "PH_IN",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "year"
]


def calcular_dosis_optima(
    variables_actuales,
    dosis_min,
    dosis_max,
    paso,
    cl_objetivo
):
    dosis_candidatas = np.arange(dosis_min, dosis_max + paso, paso)

    resultados = []
    print("Valores de variables_actuales:", {col: f"{val:.2f}" for col, val in zip(columnas_para_escalar, variables_actuales[0])})



    def predecir_cloro(modelo, escala_X, escala_y, variables_actuales):
        X2_df = pd.DataFrame(variables_actuales, columns=columnas_para_escalar)
        X2_scaled = escala_X.transform(X2_df)
        X2_scaled_df = pd.DataFrame(X2_scaled, columns=columnas_para_escalar)
        y2_scaled = modelo.predict(X2_scaled_df)
        return float(
            escala_y.inverse_transform(y2_scaled.reshape(-1, 1))[0, 0]
        )

    cl_predicho_actual = predecir_cloro(
        modelo,
        escala_X,
        escala_y,
        variables_actuales
        )


    for dosis in dosis_candidatas:

        X_df = pd.DataFrame(variables_actuales, columns=columnas_para_escalar)
        X_df["DOSIS_CL"] = dosis

        X_scaled = escala_X.transform(X_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=columnas_para_escalar)

        y_scaled = modelo.predict(X_scaled_df)

         
        cl_predicho = escala_y.inverse_transform(
            y_scaled.reshape(-1, 1)
        )[0, 0] 

        resultados.append({
            "dosis_optima": dosis,
            "cl_predicho_optimo": cl_predicho,
            "error": abs(cl_predicho - cl_objetivo)
        })

    resultados_df = pd.DataFrame(resultados)

    mejor = resultados_df.loc[resultados_df["error"].idxmin()]

    mejor["cl_predicho_dosis_actual"] = cl_predicho_actual


    return mejor
