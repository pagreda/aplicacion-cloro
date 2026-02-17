from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import numpy as np
from model_utils import calcular_dosis_optima
from datetime import datetime
import os

app = FastAPI(
    title="Optimización de Dosis de Cloro",
    version="1.0"
)

# ===============================
# CORS
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción: restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# MODELOS DE DATOS
# ===============================
class InputData(BaseModel):
    dosis_cl: float = Field(..., ge=5.0, le=9.0)
    caudal: float = Field(..., ge=5.0, le=15.0)
    orp: int = Field(..., ge=300, le=400)
    turbiedad: float = Field(..., ge=5.0, le=100.0)
    ph: float = Field(..., ge=5.0, le=15.0)
    fecha: str  # "YYYY-MM-DD"
    hora: str   # "HH:MM"


class OutputData(BaseModel):
    cl_predicho_actual: float
    dosis_cl_optima: float
    cl_predicho_optimo: float
    error: float


# ===============================
# ENDPOINT PRINCIPAL
# ===============================
@app.post("/predict", response_model=OutputData)
def predict(data: InputData):

    try:
        # ===============================
        # Construcción de variables_actuales
        # ===============================
        # ===============================
        # Fecha y hora desde el frontend
        # ===============================
        dt = datetime.fromisoformat(f"{data.fecha}T{data.hora}")

        hour = dt.hour                    # 0–23
        dow = dt.weekday()                # 0=Lunes, 6=Domingo
        month = dt.month                  # 1–12
        year = dt.year

        print(f"Fecha: {data.fecha} | Hora: {data.hora}")
        print(f"hour: {hour:02d}, dow: {dow:02d}, month: {month:02d}, year: {year:04d}")

        

        variables_actuales = np.array([[
            data.dosis_cl,
            data.caudal,
            data.orp,
            data.turbiedad,
            data.ph,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            year
        ]])

        resultado = calcular_dosis_optima(
            variables_actuales=variables_actuales,
            dosis_min=4.0,
            dosis_max=9.5,
            paso=0.05,
            cl_objetivo=0.5
        )

        def safe_float(value, default=0.0):
            if value is None:
                return default
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                return default
            return float(value)

        return {
            "cl_predicho_actual": safe_float(resultado["cl_predicho_dosis_actual"]),
            "dosis_cl_optima": safe_float(resultado["dosis_optima"]),
            "cl_predicho_optimo": safe_float(resultado["cl_predicho_optimo"]),
            "error": safe_float(resultado["error"])
        }


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===============================
# SERVIR ARCHIVOS ESTÁTICOS LOCAL
# ===============================
# Monta la carpeta 'static' en la raíz
#static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
#if os.path.exists(static_dir):
#    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Ruta raíz que sirve maincloro.html
#@app.get("/")
#async def root():
#    html_file = os.path.join(static_dir, "maincloro.html")
#    if os.path.exists(html_file):
#        return FileResponse(html_file, media_type="text/html")
#    return {"message": "maincloro.html no encontrado"}

# Ruta alternativa para maincloro.html
#@app.get("/maincloro.html")
#async def get_maincloro():
#    html_file = os.path.join(static_dir, "maincloro.html")
#    if os.path.exists(html_file):
#        return FileResponse(html_file, media_type="text/html")
#    return {"message": "maincloro.html no encontrado"}

# ===============================
# SERVIR ARCHIVOS ESTÁTICOS RENDER
# ===============================
# Obtener la ruta base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Ruta raíz para servir HTML
@app.get("/")
async def read_root():
    html_path = os.path.join(STATIC_DIR, "maincloro.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return {"error": "HTML file not found"}