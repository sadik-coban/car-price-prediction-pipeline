import os
import json
import pandas as pd
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sqlalchemy import create_engine, text
from huggingface_hub import hf_hub_download
from scipy.stats import ks_2samp, wasserstein_distance
from functools import lru_cache

import numpy as np
from catboost import CatBoostRegressor
from dotenv import load_dotenv

# --- 1. CONFIGURATION & SETUP ---
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
REPO_ID = "steelvoid/car_price_prediction"
STATIC_DIR = "static_reports"


os.makedirs(STATIC_DIR, exist_ok=True)


loaded_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Uygulama başlatılıyor... Model ön yüklemesi başlıyor.")
    try:
        # 1. Registry'i çek
        versions = get_cached_versions(HF_TOKEN)
        if versions:
            # 2. En güncel versiyonu bul (Tarihe göre sıralı geldiğini varsayıyoruz)
            latest_version = sorted(
                versions, key=lambda x: x.get("date", ""), reverse=True
            )[0]
            v_id = latest_version["version_id"]

            print(f"📥 En güncel model ({v_id}) indiriliyor...")
            # Modeli indir ve RAM'e at
            model_path = get_file_from_hf(v_id, "model.cbm")
            model = CatBoostRegressor()
            model.load_model(model_path)
            loaded_models[v_id] = model
            print(f"✅ Model ({v_id}) RAM'e yüklendi ve hazır!")
    except Exception as e:
        print(f"⚠️ Startup sırasında model yüklenemedi: {e}")
        print("Model ilk istek geldiğinde yüklenecek.")

    yield  # Uygulama burada çalışır

    print("🛑 Uygulama kapatılıyor. RAM temizleniyor...")
    loaded_models.clear()


app = FastAPI(
    title="Car Price Prediction & MLOps API",
    lifespan=lifespan,  # Lifespan'i buraya bağlıyoruz
)

# CORS Settings (Production Security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Sadece izin verilen domainler
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Mount
app.mount("/reports", StaticFiles(directory=STATIC_DIR), name="reports")

# Database Connection
# Production için echo=False ve pool ayarları önemlidir
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,  # Render free tier için makul
    max_overflow=20,
    pool_recycle=3600,
    connect_args={"keepalives": 1, "keepalives_idle": 30},
)

# --- 2. DATA MODELS (PYDANTIC) ---


class DamageInputs(BaseModel):
    """Inputs required to calculate expert_risk_score"""

    roof_status: str = Field("Orjinal", description="Orjinal, Boyalı, Değişen, Lokal")
    hood_status: str = Field("Orjinal", description="Orjinal, Boyalı, Değişen, Lokal")
    trunk_status: str = Field("Orjinal", description="Orjinal, Boyalı, Değişen, Lokal")
    doors_changed: int = Field(0, ge=0, le=4)
    doors_painted: int = Field(0, ge=0, le=4)
    doors_local: int = Field(0, ge=0, le=4)
    fenders_changed: int = Field(0, ge=0, le=4)
    fenders_painted: int = Field(0, ge=0, le=4)
    fenders_local: int = Field(0, ge=0, le=4)


class CarPredictionInput(BaseModel):
    brand: str
    series: str
    model: str
    transmission: str
    fuel: str
    body_type: str
    kb_drivetrain: str
    gb_warranty_status: str
    segment_clean: str
    year: int
    mileage: float
    engine_cc_val: float
    power_hp_val: float
    torque_nm: float
    cylinder_count: int
    is_heavy_damaged: int = Field(0, description="1 if heavy damaged, else 0")
    damage_details: DamageInputs


# --- 3. HELPER FUNCTIONS ---


def get_file_from_hf(version_id: str, filename: str):
    """Downloads a file from Hugging Face with caching."""
    # Production'da timeout yememek için local_files_only denenebilir
    # ama dosya yoksa patlar. Standart indirme en güvenlisidir.
    return hf_hub_download(
        repo_id=REPO_ID, filename=f"{version_id}/{filename}", token=HF_TOKEN
    )


def calculate_risk_score_logic(damage: DamageInputs) -> float:
    score = 0.0
    if damage.roof_status == "Değişen":
        score += 150
    elif damage.roof_status == "Boyalı":
        score += 75
    elif damage.roof_status == "Lokal Boyalı":
        score += 40

    if damage.hood_status == "Değişen":
        score += 60
    elif damage.hood_status == "Boyalı":
        score += 30
    elif damage.hood_status == "Lokal Boyalı":
        score += 15

    if damage.trunk_status == "Değişen":
        score += 40
    elif damage.trunk_status == "Boyalı":
        score += 20
    elif damage.trunk_status == "Lokal Boyalı":
        score += 10

    score += (
        (damage.doors_changed * 10)
        + (damage.doors_painted * 5)
        + (damage.doors_local * 2)
    )
    score += (
        (damage.fenders_changed * 8)
        + (damage.fenders_painted * 4)
        + (damage.fenders_local * 2)
    )
    return float(score)


def build_filter_clause(
    brand,
    series,
    min_price,
    max_price,
    min_year,
    max_year,
    min_km,
    max_km,
    fuel,
    include_series=True,
):
    filters = ["1=1"]
    params = {}
    if brand and brand != "Tümü":
        filters.append("brand = :brand")
        params["brand"] = brand
    if include_series and series and series != "Tümü":
        filters.append("series = :series")
        params["series"] = series
    if min_price:
        filters.append("price >= :min_price")
        params["min_price"] = min_price
    if max_price:
        filters.append("price <= :max_price")
        params["max_price"] = max_price
    if min_year:
        filters.append("kb_year >= :min_year")
        params["min_year"] = min_year
    if max_year:
        filters.append("kb_year <= :max_year")
        params["max_year"] = max_year
    if min_km:
        filters.append("kb_mileage >= :min_km")
        params["min_km"] = min_km
    if max_km:
        filters.append("kb_mileage <= :max_km")
        params["max_km"] = max_km
    if fuel and fuel != "Tümü":
        filters.append("kb_fuel = :fuel")
        params["fuel"] = fuel
    return " AND ".join(filters), params


# --- 4. ENDPOINTS ---


@lru_cache(maxsize=1)
def get_cached_versions(token):
    path = hf_hub_download(repo_id=REPO_ID, filename="registry.json", token=token)
    with open(path, "r") as f:
        return json.load(f)


@app.get("/versions")
def get_versions():
    try:
        data = get_cached_versions(HF_TOKEN)
        return sorted(data, key=lambda x: x.get("date", ""), reverse=True)
    except Exception as e:
        print(f"Versiyon çekme hatası: {e}")
        return []


@lru_cache(maxsize=10)
def get_shap_image_path(version_id: str):
    try:
        file_path = hf_hub_download(
            repo_id=REPO_ID, filename=f"{version_id}/shap_summary.png", token=HF_TOKEN
        )
        return file_path
    except Exception as e:
        print(f"SHAP grafiği bulunamadı ({version_id}): {e}")
        return None


@app.get("/api/shap/{version_id}")
def get_shap_plot(version_id: str):
    image_path = get_shap_image_path(version_id)
    if image_path and os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="SHAP grafiği bulunamadı.")


@app.post("/predict/{version_id}")
def predict_price(version_id: str, input_data: CarPredictionInput):
    print("\n" + "=" * 30)
    print(f"🚀 GELEN TAHMİN İSTEĞİ (Version: {version_id})")
    print(f"📦 Raw Input Data: {input_data}")
    print("=" * 30 + "\n")
    try:
        # --- MODEL CHECK ---
        if version_id not in loaded_models:
            print(f"📥 {version_id} ilk kez indiriliyor ve belleğe alınıyor...")
            # Burası sadece model daha önce hiç yüklenmemişse çalışır
            model_path = get_file_from_hf(version_id, "model.cbm")
            model = CatBoostRegressor()
            model.load_model(model_path)
            loaded_models[version_id] = model

        model = loaded_models[version_id]
        calculated_score = calculate_risk_score_logic(input_data.damage_details)

        feature_dict = {
            "brand": str(input_data.brand),
            "series": str(input_data.series),
            "model": str(input_data.model),
            "engine_cc_val": float(input_data.engine_cc_val),
            "power_hp_val": float(input_data.power_hp_val),
            "kb_drivetrain": str(input_data.kb_drivetrain),
            "gb_warranty_status": str(input_data.gb_warranty_status),
            "torque_nm": float(input_data.torque_nm),
            "cylinder_count": int(input_data.cylinder_count),
            "is_heavy_damaged": int(input_data.is_heavy_damaged),
            "year": int(input_data.year),
            "mileage": float(input_data.mileage),
            "transmission": str(input_data.transmission),
            "fuel": str(input_data.fuel),
            "body_type": str(input_data.body_type),
            "segment_clean": str(input_data.segment_clean),
            "expert_risk_score": float(calculated_score),
        }

        df = pd.DataFrame([feature_dict])
        expected_features = model.feature_names_
        df = df[expected_features]

        cat_features = [
            "brand",
            "series",
            "model",
            "transmission",
            "fuel",
            "body_type",
            "kb_drivetrain",
            "gb_warranty_status",
            "segment_clean",
        ]
        for col in cat_features:
            if col in df.columns:
                df[col] = df[col].astype(str)

        prediction_result = model.predict(df)
        q05 = prediction_result[0][0]
        q50 = prediction_result[0][1]
        q95 = prediction_result[0][2]

        q05 = q05 * 0.98
        q95 = q95 * 1.02
        implied_margin_percent = ((q95 - q05) / 2) / q50 * 100

        return {
            "price": int(q50),
            "price_range": {
                "min": int(q05),
                "max": int(q95),
                "margin_percent": round(implied_margin_percent, 1),
            },
            "version": version_id,
            "calculated_risk_score": calculated_score,
            "currency": "TL",
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def compute_histogram_bins(data1, data2, bins=20):
    combined = np.concatenate([data1, data2])
    min_val, max_val = np.min(combined), np.max(combined)
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    hist1, _ = np.histogram(data1, bins=bin_edges, density=True)
    hist2, _ = np.histogram(data2, bins=bin_edges, density=True)
    chart_data = []
    for i in range(bins):
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        chart_data.append(
            {
                "bin": round(bin_center, 2),
                "ref_density": float(hist1[i]),
                "curr_density": float(hist2[i]),
            }
        )
    return chart_data


def calculate_custom_drift(ref_df: pd.DataFrame, curr_df: pd.DataFrame):
    drift_results = []
    numeric_cols = ref_df.select_dtypes(include=[np.number]).columns.tolist()
    if "price" in numeric_cols:
        numeric_cols.remove("price")
        numeric_cols.insert(0, "price")

    for col in numeric_cols:
        ref_data = ref_df[col].dropna().values
        curr_data = curr_df[col].dropna().values
        if len(ref_data) == 0 or len(curr_data) == 0:
            continue

        ks_stat, p_value = ks_2samp(ref_data, curr_data)
        emd_score = wasserstein_distance(ref_data, curr_data)

        std_dev = np.std(ref_data)
        normalized_emd = (
            emd_score / std_dev if std_dev > 0 else (999.0 if emd_score > 0 else 0.0)
        )
        is_drifted = (p_value < 0.05) and (normalized_emd > 0.1)
        chart_data = compute_histogram_bins(ref_data, curr_data)

        drift_results.append(
            {
                "feature": col,
                "drift_detected": bool(is_drifted),
                "p_value": float(round(p_value, 5)),
                "ks_statistic": float(round(ks_stat, 4)),
                "emd_score": float(round(emd_score, 2)),
                "chart_data": chart_data,
                "normalized_emd": float(round(normalized_emd, 3)),
            }
        )
    return drift_results


@app.get("/drift/{ref_ver}/{curr_ver}")
def analyze_drift(ref_ver: str, curr_ver: str):
    try:
        ref_path = hf_hub_download(
            repo_id=REPO_ID, filename=f"{ref_ver}/train_data.csv", token=HF_TOKEN
        )
        curr_path = hf_hub_download(
            repo_id=REPO_ID, filename=f"{curr_ver}/train_data.csv", token=HF_TOKEN
        )
        ref_df = pd.read_csv(ref_path)
        curr_df = pd.read_csv(curr_path)
        results = calculate_custom_drift(ref_df, curr_df)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz Hatası: {str(e)}")


@app.get("/api/dashboard-data")
def get_dashboard(
    brand: Optional[str] = Query(None),
    series: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    min_year: Optional[int] = Query(None),
    max_year: Optional[int] = Query(None),
    min_km: Optional[int] = Query(None),
    max_km: Optional[int] = Query(None),
    fuel: Optional[str] = Query(None),
):
    base_where, base_params = build_filter_clause(
        brand,
        None,
        min_price,
        max_price,
        min_year,
        max_year,
        min_km,
        max_km,
        fuel,
        include_series=False,
    )
    final_where, final_params = build_filter_clause(
        brand,
        series,
        min_price,
        max_price,
        min_year,
        max_year,
        min_km,
        max_km,
        fuel,
        include_series=True,
    )

    with engine.connect() as conn:
        lists_query = text(
            f"SELECT ARRAY_AGG(DISTINCT brand ORDER BY brand) as brands, ARRAY_AGG(DISTINCT series ORDER BY series) as series_list FROM test.car_listings WHERE {base_where}"
        )
        lists_result = conn.execute(lists_query, base_params).fetchone()
        unique_brands = lists_result[0] if lists_result and lists_result[0] else []
        unique_series = lists_result[1] if lists_result and lists_result[1] else []

        damage_select = f"""
            json_build_array(
                json_build_object('part', 'Kaput', 'value', SUM(COALESCE(kaput_degisen,0) + COALESCE(kaput_boyali,0))),
                json_build_object('part', 'Tavan', 'value', SUM(COALESCE(tavan_degisen,0) + COALESCE(tavan_boyali,0))),
                json_build_object('part', 'Bagaj', 'value', SUM(COALESCE(bagaj_degisen,0) + COALESCE(bagaj_boyali,0))),
                json_build_object('part', 'Sol Ön Çamurluk', 'value', SUM(COALESCE(fender_fl_degisen,0) + COALESCE(fender_fl_boyali,0))),
                json_build_object('part', 'Sağ Ön Çamurluk', 'value', SUM(COALESCE(fender_fr_degisen,0) + COALESCE(fender_fr_boyali,0))),
                json_build_object('part', 'Sol Ön Kapı', 'value', SUM(COALESCE(door_fl_degisen,0) + COALESCE(door_fl_boyali,0))),
                json_build_object('part', 'Sağ Ön Kapı', 'value', SUM(COALESCE(door_fr_degisen,0) + COALESCE(door_fr_boyali,0))),
                json_build_object('part', 'Sol Arka Kapı', 'value', SUM(COALESCE(door_rl_degisen,0) + COALESCE(door_rl_boyali,0))),
                json_build_object('part', 'Sağ Arka Kapı', 'value', SUM(COALESCE(door_rr_degisen,0) + COALESCE(door_rr_boyali,0))),
                json_build_object('part', 'Sol Arka Çamurluk', 'value', SUM(COALESCE(fender_rl_degisen,0) + COALESCE(fender_rl_boyali,0))),
                json_build_object('part', 'Sağ Arka Çamurluk', 'value', SUM(COALESCE(fender_rr_degisen,0) + COALESCE(fender_rr_boyali,0)))
            )
        """

        mega_query = text(
            f"""
            WITH filtered_data AS (SELECT * FROM test.car_listings WHERE {final_where}),
            kpi_stats AS (SELECT COUNT(*) as total_count, COALESCE(AVG(price), 0) as avg_price FROM filtered_data),
            boxplot_agg AS (SELECT json_object_agg(brand, prices) as data FROM (SELECT brand, array_agg(price) as prices FROM filtered_data WHERE price IS NOT NULL GROUP BY brand) t),
            scatter_agg AS (SELECT json_object_agg(brand, points) as data FROM (SELECT brand, json_agg(json_build_array(kb_mileage, price)) as points FROM filtered_data WHERE kb_mileage IS NOT NULL AND price IS NOT NULL GROUP BY brand) t),
            line_chart_agg AS (SELECT json_agg(kb_year ORDER BY kb_year) as years, json_agg(avg_price ORDER BY kb_year) as prices FROM (SELECT kb_year, AVG(price) as avg_price FROM filtered_data WHERE kb_year IS NOT NULL GROUP BY kb_year) t),
            donut_agg AS (SELECT json_agg(json_build_object('name', kb_fuel, 'value', count)) as data FROM (SELECT kb_fuel, COUNT(*) as count FROM filtered_data WHERE kb_fuel IS NOT NULL GROUP BY kb_fuel) t),
            radar_stats AS (SELECT brand, AVG(price) as avg_price, AVG(kb_mileage) as avg_km, AVG(power_hp_val) as avg_hp, AVG(kb_fuel_cons_avg) as avg_fuel FROM filtered_data GROUP BY brand),
            global_max AS (SELECT MAX(price) as max_price, MAX(kb_mileage) as max_km, MAX(power_hp_val) as max_hp, MAX(kb_fuel_cons_avg) as max_fuel, COUNT(DISTINCT brand) as brand_count FROM filtered_data),
            damage_agg AS (SELECT {damage_select} as data FROM filtered_data)
            SELECT (SELECT total_count FROM kpi_stats), (SELECT avg_price FROM kpi_stats), (SELECT data FROM boxplot_agg), (SELECT data FROM scatter_agg), (SELECT years FROM line_chart_agg), (SELECT prices FROM line_chart_agg), (SELECT data FROM donut_agg), (SELECT data FROM damage_agg), (SELECT CASE WHEN (SELECT brand_count FROM global_max) <= 5 THEN json_build_object('indicators', json_build_array(json_build_object('name', 'Fiyat', 'max', (SELECT max_price FROM global_max) * 1.1), json_build_object('name', 'KM', 'max', (SELECT max_km FROM global_max) * 1.1), json_build_object('name', 'Beygir Gücü', 'max', (SELECT max_hp FROM global_max) * 1.1), json_build_object('name', 'Yakıt Tüketimi', 'max', (SELECT max_fuel FROM global_max) * 1.1)), 'series', (SELECT json_agg(json_build_object('name', brand, 'value', json_build_array(avg_price, avg_km, avg_hp, avg_fuel))) FROM radar_stats)) ELSE json_build_object('indicators', '[]'::json, 'series', '[]'::json) END) as radar_data
        """
        )

        result = conn.execute(mega_query, final_params).fetchone()
        if not result or result[0] == 0:
            return {
                "brands": unique_brands,
                "seriesList": unique_series,
                "kpi": {"total": 0, "avgPrice": 0},
                "boxplotData": {},
                "scatterData": {},
                "lineChartData": {},
                "donutChartData": [],
                "radarChartData": {},
                "damageChartData": [],
            }

        response_data = {
            "brands": unique_brands,
            "seriesList": unique_series,
            "kpi": {
                "total": result[0] or 0,
                "avgPrice": float(result[1]) if result[1] else 0,
            },
            "boxplotData": result[2] or {},
            "scatterData": result[3] or {},
            "lineChartData": {"years": result[4] or [], "prices": result[5] or []},
            "donutChartData": result[6] or [],
            "damageChartData": result[7] or [],
            "radarChartData": result[8] or {},
        }
        return JSONResponse(
            content=response_data, headers={"Cache-Control": "public, max-age=3600"}
        )


@app.get("/api/options")
def get_dropdown_options(
    brand: Optional[str] = Query(None), series: Optional[str] = Query(None)
):
    with engine.connect() as conn:
        response = {"brands": [], "series": [], "models": []}
        brands_query = text(
            "SELECT DISTINCT brand FROM test.car_listings ORDER BY brand"
        )
        response["brands"] = [r[0] for r in conn.execute(brands_query).fetchall()]
        if brand:
            series_query = text(
                "SELECT DISTINCT series FROM test.car_listings WHERE brand = :brand ORDER BY series"
            )
            response["series"] = [
                r[0] for r in conn.execute(series_query, {"brand": brand}).fetchall()
            ]
        if brand and series:
            models_query = text(
                "SELECT DISTINCT model FROM test.car_listings WHERE brand = :brand AND series = :series ORDER BY model"
            )  # bind parameter
            response["models"] = [
                r[0]
                for r in conn.execute(
                    models_query, {"brand": brand, "series": series}
                ).fetchall()
            ]
        return response


# Use Gunicorn for production:
# gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
