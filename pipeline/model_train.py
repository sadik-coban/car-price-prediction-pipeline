import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap  # pip install shap
from sqlalchemy import create_engine
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from dotenv import load_dotenv

# Sunucu ortamında grafik çizimi için backend ayarı
import matplotlib

matplotlib.use("Agg")

# --- AYARLAR ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


# --- PREPROCESSOR SINIFI (Aynen Korundu) ---
class CarPricePreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.segment_map = {
            "A Segment": 1,
            "B Segment": 2,
            "C Segment": 3,
            "D Segment": 4,
            "E Segment": 5,
            "F Segment": 6,
            "G Segment": 7,
            "H Segment": 8,
            "K Segment": 9,
            "J Segment": 9,
            "S Segment": 10,
            "M Segment": 10,
        }

    def _print_report(self, message):
        print(f"[INFO] {message}")

    def filter_data(self):
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=["price"])
        fuel_col = self.df["kb_fuel"].fillna(self.df["gb_fuel"])
        self.df = self.df[fuel_col != "Elektrik"]
        self._print_report(
            f"Veri Filtreleme: {initial_len} -> {len(self.df)} satır kaldı."
        )
        return self

    def resolve_kb_gb_conflicts(self):
        pairs = {
            "year": ("kb_year", "gb_year"),
            "mileage": ("kb_mileage", "gb_mileage"),
            "transmission": ("kb_transmission", "gb_transmission"),
            "fuel": ("kb_fuel", "gb_fuel"),
            "body_type": ("kb_body_type", "gb_body_type"),
            "color": ("kb_color", "gb_color"),
        }
        for target, (kb, gb) in pairs.items():
            self.df[target] = self.df[kb].fillna(self.df[gb])
        cols_to_drop = [col for pair in pairs.values() for col in pair]
        self.df.drop(columns=cols_to_drop, inplace=True)
        return self

    def impute_body_type_smart(self):
        def detect_body_type(row):
            current = str(row["body_type"])
            if current not in ["nan", "None", "", "-"]:
                return current
            text = f"{str(row.get('ad_title', ''))} {str(row.get('brand', ''))} {str(row.get('model', ''))}".lower()
            if any(x in text for x in ["cabrio", "kabrio", "roadster"]):
                return "Cabrio"
            if "coupe" in text:
                return "Coupe"
            if any(x in text for x in ["station", "sw", "touring"]):
                return "Station Wagon"
            if any(x in text for x in ["suv", "jeep", "4x4"]):
                return "SUV"
            if any(x in text for x in ["hatchback", "hb"]):
                return "Hatchback"
            return "Sedan"

        self.df["body_type"] = self.df.apply(detect_body_type, axis=1)
        return self

    def impute_segment_smart(self):
        segment_col = "gb_segment"
        segment_map = (
            self.df.groupby(["brand", "series"])[segment_col]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            .to_dict()
        )

        def filler(row):
            if pd.notnull(row[segment_col]):
                return row[segment_col]
            key = (row["brand"], row["series"])
            return segment_map.get(key, "C Segment")

        self.df["segment_clean"] = self.df.apply(filler, axis=1)
        self.df["segment_encoded"] = (
            self.df["segment_clean"].map(self.segment_map).fillna(3)
        )
        self.df.drop(columns=[segment_col], inplace=True)
        return self

    def calculate_expert_risk_score(self):
        def calculate_score(row):
            score = 0
            # Kritik Parçalar
            if row.get("tavan_degisen", 0) == 1:
                score += 150
            elif row.get("tavan_boyali", 0) == 1:
                score += 75
            elif row.get("tavan_lokal", 0) == 1:
                score += 40

            if row.get("kaput_degisen", 0) == 1:
                score += 60
            elif row.get("kaput_boyali", 0) == 1:
                score += 30
            elif row.get("kaput_lokal", 0) == 1:
                score += 15

            if row.get("bagaj_degisen", 0) == 1:
                score += 40
            elif row.get("bagaj_boyali", 0) == 1:
                score += 20
            elif row.get("bagaj_lokal", 0) == 1:
                score += 10

            # Kapı ve Çamurluklar
            doors_changed = sum(
                [
                    row.get(c, 0)
                    for c in [
                        "door_fl_degisen",
                        "door_fr_degisen",
                        "door_rl_degisen",
                        "door_rr_degisen",
                    ]
                ]
            )
            doors_painted = sum(
                [
                    row.get(c, 0)
                    for c in [
                        "door_fl_boyali",
                        "door_fr_boyali",
                        "door_rl_boyali",
                        "door_rr_boyali",
                    ]
                ]
            )
            doors_local = sum(
                [
                    row.get(c, 0)
                    for c in [
                        "door_fl_lokal",
                        "door_fr_lokal",
                        "door_rl_lokal",
                        "door_rr_lokal",
                    ]
                ]
            )
            score += (doors_changed * 10) + (doors_painted * 5) + (doors_local * 2)

            fenders_changed = sum(
                [
                    row.get(c, 0)
                    for c in [
                        "fender_fl_degisen",
                        "fender_fr_degisen",
                        "fender_rl_degisen",
                        "fender_rr_degisen",
                    ]
                ]
            )
            fenders_painted = sum(
                [
                    row.get(c, 0)
                    for c in [
                        "fender_fl_boyali",
                        "fender_fr_boyali",
                        "fender_rl_boyali",
                        "fender_rr_boyali",
                    ]
                ]
            )
            fenders_local = sum(
                [
                    row.get(c, 0)
                    for c in [
                        "fender_fl_lokal",
                        "fender_fr_lokal",
                        "fender_rl_lokal",
                        "fender_rr_lokal",
                    ]
                ]
            )
            score += (fenders_changed * 8) + (fenders_painted * 4) + (fenders_local * 2)
            return score

        self.df = self.df.fillna(0)
        self.df["expert_risk_score"] = self.df.apply(calculate_score, axis=1)

        # Temizlik
        damage_cols = [
            c
            for c in self.df.columns
            if any(
                x in c
                for x in ["tavan_", "kaput_", "bagaj_", "door_", "fender_", "count_"]
            )
        ]
        self.df.drop(columns=damage_cols, inplace=True)
        return self

    def final_cleanup(self):
        bool_cols = self.df.select_dtypes(include=["bool"]).columns
        for col in bool_cols:
            self.df[col] = self.df[col].astype(int)

        cols_to_drop = [
            "ad_title",
            "ad_id",
            "description_text",
            "scraped_at",
            "search_date",
            "location",
            "engine_cc_low",
            "engine_cc_up",
            "engine_cc_is_range",
            "power_hp_low",
            "power_hp_up",
            "power_hp_is_range",
            "kb_condition",
            "kb_seller_type",
            "gb_usage_type",
            "kb_fuel_tank",
            "trunk_capacity_lt",
            "gb_mtv_yearly",
            "kb_fuel_cons_avg",
            "city_fuel_cons",
            "highway_fuel_cons",
            "listing_date",
        ]
        self.df.drop(
            columns=[c for c in cols_to_drop if c in self.df.columns], inplace=True
        )

        cat_cols = self.df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            self.df[col] = self.df[col].fillna("Bilinmiyor").astype(str)
        return self.df

    def run_pipeline(self):
        print("🚀 Preprocessing Pipeline Başlatılıyor...")
        (
            self.filter_data()
            .resolve_kb_gb_conflicts()
            .impute_body_type_smart()
            .impute_segment_smart()
            .calculate_expert_risk_score()
            .final_cleanup()
        )
        print(f"✅ Pipeline Tamamlandı! Son Boyut: {self.df.shape}")
        return self.df


# --- ANA EĞİTİM AKIŞI ---
if __name__ == "__main__":

    # 1. VERİ ÇEKME
    if not DATABASE_URL:
        print("❌ HATA: .env dosyasında DATABASE_URL bulunamadı.")
        exit()

    # print("📥 Veritabanından veri çekiliyor...")
    # engine = create_engine(DATABASE_URL)
    # query = "SELECT * FROM test.car_listings"
    # df = pd.read_sql(query, engine)

    print("📥 Veritabanından veri çekiliyor (Sadece 18.01.2026)...")
    engine = create_engine(DATABASE_URL)

    # --- TARİH FİLTRESİ EKLENDİ ---
    # Eğer tarih sütununun adı 'scraped_at' değilse (örn: 'created_at', 'listing_date') düzeltmelisin.
    # '::date' ifadesi, "2026-01-18 14:30:00" gibi saatli verileri "2026-01-18" yapar.
    query = """
        SELECT * FROM test.car_listings 
        WHERE scraped_at::date = '2026-01-18'
    """

    df = pd.read_sql(query, engine)

    # 2. PREPROCESSING
    preprocessor = CarPricePreprocessor(df)
    clean_df = preprocessor.run_pipeline()

    # 3. FEATURES & TARGET
    X = clean_df.drop(
        [
            "price",
            "kb_trade_available",
            "gb_is_first_owner",
            "kb_is_heavy_damaged",
            "wheelbase_mm",
            "length_mm",
            "height_mm",
            "max_speed_kmh",
            "width_mm",
            "segment_encoded",
            "color",
            "curb_weight_kg",
            "weight_kg",
            "accel_0_100",
            "tramer_fee",
        ],
        axis=1,
        errors="ignore",
    )
    y = clean_df["price"]

    # 4. KATEGORİK DEĞİŞKENLER
    text_features = ["model"]
    cat_features = [
        c for c in X.select_dtypes(include=["object"]).columns if c not in text_features
    ]

    X[text_features] = X[text_features].fillna("Bilinmiyor").astype(str)
    X[cat_features] = X[cat_features].fillna("Bilinmiyor").astype("category")
    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = X[col].astype("float32")
    for col in X.select_dtypes(include=["int64"]).columns:
        X[col] = X[col].astype("int32")

    print(f"\n📊 Veri Hazır: {X.shape[0]} satır, {X.shape[1]} sütun")

    # 5. SPLIT (HOLD-OUT)
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )
    print(
        f"🔹 Eğitim Havuzu: {len(X_train_full)} satır | Test Kasası: {len(X_holdout)} satır"
    )

    # 6. FİNAL MODEL EĞİTİMİ (QUANTILE REGRESSION)
    print("\n🚀 Final Model (Quantile Regression) Eğitiliyor...")
    final_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.1,
        depth=6,
        loss_function="MultiQuantile:alpha=0.05,0.5,0.95",
        cat_features=cat_features,
        text_features=text_features,
        verbose=200,
        early_stopping_rounds=100,
    )

    final_model.fit(X_train_full, y_train_full, eval_set=(X_holdout, y_holdout))

    # 7. PERFORMANS DEĞERLENDİRME
    print("\n📊 Performans Hesaplanıyor...")
    # Predict (N, 3) boyutunda döner: [q05, q50, q95]
    all_preds = final_model.predict(X_holdout)
    pred_median = all_preds[:, 1]  # %50 (Medyan) tahminini alıyoruz (Index 1)

    final_r2 = r2_score(y_holdout, pred_median)
    final_mae = mean_absolute_error(y_holdout, pred_median)
    final_rmse = root_mean_squared_error(y_holdout, pred_median)
    final_mape = np.mean(np.abs((y_holdout - pred_median) / y_holdout)) * 100

    # Coverage Oranı (Güven Aralığı Başarısı)
    pred_low = all_preds[:, 0]
    pred_high = all_preds[:, 2]
    in_range = np.sum((y_holdout >= pred_low) & (y_holdout <= pred_high))
    coverage = (in_range / len(y_holdout)) * 100

    print(f"🏆 Final Test R² (Median): {final_r2:.4f}")
    print(f"📉 Final Test MAE: {final_mae:.0f} TL")
    print(f"📉 Final Test MAPE: %{final_mape:.2f}")
    print(f"🎯 Coverage (%5-%95 Aralığı): %{coverage:.1f} (Hedef: %90)")

    # 8. SHAP ANALİZİ (STRING VE BOYUT HATASI GİDERİLMİŞ)
    print("\n🧠 SHAP Analizi Oluşturuluyor (Pool Yöntemi)...")

    # 8.1 Veriyi CatBoost Havuzuna (Pool) Çevir -> String Hatasını Çözer
    shap_pool = Pool(
        data=X_holdout,
        label=y_holdout,
        cat_features=cat_features,
        text_features=text_features,
    )

    # 8.2 CatBoost'un Kendi Hesaplayıcısını Kullan
    shap_values_raw = final_model.get_feature_importance(shap_pool, type="ShapValues")

    print(f"   -> Ham SHAP Boyutu: {np.shape(shap_values_raw)}")

    # 8.3 DOĞRU KATMANI SEÇME (MultiQuantile Çıktısı)
    if isinstance(shap_values_raw, list):
        print("   -> Liste formatı algılandı. Medyan (index 1) seçiliyor.")
        shap_values = shap_values_raw[1]

    elif len(np.shape(shap_values_raw)) == 3:
        dims = np.shape(shap_values_raw)
        print(f"   -> 3D Array algılandı: {dims}")

        # En küçük boyut genellikle Quantile sayısıdır (3 tane)
        if dims[2] == 3:
            # (N, F, 3) -> Sondaki 3'lüden ortadakini al
            shap_values = shap_values_raw[:, :, 1]
        elif dims[1] == 3:
            # (N, 3, F) -> Ortadaki 3'lüden ortadakini al
            shap_values = shap_values_raw[:, 1, :]
        else:
            shap_values = shap_values_raw[:, :, 1]
    else:
        print("   -> Standart 2D Array algılandı.")
        shap_values = shap_values_raw

    # 8.4 BIAS TERİMİ TEMİZLİĞİ
    n_features_X = X_holdout.shape[1]
    n_features_SHAP = shap_values.shape[1]

    if n_features_SHAP == n_features_X + 1:
        print("   -> Bias (sabit) sütunu çıkarılıyor.")
        shap_values = shap_values[:, :-1]

    # 8.5 GRAFİK ÇİZİMİ
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_holdout, show=False)
        plt.savefig("shap_summary.png", bbox_inches="tight", dpi=300)
        plt.close()
        print("✅ SHAP grafiği 'shap_summary.png' olarak kaydedildi.")
    except Exception as e:
        print(f"⚠️ SHAP Grafik Hatası: {str(e)}")

    # 9. MLOPS ARTIFACTS KAYIT
    print("\n📦 Dosyalar Paketleniyor...")

    # Model
    final_model.save_model("model.cbm")

    # CSV Data (Drift için)
    train_export = X_train_full.copy()
    train_export["price"] = y_train_full
    train_export.to_csv("train_data.csv", index=False)

    test_export = X_holdout.copy()
    test_export["price"] = y_holdout
    test_export.to_csv("test_data.csv", index=False)

    # Metrics JSON
    metrics_data = {
        "r2": round(final_r2, 4),
        "mae": round(final_mae, 2),
        "rmse": round(final_rmse, 2),
        "mape": round(final_mape, 2),
        "coverage_percent": round(coverage, 2),
        "train_size": len(X_train_full),
        "test_size": len(X_holdout),
        "model_type": "CatBoost MultiQuantile (0.05, 0.5, 0.95)",
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)

    print("✅ TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
