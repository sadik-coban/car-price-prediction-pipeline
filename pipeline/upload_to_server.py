import os
import json
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
from huggingface_hub.utils import EntryNotFoundError
from dotenv import load_dotenv


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

load_dotenv()
REPO_ID = "steelvoid/car_price_prediction"
TOKEN = os.getenv("HF_TOKEN")
VERSION = "v1"
api = HfApi(token=TOKEN)


def push_new_version():
    print(f"🚀 {VERSION} paketleniyor ve yükleniyor...")

    # A. Yüklenecek Dosyaların Listesi
    local_files = [
        "model.cbm",
        "train_data.csv",
        "test_data.csv",
        "metrics.json",
        "shap_summary.png",
    ]

    # Dosya Kontrolü
    for f in local_files:
        if not os.path.exists(f):
            print(f"❌ HATA: '{f}' dosyası bulunamadı! Önce model eğitimi yapılmalı.")
            return

    try:
        # --- 1. REGISTRY GÜNCELLEME ---
        print("📝 Registry dosyası hazırlanıyor...")

        registry = []
        try:
            reg_path = hf_hub_download(repo_id=REPO_ID, filename="registry.json")
            with open(reg_path, "r") as f:
                registry = json.load(f)
        except (EntryNotFoundError, FileNotFoundError):
            print("ℹ️ Registry bulunamadı, yeni oluşturuluyor.")
            registry = []

        # Yeni metrikleri oku
        with open("metrics.json", "r") as f:
            new_metrics = json.load(f)

        new_entry = {
            "version_id": VERSION,
            "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "metrics": new_metrics,
            "description": f"Quantile Regression Model ({VERSION}) - Auto Push",
        }

        # Registry Listesini Güncelle
        existing_index = next(
            (index for (index, d) in enumerate(registry) if d["version_id"] == VERSION),
            None,
        )
        if existing_index is not None:
            registry[existing_index] = new_entry
        else:
            registry.insert(0, new_entry)

        # Registry'yi diske yaz (Yükleme listesine eklemek için)
        with open("registry.json", "w") as f:
            json.dump(registry, f, indent=4)

        # --- 2. ATOMIC COMMIT (HEPSİNİ TEK SEFERDE YÜKLE) ---
        print("📦 Tüm dosyalar tek pakette gönderiliyor (Atomic Commit)...")

        operations = []

        # a. Model ve Veri Dosyaları
        for f in local_files:
            operations.append(
                CommitOperationAdd(path_in_repo=f"{VERSION}/{f}", path_or_fileobj=f)
            )

        # b. Registry Dosyası
        operations.append(
            CommitOperationAdd(
                path_in_repo="registry.json", path_or_fileobj="registry.json"
            )
        )

        # c. Commit İşlemi (Yüklemeyi Başlat)
        api.create_commit(
            repo_id=REPO_ID,
            operations=operations,
            commit_message=f"Upload {VERSION} artifacts and update registry",
        )

        print(f"\n✅ BAŞARILI! {VERSION} ve tüm dosyalar hatasız yüklendi.")
        print("👉 Dashboard'ı yenileyebilirsin.")

    except Exception as e:
        print(f"\n❌ Yükleme sırasında hata oluştu: {e}")
        # Detaylı hata görmek için:
        # import traceback
        # traceback.print_exc()


if __name__ == "__main__":
    push_new_version()
