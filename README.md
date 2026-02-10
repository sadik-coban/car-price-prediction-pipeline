# Car Price Prediction Pipeline

An end-to-end Machine Learning pipeline designed for automated data collection, persistent storage, and versioned model deployment. This project bypasses local heavy-data tracking in favor of a cloud-native **Hugging Face Model Store** architecture.

## Pipeline Workflow

The system follows a linear, automated flow from raw data to production-ready inference:

1. **Scrape:** Raw car advertisements are collected via custom scrapers.
2. **Database (Persistence):** Extracted data is pushed to a persistence store.
* *Logic:* Ad IDs are used as unique identifiers. The system utilizes `INSERT IGNORE` or `UPSERT` logic—if an `ad_id` already exists, the record is **not overwritten**, ensuring data integrity and historical consistency.


3. **Preprocess:** Raw data is cleaned, features are engineered, and categorical variables are encoded for the CatBoost regressor.
4. **Train:** The model is trained on the latest available data from the database.
5. **Hugging Face Model Store:** * The trained model (`model.cbm`) and evaluation metrics (`metrics.json`) are pushed to the HF Hub.
* **Versioning:** Files are organized in versioned folders (e.g., `v1/`, `v2/`) to allow easy rollback and comparison.


6. **Inference:** The production environment pulls the specific versioned model from Hugging Face for real-time or batch predictions.

---

## Project Structure

```text
├── scraper/                 # Web scraping scripts
├── data_pipeline/           # Database connection and preprocessing logic
├── training/                # Model training and evaluation
├── api/                     # Prediction scripts (API)
├── .env.example             # Template for DB_URL and HF_TOKEN
├── .gitignore               # Optimized for Python/ML (ignores data and .env)
├── test.car_listings.sql    # Database schema (Tables, Indexes, Constraints)
└── requirements.txt         # Project dependencies
```

---

## Model Registry & Versioning

Instead of using DVC, this project leverages the **Hugging Face Hub** as a central artifact repository. Each training run generates a set of artifacts:

| Artifact | Description |
| --- | --- |
| `model.cbm` | The serialized CatBoost model. |
| `train_data.csv` | A snapshot of the training split used. |
| `test_data.csv` | A snapshot of the test split for reproducibility. |
| `metrics.json` | Detailed performance scores: **R², MAE, RMSE, MAPE, and Coverage.** |
| `shap_summary.png` | Feature importance visualization. |

> **Note:** Models are accessed via the `huggingface_hub` library, allowing the inference service to pull the latest "stable" tag or a specific version folder.

---

## Model Architecture: CatBoost & MultiQuantile Loss

The core of this pipeline is powered by **CatBoost**, a high-performance gradient boosting library specifically optimized for handling categorical features—making it the perfect fit for car data (Brand, Model, Fuel Type, etc.).

### 1. Why CatBoost?

* **Native Categorical Handling:** CatBoost processes categorical variables without requiring manual One-Hot Encoding or Label Encoding, preserving the relational information between car features.
* **Robustness:** It is less sensitive to hyperparameter tuning and naturally resists overfitting compared to other GBM frameworks.

### 2. Understanding MultiQuantile Loss

Unlike standard regression models that predict a single point estimate (the mean), this model uses **MultiQuantile Loss**. This allows the pipeline to output multiple values simultaneously:

* ** (Lower Bound):** The "bargain" price. Only 5% of similar cars are priced lower than this.
* ** (Median):** The most likely market price.
* ** (Upper Bound):** The "premium" price. 95% of similar cars are priced below this.

The model minimizes the **Pinball Loss** function for each quantile :

### 3. Business Value of Quantiles

In the car market, a single price is often misleading due to variations in vehicle condition or urgency of sale. By providing a **valuation interval** (e.g., $1.2M - $1.4M), we offer:

* **Confidence:** A measure of how certain the model is about the price.
* **Risk Assessment:** Identifying "outlier" listings that are priced far outside the predicted 90% interval ( to ).

---
## Execution Sequence

To run the pipeline from end-to-end, execute the scripts in the following order. Each script handles a specific stage of the data lifecycle:

### 1. Data Collection

```bash
python scraper/main.py
```

* **Action:** Triggers the web scraper to fetch the latest car advertisements.
* **Result:** Raw data is collected and prepared for staging.

### 2. Database Persistence & Deduplication

```bash
python pipeline/process_for_db.py
```

* **Action:** Checks the scraped data against the existing database records.
* **Result:** Only unique `ad_id`s are appended to the database. No data is overwritten.

### 3. Model Training & Evaluation

```bash
python pipeline/model_train.py
```

* **Action:** Preprocesses the data, trains the CatBoost model, and generates performance artifacts.
* **Result:** Creates `model.cbm`, `metrics.json`, and `shap_summary.png` locally.

### 4. Cloud Deployment (Hugging Face)

```bash
python pipeline/upload_to_server.py
```

* **Action:** Packages the training artifacts into a versioned folder.
* **Result:** Uploads the entire versioned bundle to the **Hugging Face Model Store**.
---

## ⚙️ Setup & Configuration

1. **Clone the repo:**
```bash
git clone https://github.com/sadik-coban/car-price-prediction-pipeline.git
```


2. **Environment Variables:**
Create a `.env` file from the example:
```bash
cp .env.example .env
# Add your Database URL, Hugging Face Write Token and Base URL of scraped website
```

---
### Environment Configuration

Create a `.env` file in the root directory and define the following variables:

```env
# Database Connection
DATABASE_URL=postgresql://user:password@localhost:5432/db_name

# Hugging Face API Access
HF_TOKEN=your_hugging_face_write_token_here

# Scraper Settings
BASE_URL=https://www.example.com
```
---
### Why this approach?

* **No Overwrites:** The Database logic ensures we don't lose old ad data if prices change; we keep the first instance of the ad for a cleaner "original price" baseline.
* **Decoupled Data:** GitHub stays light (source code only). Hugging Face handles the heavy lifting of model storage.
* **Transparency:** Metrics are versioned alongside the model, so you always know *why* a model is performing the way it is.
---