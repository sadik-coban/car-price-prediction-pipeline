# BMW Price Prediction & Automated Pipeline

An end-to-end data science project designed to scrape, process, and predict BMW car prices in the Turkish market. This project demonstrates a full-stack ML approach from raw data extraction to automated model orchestration.

## Project Overview

I developed this project to bridge the gap between statistical theory and production-grade data engineering. The system automates the lifecycle of a machine learning model, ensuring data is always fresh and predictions are reliable.

### Key Features

* **Automated Scraping:** Custom-built scraper to collect real-time data from automotive portals.
* **Orchestration:** Using **Apache Airflow** to manage data pipelines and task scheduling.
* **Advanced Modeling:** Leveraging **CatBoost** for handling categorical features and providing high-accuracy price estimates.
* **Statistical Preprocessing:** Robust handling of missing values, outliers, and feature engineering based on automotive domain knowledge.

---

## 🛠 Tech Stack

* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** CatBoost
* **Workflow Management:** Apache Airflow
* **Experiment Tracking (Planned):** MLflow
* **Data Versioning:** DVC

---

## Project Architecture

1. **Ingestion:** Airflow triggers the scraper to fetch the latest BMW listings.
2. **Processing:** Raw data is cleaned, outliers (Z-score/IQR) are handled, and categorical variables are encoded.
3. **Training:** CatBoost model is trained on the processed dataset.
4. **Deployment (WIP):** Integrating the model into a web interface (Streamlit/FastAPI).

---

## Roadmap & Future Enhancements (Work in Progress)

This project is under active development. My next steps are:

* [ ] **MLflow Integration:** Implementing experiment tracking to monitor model versioning and metrics.
* [ ] **Model Deployment:** Deploying the predictor with Fast API and Docker.
* [ ] **System Optimization**: Refactoring the project into a modular, production-grade architecture and finalizing end-to-end Airflow DAG orchestration to ensure system reliability and scalability.
