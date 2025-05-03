## 🛺 **TaxiTrack**

**TaxiTrack** is a modern data platform designed to manage, transform, and analyze NYC taxi trip data. It offers real-time demand prediction and intuitive business insights through dashboards and a web app.

![NYC Taxi](https://media.posterlounge.com/img/products/760000/755512/755512_poster.jpg)

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Architecture](#architecture)
* [Machine Learning](#machine-learning)
* [Getting Started](#getting-started)
* [Services](#services)
* [Usage](#usage)

---

## Overview

**TaxiTrack** is a full-stack data pipeline built using Python, offering a seamless flow from data ingestion to machine learning predictions and business insights. The platform processes NYC taxi data, transforming it into actionable insights and predictions for ride demand.

### 🔧 **Tech Stack**

* **Python**: Data ingestion, machine learning, logging, and deployment
* **PostgreSQL**: Data storage (raw and transformed)
* **dbt**: SQL-based transformation and testing
* **Metabase**: Dashboard for metrics like trip volume and revenue
* **Streamlit**: Web app for ML-powered demand forecasting
* **Docker**: Containerizes services like PostgreSQL, Metabase, and dbt
* **Git & GitHub**: Version control and collaboration
* **Python Logging**: Tracks ETL jobs and model performance

---

## Features

* **Data Ingestion**: Downloads and stores NYC green taxi data in PostgreSQL
* **Data Transformation**: Cleans and aggregates data using dbt
* **Built-in dbt Testing**: Ensures data integrity
* **Interactive Web App**: Predicts future ride demand via Streamlit
* **Real-Time Dashboards**: Visualizes business metrics with Metabase
* **Version Control**: Uses Git for reproducible development
* **Logging**: Monitors pipeline health and performance

---

## Architecture

The following components are orchestrated with **Docker Compose**:

![TaxiTrack Pipeline](sandbox:/mnt/data/A_flowchart_diagram_in_this_digital_vector_illustr.png)

* **PostgreSQL**: Stores raw and modeled data
* **dbt**: Data transformation and testing
* **Metabase**: Dashboards for business insights
* **Streamlit**: Hosts the demand prediction model
* **Ingestion Script**: Fetches and stores data
* **Logging**: Tracks the status of the pipeline

---

## Machine Learning

### Overview

The ML component forecasts taxi demand using time-series data:

* Predicts ride volume by time and other factors
* Built with `pandas`, `scikit-learn`, `xgboost`, and `joblib`

### Pipeline

1. **Data Preparation**: Cleansed data from dbt for training
2. **Feature Engineering**: Extracts time-based features
3. **Model Training**: Trains and validates on historical data
4. **Model Persistence**: Saves model with `joblib`
5. **Streamlit Deployment**: Exposes a UI for predictions

---

## Getting Started

### Prerequisites

* Python ≥ 3.8
* Docker & Docker Compose

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Ziadashraf301/TaxiTrack.git
   cd TaxiTrack
   ```

2. Set up environment variables if required (`.env`)

3. Build and start services:

   ```bash
   docker-compose up -d
   ```

4. Run the ingestion script manually or schedule it:

   ```bash
      python ingestion/main.py --user USER --password PASSWORD --host HOST --port PORT --db DB --file_name FILE_NAME
   ```

5. Run dbt transformations:

   ```bash
   docker exec -it dbt bash       
   dbt run
   ```

6. Start the Streamlit app:

   ```bash
   streamlit run Taxi_demand_prediction/taxi_app/taxi_prediction_app.py
   ```

---

## Services

| Service       | Description                          | Port |
| ------------- | ------------------------------------ | ---- |
| PostgreSQL    | Stores raw and transformed trip data | 5432 |
| Metabase      | Dashboards and data exploration      | 3000 |
| Streamlit     | ML prediction interface              | 8501 |
| pgAdmin (opt) | PostgreSQL admin GUI (if enabled)    | 5050 |

---

## Usage

* Ingest new data into PostgreSQL with the ingestion script
* Run dbt models to transform and validate data
* Create and modify dashboards in Metabase
* Train and deploy new ML models as needed
* Interact with predictions via the Streamlit app

---
