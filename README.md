# TaxiTrack

TaxiTrack is an advanced data pipeline and visualization platform designed for managing and analyzing taxi-related data. Leveraging tools like Apache Airflow, PostgreSQL, Metabase, and dbt, along with machine learning models, this project facilitates data ingestion, storage, transformation, prediction, and visualization.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Machine Learning](#machine-learning)
- [Getting Started](#getting-started)
- [Services](#services)
- [Usage](#usage)

## Overview

The project is built to provide a comprehensive data management and machine learning solution for taxi data. It includes:
- A robust ETL (Extract, Transform, Load) pipeline using Apache Airflow.
- Data storage in PostgreSQL.
- Data transformations using dbt.
- Machine learning models for predictive analysis.
- Interactive dashboards and analytics with Metabase.

## Features

- **Data Ingestion**: Automated data ingestion using Apache Airflow.
- **Data Transformation**: Transform raw data into meaningful insights with dbt.
- **Data Visualization**: Create dashboards and visualizations with Metabase.
- **Machine Learning**: Predict and analyze trends with machine learning models.
- **Database Management**: Manage PostgreSQL databases with pgAdmin.

## Architecture

The project uses Docker Compose to orchestrate the following services:
- **PostgreSQL**: Primary database for Airflow and ingested data.
- **pgAdmin**: PostgreSQL database management.
- **Apache Airflow**: Workflow orchestration tool for ETL processes.
- **Metabase**: Data visualization and dashboarding platform.
- **dbt**: Data transformation tool for analytics engineering.

## Machine Learning

### Overview

The machine learning component of this project is designed to provide predictive insights from the taxi data. It includes:
- Predictive modeling for ride demand and supply.
- Analysis of factors such as location, time, and weather affecting taxi demand.
- Models implemented using Python libraries such as `scikit-learn` and `pandas`.

### Steps

1. **Data Preparation**: Data is ingested and cleaned using Apache Airflow and dbt.
2. **Feature Engineering**: Key features are extracted and engineered for model training.
3. **Model Training**: Machine learning models are trained on historical data to predict taxi demand and other metrics.
4. **Model Deployment**: Models are integrated into the pipeline to generate real-time predictions.

### Example Models

- **Demand Prediction Model**: Uses historical data to predict the number of rides requested at a given time and location.
- **Driver Allocation Model**: Predicts the optimal allocation of drivers to meet demand efficiently.

## Getting Started

### Prerequisites

- Ensure you have Docker and Docker Compose installed on your machine.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ziadashraf301/TaxiTrack.git
   cd TaxiTrack
   ```

2. Set up the environment variables:
   - Update the `.env` file with your configuration (if required).

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Access the services:
   - Airflow Webserver: [http://localhost:8080](http://localhost:8080) (Username: `admin`, Password: `admin`)
   - pgAdmin: [http://localhost:5050](http://localhost:5050) (Email: `admin@admin.com`, Password: `admin`)
   - Metabase: [http://localhost:3000](http://localhost:3000)

## Services

- **PostgreSQL**:
  - Hosts the Airflow metadata database and ingested data.
  - Ports:
    - `5432`: Airflow database
    - `5433`: Ingested data database
- **pgAdmin**:
  - Web-based PostgreSQL management tool.
  - Port: `5050`
- **Metabase**:
  - Data visualization and dashboarding.
  - Port: `3000`
- **Airflow**:
  - Workflow orchestration.
  - Ports:
    - Webserver: `8080`
- **dbt**:
  - Data transformation tool for data engineering.

## Usage

1. Add your DAGs (Directed Acyclic Graphs) to the `dags/` directory.
2. Configure your dbt models in the `dbt/` directory.
3. Use Metabase to create interactive dashboards and visualizations.
4. Add machine learning models in the `ml/` directory and integrate them into the pipeline.
