#!/bin/bash
set -e

# ==============================================================================
# Initialize Multiple PostgreSQL Databases on First Boot
# Creates isolated databases for Airflow, MLflow, and Metabase on a single Postgres instance
# ==============================================================================

# Refresh collation versions to match current OS glibc version if needed
psql -v ON_ERROR_STOP=0 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    ALTER DATABASE template1 REFRESH COLLATION VERSION;
    ALTER DATABASE postgres REFRESH COLLATION VERSION;
    ALTER DATABASE "$POSTGRES_DB" REFRESH COLLATION VERSION;
EOSQL

function create_user_and_database() {
    local database=$1
    local password=$2
    echo "Creating database '$database' with user '$database'..."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        DO \$\$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$database') THEN
                CREATE USER $database WITH ENCRYPTED PASSWORD '$password';
            END IF;
        END
        \$\$;
        SELECT 'CREATE DATABASE $database OWNER $database'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$database')\gexec
        GRANT ALL PRIVILEGES ON DATABASE $database TO $database;
EOSQL
}

# Create MLflow database and user
create_user_and_database "mlflow" "${MLFLOW_DB_PASSWORD:-mlflow123}"

# Create Metabase database and user
create_user_and_database "metabase" "${METABASE_DB_PASSWORD:-metabase123}"

echo "All additional databases initialized successfully."
