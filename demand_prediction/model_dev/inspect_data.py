from data_loader import ClickHouseDataLoader
from config import CLICKHOUSE_CONFIG

def inspect_data():
    """Inspect the raw data structure"""
    print("🔍 Inspecting ClickHouse Data...")
    
    loader = ClickHouseDataLoader()
    
    try:
        # Fetch a small sample
        df = loader.fetch_timeseries_data(
            table_name=CLICKHOUSE_CONFIG["default_table"],
            start_date="2019-01-01",
            end_date="2020-04-30"  # Just 3 days for inspection
        )
        
        print("✅ Data fetched successfully!")
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        print(f"\nData Types:")
        print(df.dtypes)
        
        print(f"\nBasic Statistics:")
        print(df.describe())
        
        print(f"\nGroup Combinations (first 10):")
        groups = df[['pickup_zone', 'pickup_borough', 'service_type']].drop_duplicates()
        print(groups.head(10))
        
        print(f"\nDate Range:")
        print(f"Min date: {df['pickup_date'].min()}")
        print(f"Max date: {df['pickup_date'].max()}")
        print(f"Hour range: {df['pickup_hour'].min()} to {df['pickup_hour'].max()}")
        
        print(f"\nMissing Values:")
        print(df.isnull().sum())
        
    except Exception as e:
        print(f"❌ Inspection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_data()