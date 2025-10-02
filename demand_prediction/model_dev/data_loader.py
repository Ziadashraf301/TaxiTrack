import pandas as pd
from clickhouse_connect import get_client
from config import DATA_CONFIG,PATH_CONFIG,CLICKHOUSE_CONFIG
from logger import setup_logger
from pathlib import Path

log_dir = Path(PATH_CONFIG["logs_dir"])
logger = setup_logger(__name__, f"{log_dir}/pipeline.log")

class ClickHouseDataLoader:
    def __init__(self):
        logger.info("Initializing ClickHouseDataLoader")
        self.data_config = DATA_CONFIG
        self.connection_params = CLICKHOUSE_CONFIG
        self.chunk_size = 500000
        self.encoder_dir = PATH_CONFIG["encoder_dir"]
        self.min_hours = 16068
        self.Training = True
        
    def connect_to_db(self):
        """Establish connection to ClickHouse database"""
        try:
                client = get_client(**self.connection_params)
                logger.info("Successfully connected to ClickHouse")
                return client

        except Exception as e:
            logger.error(f"Error connecting to ClickHouse: {e}")
            raise
    
    def fetch_timeseries_data(self, table_name, start_date=None, end_date=None, groups=None):
        """Fetch time series data from ClickHouse - ONLY ESSENTIAL COLUMNS, optionally filter by groups"""
        try:
            client = self.connect_to_db()
            
            # Base query - only needed columns
            base_query = f"""
                SELECT 
                    pickup_date,
                    pickup_hour, 
                    pickup_zone,
                    pickup_borough,
                    service_type,
                    total_trips  
                FROM {table_name}
            """
            
            # Add date filtering if provided
            where_conditions = []
            if start_date:
                where_conditions.append(f"pickup_date >= '{start_date}'")
            if end_date:
                where_conditions.append(f"pickup_date <= '{end_date}'")
            
            # Add group filtering if provided
            if groups:
                group_conditions = []
                for g in groups:
                    conditions = []
                    if "pickup_borough" in g:
                        conditions.append(f"pickup_borough = '{g['pickup_borough']}'")
                    if "pickup_zone" in g:
                        conditions.append(f"pickup_zone = '{g['pickup_zone']}'")
                    if "service_type" in g:
                        conditions.append(f"service_type = '{g['service_type']}'")
                    if conditions:
                        group_conditions.append("(" + " AND ".join(conditions) + ")")
                if group_conditions:
                    where_conditions.append("(" + " OR ".join(group_conditions) + ")")
            
            if where_conditions:
                base_query += " WHERE " + " AND ".join(where_conditions)
            
            base_query += " ORDER BY pickup_zone, pickup_borough, service_type, pickup_date, pickup_hour"
            
            # Fetch data in chunks
            df = pd.DataFrame()
            offset = 0
            
            while True:
                chunk_query = f"{base_query} LIMIT {self.chunk_size} OFFSET {offset}"
                chunk = client.query_df(chunk_query)
                
                if chunk.empty:
                    break
                    
                df = pd.concat([df, chunk], ignore_index=True)
                logger.info(f"Fetched {len(chunk)} rows (total: {len(df)})")
                offset += self.chunk_size
            
            client.close()
            logger.info(f"Total data fetched: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def filter_low_volume_groups(self, df, min_hours = None):
        """
        Filter out groups with insufficient total trips.
        Keeps only groups that have at least min_total_trips across the entire period.
        
        Args:
            df: DataFrame with columns [pickup_zone, pickup_borough, service_type, total_trips]
            min_hours: Minimum sum of hours required to keep a group (default: 16068)
        
        Returns:
            Filtered DataFrame containing only high-volume groups
        """
        group_cols = ['pickup_zone', 'pickup_borough', 'service_type']
        target_col = 'total_trips'

        if min_hours is None:
            min_hours = self.min_hours
        
        logger.info(f"Filtering groups with minimum {min_hours} hours")
        
        # Calculate total trips per group
        group_totals = df.groupby(group_cols)[target_col].count().reset_index()
        group_totals.columns = group_cols + ['total_hours']
        
        # Filter groups that meet minimum threshold
        valid_groups = group_totals[group_totals['total_hours'] >= min_hours]
        
        logger.info(f"Groups before filtering: {len(group_totals)}")
        logger.info(f"Groups after filtering: {len(valid_groups)}")
        logger.info(f"Removed {len(group_totals) - len(valid_groups)} low-volume groups")
        
        # Merge to keep only valid groups
        df_filtered = df.merge(
            valid_groups[group_cols],
            on=group_cols,
            how='inner'
        )
        
        logger.info(f"Rows before filtering: {len(df)}")
        logger.info(f"Rows after filtering: {len(df_filtered)}")
        
        # Log top groups by volume
        top_10 = valid_groups.nlargest(10, 'total_hours')
        logger.info("Top 10 groups by volume:")
        for _, row in top_10.iterrows():
            logger.info(f"  {row['pickup_zone']} | {row['pickup_borough']} | {row['service_type']} | {row['total_hours']:,} hours")
        
        return df_filtered

    def combine_group_columns(self, df):
        """
        Combine pickup_zone, pickup_borough, and service_type into a single 'group_id' column.
        Format: zone_borough_servicetype (e.g., "JFK_Queens_yellow")
        Drops the original three columns to save memory.
        """
        logger.info("Combining group columns into single 'group_id'")
        
        # Create combined group identifier
        df['group_id'] = (
            df['pickup_zone'].astype(str) + '_' + 
            df['pickup_borough'].astype(str) + '_' + 
            df['service_type'].astype(str)
        )
        
        # Clean up: remove spaces and special characters
        df['group_id'] = df['group_id'].str.replace(' ', '_').str.replace('/', '_')
        
        # Drop original columns
        df = df.drop(columns=['pickup_zone', 'pickup_borough', 'service_type'])
        
        logger.info(f"Created {df['group_id'].nunique()} unique group IDs")
        logger.info(f"Sample group IDs: {df['group_id'].unique()[:5].tolist()}")
        
        return df
    
    def merge_date_hour(self, df):
        """Merge pickup_date and pickup_hour to create proper datetime"""
        df['pickup_date'] = pd.to_datetime(df['pickup_date'])
        df['pickup_datetime'] = df['pickup_date'] + pd.to_timedelta(df['pickup_hour'], unit='h')
        df[self.data_config['timestamp_col']] = df['pickup_datetime']
        
        # Drop original columns
        df = df.drop(columns=['pickup_date', 'pickup_hour'])
        return df
    
    def validate_data(self, df):
        """Validate data quality and structure"""
        # Update to use group_id instead of separate group columns
        required_cols = ['group_id', self.data_config["timestamp_col"], self.data_config["target_col"]]
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Successfully Data Validation")
        return True
    
    def interpolate_missing_hours_per_group(self, df):
        """Interpolate missing hours for each group separately - ONLY TARGET VARIABLE"""
        timestamp_col = self.data_config["timestamp_col"]
        group_col = self.data_config["group_col"]
        target_col = self.data_config["target_col"]
        frequency = self.data_config["frequency"]
        
        # Create complete hourly range for the entire dataset
        min_date = df[timestamp_col].min()
        max_date = df[timestamp_col].max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq=frequency)
        
        # Get all unique groups
        unique_groups = df[group_col].unique()
        
        # Create complete DataFrame with all combinations
        complete_data = []
        
        for group_id in unique_groups:
            group_df = pd.DataFrame({
                timestamp_col: full_date_range,
                group_col: group_id
            })
            complete_data.append(group_df)
        
        complete_df = pd.concat(complete_data, ignore_index=True)
        
        # Merge with original data to fill in existing values
        merged_df = complete_df.merge(
            df, 
            on=[timestamp_col, group_col], 
            how='left',
            suffixes=('', '_original')
        )
        
        # ONLY INTERPOLATE TARGET VARIABLE (total_trips)
        def interpolate_group(group):
            group = group.sort_values(timestamp_col)
            
            # Interpolate only the target variable
            group[target_col] = group[target_col].fillna(group[target_col].ewm(span=24, adjust=False).mean())
            
            # Fill remaining NaN with ffill, then bfill then 0 for target variable
            group[target_col] = group[target_col].ffill().fillna(0).astype('int')

            return group
        
        # Apply interpolation to each group
        interpolated_df = merged_df.groupby(group_col).apply(
            interpolate_group
        ).reset_index(drop=True)
        
        logger.info(f"Successfully Data interpolation")
        logger.info(f"Data after interpolation: {len(interpolated_df)} rows")
        logger.info(f"Number of groups: {len(unique_groups)}")
        logger.info(f"Hours per group: {len(full_date_range)}")        
        return interpolated_df
    
    def get_processed_data(self, table_name, start_date=None, end_date=None, groups=None, min_hours=16068):
        """
        Main method to get processed and interpolated data
        
        Args:
            table_name: Name of the table to query
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            groups: Specific groups to filter (optional)
            min_total_trips: Minimum total trips threshold for filtering groups (default: 16068)
        
        Returns:
            Processed and interpolated DataFrame with only high-volume groups
            Columns: [pickup_datetime, group_id, total_trips, pickup_date, pickup_hour]
        """
        # Fetch raw data (only essential columns)
        df = self.fetch_timeseries_data(table_name=table_name, start_date=start_date, end_date=end_date, groups=groups)
        
        if self.Training:
            # Filter low-volume groups BEFORE any processing
            df = self.filter_low_volume_groups(df, min_hours=min_hours)
        
        # Combine group columns into single column (before merge_date_hour)
        df = self.combine_group_columns(df)
        
        # Merge date and hour
        df = self.merge_date_hour(df)
        
        # Validate data
        self.validate_data(df)
        
        # Interpolate missing hours for each group
        df_processed = self.interpolate_missing_hours_per_group(df)
        

        return df_processed