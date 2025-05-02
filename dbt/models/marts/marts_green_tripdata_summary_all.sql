{{
  config(
    alias='green_tripdata_summary_all',
    materialized='table'
  )
}}

{{ generate_tripdata_summary('staging_green_tripdata_all') }}
