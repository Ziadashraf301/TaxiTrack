{{
  config(
    alias='green_tripdata_summary_2019',
    materialized='table'
  )
}}

{{ generate_tripdata_summary('staging_green_tripdata_2019_all') }}
