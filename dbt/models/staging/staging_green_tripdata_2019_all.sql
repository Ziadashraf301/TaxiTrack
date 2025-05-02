{{
  config(
    alias='staging_green_tripdata_2019_all'
  )
}}

with combined as (
  {{ union_raw_tables('2019', range(1,13)) }}
)

select * from combined