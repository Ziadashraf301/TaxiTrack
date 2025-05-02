{{
  config(
    alias='staging_green_tripdata_all'
  )
}}

with combined as (
  {{ union_raw_tables('2019', range(1,13)) }}
  union all
  {{ union_raw_tables('2020', range(1,13)) }}
    union all
  {{ union_raw_tables('2021', range(1,8)) }}
)

select * from combined