
  create view "ingest_db"."public"."staging_green_tripdata_all__dbt_tmp"
    
    
  as (
    

with combined as (
  select * from "ingest_db"."public"."green_tripdata_2019-01" union all select * from "ingest_db"."public"."green_tripdata_2019-02" union all select * from "ingest_db"."public"."green_tripdata_2019-03" union all select * from "ingest_db"."public"."green_tripdata_2019-04" union all select * from "ingest_db"."public"."green_tripdata_2019-05" union all select * from "ingest_db"."public"."green_tripdata_2019-06" union all select * from "ingest_db"."public"."green_tripdata_2019-07" union all select * from "ingest_db"."public"."green_tripdata_2019-08" union all select * from "ingest_db"."public"."green_tripdata_2019-09" union all select * from "ingest_db"."public"."green_tripdata_2019-10" union all select * from "ingest_db"."public"."green_tripdata_2019-11" union all select * from "ingest_db"."public"."green_tripdata_2019-12"
  union all
  select * from "ingest_db"."public"."green_tripdata_2020-01" union all select * from "ingest_db"."public"."green_tripdata_2020-02" union all select * from "ingest_db"."public"."green_tripdata_2020-03" union all select * from "ingest_db"."public"."green_tripdata_2020-04" union all select * from "ingest_db"."public"."green_tripdata_2020-05" union all select * from "ingest_db"."public"."green_tripdata_2020-06" union all select * from "ingest_db"."public"."green_tripdata_2020-07" union all select * from "ingest_db"."public"."green_tripdata_2020-08" union all select * from "ingest_db"."public"."green_tripdata_2020-09" union all select * from "ingest_db"."public"."green_tripdata_2020-10" union all select * from "ingest_db"."public"."green_tripdata_2020-11" union all select * from "ingest_db"."public"."green_tripdata_2020-12"
    union all
  select * from "ingest_db"."public"."green_tripdata_2021-01" union all select * from "ingest_db"."public"."green_tripdata_2021-02" union all select * from "ingest_db"."public"."green_tripdata_2021-03" union all select * from "ingest_db"."public"."green_tripdata_2021-04" union all select * from "ingest_db"."public"."green_tripdata_2021-05" union all select * from "ingest_db"."public"."green_tripdata_2021-06" union all select * from "ingest_db"."public"."green_tripdata_2021-07"
)

select * from combined
  );