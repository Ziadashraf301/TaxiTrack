
  create view "ingest_db"."public"."staging_green_tripdata_2019_all__dbt_tmp"
    
    
  as (
    

with combined as (
  select * from "ingest_db"."public"."green_tripdata_2019-01" union all select * from "ingest_db"."public"."green_tripdata_2019-02" union all select * from "ingest_db"."public"."green_tripdata_2019-03" union all select * from "ingest_db"."public"."green_tripdata_2019-04" union all select * from "ingest_db"."public"."green_tripdata_2019-05" union all select * from "ingest_db"."public"."green_tripdata_2019-06" union all select * from "ingest_db"."public"."green_tripdata_2019-07" union all select * from "ingest_db"."public"."green_tripdata_2019-08" union all select * from "ingest_db"."public"."green_tripdata_2019-09" union all select * from "ingest_db"."public"."green_tripdata_2019-10" union all select * from "ingest_db"."public"."green_tripdata_2019-11" union all select * from "ingest_db"."public"."green_tripdata_2019-12"
)

select * from combined
  );