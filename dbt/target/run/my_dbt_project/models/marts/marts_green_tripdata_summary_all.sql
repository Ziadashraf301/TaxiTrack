
  
    

  create  table "ingest_db"."public"."green_tripdata_summary_all__dbt_tmp"
  
  
    as
  
  (
    


select
    
coalesce(
  case "VendorID"
    when 1 then 'Creative Mobile Technologies'
    when 2 then 'VeriFone Inc.'
    else 'Other'
  end,
  'Unknown'
)  as vendor_label,
    
  extract(month  from lpep_pickup_datetime) as trip_month,
  extract(quarter from lpep_pickup_datetime) as trip_quarter,
  extract(day    from lpep_pickup_datetime) as trip_day,
  extract(dow    from lpep_pickup_datetime) as trip_weekday,
  case when extract(dow from lpep_pickup_datetime) in (0,6) then 'Weekend' else 'Weekday' end as day_type,
  extract(hour   from lpep_pickup_datetime) as trip_hour
,
    
coalesce(
  case payment_type
    when 1 then 'Credit Card'
    when 2 then 'Cash'
    when 3 then 'No Charge'
    when 4 then 'Dispute'
    when 5 then 'Unknown'
    when 6 then 'Voided Trip'
    else 'Other'
  end,
  'Unknown'
)  as payment_type_label,
    
    case
      when avg(trip_distance) <= 2 then 'Short'
      when avg(trip_distance) <= 5 then 'Medium'
      else 'Long'
    end
 as trip_type,
    count(*) as trip_count,
    round(avg(trip_distance)::numeric,2) as avg_distance,
    round(sum(trip_distance)::numeric,2) as total_distance,
    round(avg(fare_amount)::numeric,2) as avg_fare,
    round(sum(fare_amount)::numeric,2) as total_fare,
    round(sum(tip_amount)::numeric,2) as total_tip,
    round(avg(tip_amount)::numeric,2) as avg_tip,
    round(avg(total_amount)::numeric,2) as avg_total,
    round(avg(passenger_count)::numeric,2) as avg_passenger_count,
    count(distinct "PULocationID") as unique_pickup_locations,
    min(lpep_pickup_datetime) as first_trip,
    max(lpep_pickup_datetime) as last_trip,

    -- Peak morning and evening hours using the hour extracted from pickup datetime
    count(*) filter(where extract(hour from lpep_pickup_datetime) between 6 and 9) as peak_morning_hours,
    count(*) filter(where extract(hour from lpep_pickup_datetime) between 17 and 19) as peak_evening_hours,
    
    round(
      max(
        case
          when lpep_dropoff_datetime is not null then
            extract(epoch from lpep_dropoff_datetime - lpep_pickup_datetime) / 60
          else null
        end
      ), 2
    )
 as trip_duration_minutes,
    -- Calculating average tip rate
    round(
    avg(
        case
            when fare_amount is null or fare_amount = 0 then null
            else cast(tip_amount as float) / fare_amount
        end
    )
::numeric, 2) as tip_rate

from "ingest_db"."public"."staging_green_tripdata_all"
where 
    trip_distance > 0
    AND fare_amount > 0
    AND passenger_count > 0
    AND EXTRACT(EPOCH FROM lpep_dropoff_datetime - lpep_pickup_datetime) > 60

group by vendor_label, trip_month, trip_quarter, trip_day,
         trip_weekday, day_type, payment_type_label, trip_hour
order by vendor_label, trip_month, trip_weekday, trip_hour

  );
  