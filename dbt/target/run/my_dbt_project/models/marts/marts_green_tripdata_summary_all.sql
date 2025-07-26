
      
        
            delete from "ingest_db"."public"."green_tripdata_summary_all"
            where (
                trip_uid) in (
                select (trip_uid)
                from "green_tripdata_summary_all__dbt_tmp164426003910"
            );

        
    

    insert into "ingest_db"."public"."green_tripdata_summary_all" ("trip_uid", "vendor_label", "trip_month", "trip_quarter", "trip_day", "trip_weekday", "day_type", "trip_hour", "payment_type_label", "trip_type", "trip_count", "avg_distance", "total_distance", "avg_fare", "total_fare", "total_tip", "avg_tip", "avg_total", "avg_passenger_count", "unique_pickup_locations", "first_trip", "last_trip", "peak_morning_hours", "peak_evening_hours", "trip_duration_minutes", "tip_rate")
    (
        select "trip_uid", "vendor_label", "trip_month", "trip_quarter", "trip_day", "trip_weekday", "day_type", "trip_hour", "payment_type_label", "trip_type", "trip_count", "avg_distance", "total_distance", "avg_fare", "total_fare", "total_tip", "avg_tip", "avg_total", "avg_passenger_count", "unique_pickup_locations", "first_trip", "last_trip", "peak_morning_hours", "peak_evening_hours", "trip_duration_minutes", "tip_rate"
        from "green_tripdata_summary_all__dbt_tmp164426003910"
    )
  