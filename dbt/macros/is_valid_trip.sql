{% macro is_valid_trip() %}
    trip_distance > 0
    AND fare_amount > 0
    AND passenger_count > 0
    AND EXTRACT(EPOCH FROM lpep_dropoff_datetime - lpep_pickup_datetime) > 60
{% endmacro %}