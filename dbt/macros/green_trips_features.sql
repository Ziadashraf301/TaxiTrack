{% macro green_trips_features(table) %}
select
    *,
    toHour(pickup_datetime) as pickup_hour,
    toDayOfWeek(pickup_datetime) as pickup_day_of_week,
    dateDiff('second', pickup_datetime, dropoff_datetime)/60.0 as trip_duration_minutes
from {{ table }}
{% endmacro %}
