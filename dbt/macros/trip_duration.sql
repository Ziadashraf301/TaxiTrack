{% macro max_trip_duration(pickup_col, dropoff_col) %}
    round(
      max(
        case
          when {{ dropoff_col }} is not null then
            extract(epoch from {{ dropoff_col }} - {{ pickup_col }}) / 60
          else null
        end
      ), 2
    )
{% endmacro %}
