{% macro extract_date_parts(ts_column) %}
  extract(month  from {{ ts_column }}) as trip_month,
  extract(quarter from {{ ts_column }}) as trip_quarter,
  extract(day    from {{ ts_column }}) as trip_day,
  extract(dow    from {{ ts_column }}) as trip_weekday,
  case when extract(dow from {{ ts_column }}) in (0,6) then 'Weekend' else 'Weekday' end as day_type,
  extract(hour   from {{ ts_column }}) as trip_hour
{% endmacro %}