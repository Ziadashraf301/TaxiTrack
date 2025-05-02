{% macro union_raw_tables(year, months) %}
{% set sql_parts = [] %}
{% for m in months %}
  {{ sql_parts.append(
     "select * from " ~ source('raw', 'green_tripdata_' ~ year ~ '-' ~ '%02d'|format(m))
  ) }}
{% endfor %}
{{ return(sql_parts | join(' union all ')) }}
{% endmacro %}