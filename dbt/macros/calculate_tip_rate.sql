{% macro calculate_tip_rate(tip_column, fare_column) %}
    avg(
        case
            when {{ fare_column }} is null or {{ fare_column }} = 0 then null
            else cast({{ tip_column }} as float) / {{ fare_column }}
        end
    )
{% endmacro %}
