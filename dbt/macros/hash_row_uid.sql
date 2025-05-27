{% macro generate_trip_uid(columns=[]) %}
    md5(
        concat_ws(
            '||',
            {% for col in columns -%}
                cast({{ col }} as text){% if not loop.last %}, {% endif %}
            {%- endfor %}
        )
    )
{% endmacro %}
