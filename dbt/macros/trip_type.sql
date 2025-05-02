{% macro classify_trip_distance(distance_column) %}
    case
      when avg({{ distance_column }}) <= 2 then 'Short'
      when avg({{ distance_column }}) <= 5 then 'Medium'
      else 'Long'
    end
{% endmacro %}
