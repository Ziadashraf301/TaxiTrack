{% macro vendor_label(column) %}
coalesce(
  case "{{ column }}"
    when 1 then 'Creative Mobile Technologies'
    when 2 then 'VeriFone Inc.'
    else 'Other'
  end,
  'Unknown'
) {% endmacro %}