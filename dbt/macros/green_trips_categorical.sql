{% macro green_trips_categorize(table) %}
select
    *,
    case vendor_id
        when 1 then 'Creative Mobile Technologies, LLC'
        when 2 then 'Curb Mobility, LLC'
        when 6 then 'Myle Technologies Inc'
        when 7 then 'Helix'
        else 'Unknown'
    end as vendor_name,
    
    case payment_type
        when 0 then 'Flex Fare Trip'
        when 1 then 'Credit Card'
        when 2 then 'Cash'
        when 3 then 'No Charge'
        when 4 then 'Dispute'
        when 5 then 'Unknown'
        when 6 then 'Voided Trip'
    end as payment_type_name,

    case trip_type
        when 1 then 'Street_hail'
        when 2 then 'Dispatch'
    end as trip_type_name,

    case rate_code_id
        when 1 then 'Standard Rate'
        when 2 then 'JFK'
        when 3 then 'Newark'
        when 4 then 'Nassau_Westchester'
        when 5 then 'Negotiated Fare'
        when 6 then 'Group Ride'
        else 'Unknown'
    end as rate_code_description,

    case store_and_forward_flag
        when 'Y' then 'Yes'
        when 'N' then 'No'
        else 'Unknown'
    end as store_and_forward_flag_description,

    'green_trip' as service_type
from {{ table }}
{% endmacro %}
