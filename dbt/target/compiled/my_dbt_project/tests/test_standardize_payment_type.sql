-- tests/test_standardize_payment_type.sql

with test_data as (
    select 1 as payment_type, 'Credit Card' as expected
    union all
    select 3, 'No Charge'
    union all
    select 6, 'Voided Trip'
),

actual as (
    select
        *,
        
coalesce(
  case payment_type
    when 1 then 'Credit Card'
    when 2 then 'Cash'
    when 3 then 'No Charge'
    when 4 then 'Dispute'
    when 5 then 'Unknown'
    when 6 then 'Voided Trip'
    else 'Other'
  end,
  'Unknown'
)  as actual
    from test_data
)

select *
from actual
where not (
    actual = expected
    or (actual is null and expected is null)
)