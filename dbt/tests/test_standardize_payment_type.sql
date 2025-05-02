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
        {{ payment_label('payment_type') }} as actual
    from test_data
)

select *
from actual
where not (
    actual = expected
    or (actual is null and expected is null)
)
