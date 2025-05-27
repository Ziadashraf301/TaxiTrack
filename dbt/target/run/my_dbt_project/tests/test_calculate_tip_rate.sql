select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      -- tests/test_calculate_tip_rate.sql

with test_data as (
    select 5 as tip_amount, 10 as fare_amount, 0.5 as expected_tip_rate
    union all
    select 3, 0, NULL
    union all
    select NULL, 15, NULL
),

actual as (
    select
        *,
        
    avg(
        case
            when fare_amount is null or fare_amount = 0 then null
            else cast(tip_amount as float) / fare_amount
        end
    )
 as actual_tip_rate
    from test_data
)

select *
from actual
where not (
    (actual_tip_rate = expected_tip_rate)
    or (actual_tip_rate is null and expected_tip_rate is null)
)
      
    ) dbt_internal_test