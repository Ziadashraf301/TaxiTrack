{{
    config(
        materialized='incremental',
        incremental_strategy='append',
        unique_key=['pickup_date', 'borough', 'service_type']
    )
}}

-- ================================================================
-- Business Analytics Mart: Daily Taxi Performance Metrics
-- Provides daily KPIs by borough and service type
-- Tracks revenue, trips, passengers, and build timestamp
-- ================================================================

with base as (
    select
        pickup_datetime,
        service_type,
        fare_amount,
        tip_amount,
        congestion_surcharge,
        passenger_count,
        pickup_borough as borough
    from {{ ref('stg_all_trips') }}
),

aggregated as (
    select
        toDate(pickup_datetime) as pickup_date,
        borough,
        service_type,
        count(*) as num_trips,
        sum(passenger_count) as total_passengers,
        sum(fare_amount) as total_fare,
        sum(tip_amount) as total_tips,
        sum(congestion_surcharge) as total_congestion_fees,
        sum(fare_amount + tip_amount + congestion_surcharge) as total_revenue,
        now() as dbt_mart_build  
    from base
    group by
        pickup_date,
        borough,
        service_type
)

select * from aggregated
