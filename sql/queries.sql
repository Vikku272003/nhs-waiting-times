-- NHS Waiting Times Analysis Queries

-- 1. Monthly trend
SELECT
    month,
    SUM(total_waiting) AS total_waiting,
    SUM(over_52_weeks) AS over_52_weeks,
    ROUND(AVG(pct_within_18_weeks), 1) AS avg_pct_within_18
FROM nhs_waiting_times
GROUP BY month
ORDER BY month;

-- 2. Worst specialties by wait time
SELECT
    specialty,
    ROUND(AVG(median_wait_weeks), 1) AS avg_median_wait_weeks,
    ROUND(AVG(pct_within_18_weeks), 1) AS avg_pct_within_18,
    SUM(over_52_weeks) AS total_over_52_weeks
FROM nhs_waiting_times
GROUP BY specialty
ORDER BY avg_median_wait_weeks DESC;

-- 3. Regional performance
SELECT
    region,
    ROUND(AVG(pct_within_18_weeks), 1) AS avg_pct_within_18,
    ROUND(AVG(median_wait_weeks), 1) AS avg_median_wait,
    SUM(total_waiting) AS total_patients
FROM nhs_waiting_times
GROUP BY region
ORDER BY avg_pct_within_18 ASC;

-- 4. COVID impact by period
SELECT
    CASE
        WHEN month < '2020-07-01' THEN '1. Pre-COVID'
        WHEN month < '2021-06-01' THEN '2. COVID Impact'
        WHEN month < '2023-01-01' THEN '3. Recovery'
        ELSE '4. Current'
    END AS period,
    ROUND(AVG(pct_within_18_weeks), 1) AS avg_pct_within_18,
    ROUND(AVG(median_wait_weeks), 1) AS avg_median_wait,
    SUM(over_52_weeks) AS total_over_52
FROM nhs_waiting_times
GROUP BY period
ORDER BY period;