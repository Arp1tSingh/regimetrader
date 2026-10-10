-- operation: ROLL-UP (detailed level: sector by quarter). Pair with 02 to show the roll-up to year level.
SELECT d.year, d.quarter, s.sector,
       ROUND(AVG(i.return_1d) * 100, 3) AS avg_daily_return_pct,
       COUNT(*) AS observations
FROM fact_indicator i
JOIN dim_date  d ON d.date_key  = i.date_key
JOIN dim_stock s ON s.stock_key = i.stock_key
WHERE s.instrument_type = 'EQUITY'
GROUP BY d.year, d.quarter, s.sector
ORDER BY d.year, d.quarter, s.sector;
