-- operation: SLICE (fix one dimension: the year)
-- param: year = 2022
SELECT s.sector,
       ROUND(AVG(i.return_1d) * 100, 3) AS avg_daily_return_pct,
       SUM(p.volume) AS total_volume
FROM fact_indicator i
JOIN fact_price p ON p.date_key = i.date_key AND p.stock_key = i.stock_key
JOIN dim_date  d ON d.date_key  = i.date_key
JOIN dim_stock s ON s.stock_key = i.stock_key
WHERE d.year = :year AND s.instrument_type = 'EQUITY'
GROUP BY s.sector
ORDER BY avg_daily_return_pct DESC;
