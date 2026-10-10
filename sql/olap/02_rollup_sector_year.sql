-- operation: ROLL-UP (rolled up: sector by year)
SELECT d.year, s.sector,
       ROUND(AVG(i.return_1d) * 100, 3) AS avg_daily_return_pct
FROM fact_indicator i
JOIN dim_date  d ON d.date_key  = i.date_key
JOIN dim_stock s ON s.stock_key = i.stock_key
WHERE s.instrument_type = 'EQUITY'
GROUP BY d.year, s.sector
ORDER BY d.year, s.sector;
