-- operation: DRILL-DOWN (sector -> individual stocks and months)
-- param: sector = IT
-- param: year = 2025
SELECT d.year, d.month, s.symbol,
       ROUND(AVG(i.return_1d) * 100, 3) AS avg_daily_return_pct
FROM fact_indicator i
JOIN dim_date  d ON d.date_key  = i.date_key
JOIN dim_stock s ON s.stock_key = i.stock_key
WHERE s.sector = :sector AND d.year = :year
GROUP BY d.year, d.month, s.symbol
ORDER BY d.month, s.symbol;
