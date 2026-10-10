-- operation: PIVOT (sectors as rows, years as columns)
-- The year columns are literal. src.olap.run regenerates this statement from olap.pivot_years in config
-- (function build_pivot_sql); this file is the default for 2022-2024 and the template for that function.
SELECT s.sector,
       ROUND(AVG(CASE WHEN d.year = 2022 THEN i.return_1d END) * 100, 3) AS y2022,
       ROUND(AVG(CASE WHEN d.year = 2023 THEN i.return_1d END) * 100, 3) AS y2023,
       ROUND(AVG(CASE WHEN d.year = 2024 THEN i.return_1d END) * 100, 3) AS y2024
FROM fact_indicator i
JOIN dim_date  d ON d.date_key  = i.date_key
JOIN dim_stock s ON s.stock_key = i.stock_key
WHERE s.instrument_type = 'EQUITY'
GROUP BY s.sector
ORDER BY s.sector;
