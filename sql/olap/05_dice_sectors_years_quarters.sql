-- operation: DICE (restrict several dimensions at once)
-- param: sector_a = Banking
-- param: sector_b = IT
-- param: year_from = 2023
-- param: year_to = 2025
-- param: quarter_a = 1
-- param: quarter_b = 2
SELECT d.year, d.quarter, s.sector,
       ROUND(AVG(i.return_1d) * 100, 3) AS avg_daily_return_pct
FROM fact_indicator i
JOIN dim_date  d ON d.date_key  = i.date_key
JOIN dim_stock s ON s.stock_key = i.stock_key
WHERE s.sector IN (:sector_a, :sector_b)
  AND d.year BETWEEN :year_from AND :year_to
  AND d.quarter IN (:quarter_a, :quarter_b)
GROUP BY d.year, d.quarter, s.sector
ORDER BY d.year, d.quarter, s.sector;
