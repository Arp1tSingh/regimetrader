-- operation: WINDOW FUNCTION (top 5 momentum stocks on one date)
-- param: as_of_date = 2025-06-30
SELECT *
FROM (
    SELECT d.full_date, s.symbol, i.return_20d,
           RANK() OVER (PARTITION BY d.full_date ORDER BY i.return_20d DESC) AS momentum_rank
    FROM fact_indicator i
    JOIN dim_date  d ON d.date_key  = i.date_key
    JOIN dim_stock s ON s.stock_key = i.stock_key
    WHERE s.instrument_type = 'EQUITY' AND d.full_date = :as_of_date
) ranked
WHERE momentum_rank <= 5;
