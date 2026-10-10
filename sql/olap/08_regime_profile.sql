-- operation: MINING-LINKED OLAP (what the market looks like in each regime)
-- Requires fact_indicator rows for the INDEX instrument (indicators are computed for EQUITY and INDEX).
SELECT r.regime_label,
       COUNT(*) AS days,
       ROUND(AVG(i.return_1d) * 100, 3) AS avg_index_return_pct,
       ROUND(AVG(m.mkt_vol_20d) * 100, 1) AS avg_annual_vol_pct
FROM fact_market_regime m
JOIN dim_regime r ON r.regime_key = m.regime_key
JOIN fact_indicator i ON i.date_key = m.date_key
JOIN dim_stock s ON s.stock_key = i.stock_key AND s.instrument_type = 'INDEX'
GROUP BY r.regime_label
ORDER BY avg_index_return_pct DESC;
