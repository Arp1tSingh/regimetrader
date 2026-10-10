-- operation: MINING-LINKED OLAP (does the classifier work differently across regimes?)
-- Only the primary strategy run writes signal/target_weight, so this describes the selected model.
-- param: model_name = rf
SELECT r.regime_label,
       COUNT(*) AS signals,
       ROUND(AVG(CASE WHEN f.next_day_return > 0 THEN 1.0 ELSE 0.0 END), 3) AS hit_rate,
       ROUND(AVG(f.next_day_return) * 100, 3) AS avg_next_day_return_pct
FROM fact_signal f
JOIN dim_regime r ON r.regime_key = f.regime_key
JOIN dim_model  m ON m.model_key  = f.model_key
WHERE f.signal = 1 AND f.next_day_return IS NOT NULL AND m.model_name = :model_name
GROUP BY r.regime_label;
