# Worked examples for exam and viva practice

Taken from the original blueprint (files `04` and `06`). Every number here is also an automated test fixture in `docs/spec/TESTS.md`, so the code is checked against the same answers.

---

## 1. K-Means: assign a day to a regime

K-Means assigns a day to the nearest cluster centre. Two standardized features (z-scores of 20-day return and volatility), three centres:

| Cluster | Centre (return z, volatility z) |
|---|---|
| Bull | (1.0, -0.7) |
| Sideways | (0.0, -0.3) |
| Bear | (-1.4, 1.5) |

A new day has z-scores (1.2, -0.6).

- Distance to Bull = sqrt((1.2 - 1.0)^2 + (-0.6 + 0.7)^2) = sqrt(0.04 + 0.01) = sqrt(0.05) = **0.224**
- Distance to Sideways = sqrt((1.2 - 0.0)^2 + (-0.6 + 0.3)^2) = sqrt(1.44 + 0.09) = sqrt(1.53) = **1.237**
- Distance to Bear = sqrt((1.2 + 1.4)^2 + (-0.6 - 1.5)^2) = sqrt(6.76 + 4.41) = sqrt(11.17) = **3.342**

The nearest centre is Bull, so the day is labelled **BULL**. Standardizing first matters: without it, a feature with large numeric values would dominate the distance.

---

## 2. Classifier evaluation from a confusion matrix

|  | Predicted up | Predicted down |
|---|---|---|
| **Actually up** | TP | FN |
| **Actually down** | FP | TN |

Out-of-sample results on 1,000 predictions: TP = 280, FP = 230, FN = 210, TN = 280.

- Accuracy = (280 + 280) / 1000 = **0.560**
- Precision = 280 / (280 + 230) = 280 / 510 = **0.549**
- Recall = 280 / (280 + 210) = 280 / 490 = **0.571**
- F1 = 2 x 0.549 x 0.571 / (0.549 + 0.571) = **0.560**

Actual up days = TP + FN = 490, so 49% of days are up and 51% are down. The majority-class baseline (always predict "down") scores **0.51**. The model's 0.56 is 5 percentage points above that, a realistic good result for daily direction. Always report the baseline next to the accuracy, because accuracy alone hides how easy the problem is.

---

## 3. Association rules (Apriori): support, confidence, lift

Each week is a "transaction" containing the sectors that rose. Example with 10 weeks: Banking up in 6 weeks, IT up in 5 weeks, both up in 4 weeks.

- Support (Banking and IT) = 4 / 10 = **0.40**
- Confidence (Banking implies IT) = 4 / 6 = **0.667**
- Lift = confidence / support(IT) = 0.667 / 0.5 = **1.33**

A lift above 1 means the two sectors rise together more often than chance.

---

## 4. Why costs and holding period matter

A model is right 55% of the time and every trade wins or loses exactly 1%.

- Expected gross return per trade = 0.55 x 1% - 0.45 x 1% = **0.10%**
- Round-trip cost = **0.30%**
- Expected net return per trade = 0.10% - 0.30% = **-0.20%**

A 55% hit rate loses money after costs. That is why the strategy uses hysteresis, a rebalance band and regime filters to cut turnover, and why results must be shown **after** costs.

### Cost of one trade

Buy 100 shares quoted at an open of 1,000 (slippage 0.05%, all-in cost 0.10%).

- Fill price = 1,000 x (1 + 0.0005) = **1,000.50**
- Trade value = 100 x 1,000.50 = **100,050.00**
- Cost = 0.10% x 100,050 = **100.05**
- Cash out = 100,050.00 + 100.05 = **100,150.05**

A full round trip (buy and sell) costs about 2 x (0.10% + 0.05%) = **0.30%** of trade value.

---

## 5. Sharpe ratio and maximum drawdown

**Sharpe:** daily mean excess return = 0.05%, daily standard deviation = 0.9%.

- Sharpe = sqrt(252) x 0.0005 / 0.009 = 15.875 x 0.05556 = **0.88** (0.8819 unrounded)

**Maximum drawdown:** equity path 100, 120, 90, 110.

- Peak = 120, trough after the peak = 90
- Drawdown = (90 - 120) / 120 = **-25%**

---

## 6. One day of the bot (illustrative)

Regime BULL (entry 0.55, 10 slots, 10% each). Held 8 stocks; one now has `prob_up` 0.47, below the 0.50 exit level, so it is dropped and 7 are kept. Free slots = 10 - 7 = 3, filled by the 3 highest-probability stocks not held, each at or above 0.55. Orders for tomorrow: SELL 1, BUY 3, each target 10% of equity (subject to the 25% rebalance band). If the regime had switched to BEAR, the target would be empty and all 7 positions would be sold at the next open.
