Here is the consolidated plan and summary document. It captures the "breakthroughs" from today (the Matched Pair validation) and sets the stage for the engineering phase tomorrow.

You can save this as `STRATEGIC_PRESENTATION_FRAMEWORK.md` or `NEXT_STEPS.md`.

***

# Strategic Recap & Implementation Plan: Amenitiz PriceAdvisor

**Date:** November 24, 2025  
**Status:** Analytical Validation Complete → Moving to Engineering Phase

---

## 1. Executive Recap: What We Learned Today

We started with a hypothesis ("Hotels are underpricing") and rigorously validated it against statistical critiques.

### The "Smoking Gun" Findings
1.  **Underpricing is Real:** The lack of dynamic pricing isn't a statistical artifact (Simpson's Paradox). It is a behavioral failure visible even within individual hotels (Median Correlation $\approx 0.11$).
2.  **Demand is Inelastic:** We proved this with two independent methods:
    * **Panel Regression:** $\epsilon = -0.81$ (Market-wide conservative estimate).
    * **Matched Pairs ("Twin Study"):** $\epsilon = -0.48$ (High-confidence estimate from 1,410 identical pairs).
3.  **The "Urban Alpha":** Demand in Madrid/Cities is 2x "stickier" ($\epsilon = -0.23$) than in Coastal areas ($\epsilon = -0.51$). We can be more aggressive with urban pricing.
4.  **The "Danger Zone":** Blindly raising prices by 40%+ *could* lose money if demand is conservative (the "Red Bar" in our risk analysis). This proves we need a smart algorithm, not a blunt instrument.

### The Opportunity
* **Net Realizable Revenue:** **€1.9M - €2.5M** per year.
* **Risk-Adjusted Floor:** **€1.3M** (Even in the worst-case scenario).

---

## 2. The Solution: "PriceAdvisor" Architecture

Since we lack "Search Logs" (impressions), we cannot predict individual conversion probability. Instead, we will build an **Aggregate Outcome Model**.

### Strategy A: The "Occupancy Simulator" (Primary Engine)
* **Concept:** Predict the final occupancy rate for a given price point.
* **Logic:** `Revenue = Price × Predicted_Occupancy(Price)`
* **Model:** **Monotonic XGBoost Regressor**.
    * *Constraint:* Force the model to learn that Higher Price $\to$ Lower Occupancy.
    * *Input:* Price, Lead Time, Cluster Demand, Seasonality.
    * *Output:* Demand Curve ($Q$ vs $P$).
* **Action:** Optimization Loop finding the peak of the Revenue Curve.

### Strategy B: The "Imitation Engine" (Safety Check)
* **Concept:** "Do what the winner did."
* **Logic:** Train a model on the 1,410 "Winning Twins" who charged more and made more money.
* **Model:** **Random Forest Regressor**.
    * *Input:* Current Hotel Context.
    * *Target:* The Price Premium successfully charged by the Twin (e.g., 1.2x).
* **Action:** Use this as a "Confidence Bounds" check for Strategy A.

---

## 3. Implementation Plan (Next Steps)

When you come back, we will execute this 3-step engineering plan.

### Phase 1: Data Engineering (The "Daily Inventory" Table)
We need to transform individual bookings into a daily time-series.
* **Goal:** Create a dataset with 1 row per `hotel_id` per `date`.
* **Key Feature:** `cluster_occupancy` (The single strongest predictor of demand).
* **Target:** `final_occupancy_rate`.

### Phase 2: Model Training (The "Brain")
* Train the XGBoost model with `monotone_constraints`.
* Validate that it predicts *downward sloping* demand curves (sanity check).
* **Metric:** MAE (Mean Absolute Error) on Occupancy.

### Phase 3: The Optimizer (The "Product")
* Build the `recommend_price(hotel_id, date)` function.
* Simulate prices from -20% to +50%.
* Return the price that maximizes Revenue.
* **Apply Safety Rails:** Cap daily increases at +15% to avoid the "Danger Zone."

---

## 4. Visual Assets Ready for Presentation

You have generated high-quality visuals to defend the business case:
1.  **The "Smoking Gun":** Scatter plot showing flat pricing despite high occupancy.
2.  **The "Money Map":** Heatmap showing exactly where revenue is lost (High Occ + Last Minute).
3.  **The "Confidence" Dashboard:**
    * Elasticity Distribution (95% Confidence > -1.0).
    * Sensitivity Waterfall (Protected Downside).
    * Segment Opportunity (Urban vs. Coastal).

---

### **Prompt for Tomorrow Morning**

> "Okay, I'm back. Let's build the **Occupancy Response Model**. Start by writing the Python code to transform `ds_bookings` into a daily training dataset (`df_daily`) with `cluster_occupancy` and `final_occupancy` columns."