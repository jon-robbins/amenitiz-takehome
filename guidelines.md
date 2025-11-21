# Data Science Guidelines - Amenitiz Project

## 1. Data Philosophy
- **Profile before you Filter:** Never drop "bad data" without first counting it and visualizing it. We must understand the *magnitude* of an error before fixing it.
- **Calculated Fields:**
  - `Daily Price` = `total_price` / `(departure - arrival)`. Never use raw `total_price` for analysis.
  - `Lead Time` = `arrival_date` - `created_at`.
  - `Occupancy` = `rooms_sold` / `total_capacity`.

## 2. Technical Standards
- **DuckDB First:** Use SQL for heavy lifting (aggregations, joins, date math). Use Pandas only for plotting and final formatting.
- **Type Safety:** The raw CSVs contain `NULL` strings and mixed types.
  - Always assume columns might be strings unless the `init_db` explicitly cast them.
  - Use `TRY_CAST(col AS TYPE)` in SQL if unsure.
  - Use `NULLIF(col, 'NULL')` to handle string nulls.

## 3. Reproducibility
- All plots must be saved to the `output/` directory.
- Plots must have:
  - A descriptive Title.
  - Labels for X and Y axes (with units, e.g., "Price (€)").
  - A Legend if multiple groups are shown.

## 4. Specific Known Issues (Do not fix yet, just monitor)
- **Linkage:** A large portion of `rooms` do not link to `bookings`. We profile this split, we do not ignore it.
- **Overbooking:** Occupancy > 100% exists. We track the frequency of this, we do not blindly delete it yet.