# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.stats import mstats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from notebooks.utils.db import init_db
from notebooks.utils.data_cleaning import clean_data

# Initialize database connection
con_raw = init_db()
con = clean_data(con_raw)
#%% [markdown]
# Now we're going to start creating some features based on the individual tables, without any joins. 
# 
#  

df_bookings_raw = con.execute("select * from bookings").fetchdf()

def add_booking_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out['arrival_date'] = pd.to_datetime(out['arrival_date']).dt.date
    out['departure_date'] = pd.to_datetime(out['departure_date']).dt.date
    out['created_at'] = pd.to_datetime(out['created_at'])

    # Core
    out['booking_length'] = (pd.to_datetime(out['departure_date']) -
                             pd.to_datetime(out['arrival_date'])).dt.days
    # Convert arrival_date to datetime for comparison with created_at (which is already datetime)
    out['lead_time_days'] = (pd.to_datetime(out['arrival_date']) -
                             out['created_at']).dt.days

    # Temporal signals
    out['arrival_month'] = pd.to_datetime(out['arrival_date']).dt.month
    out['arrival_dow'] = pd.to_datetime(out['arrival_date']).dt.dayofweek
    out['booking_hour'] = out['created_at'].dt.hour

    return out

df_bookings = add_booking_features(df_bookings_raw)
#%%

df_booked_rooms_raw = con.execute("select * from booked_rooms").fetchdf()
def add_booked_room_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out['guests_total'] = out['total_adult'] + out['total_children']
    out['is_family'] = (out['total_children'].fillna(0) > 0).astype(int)

    # --- Quartile-based room size buckets -------------------------------
    # Drop missing values before computing quantiles to avoid NaN cut thresholds
    qs = out['room_size'].dropna().quantile([0, 0.25, 0.50, 0.75, 1.0]).tolist()

    # Ensure strictly increasing boundaries (pandas quirk)
    # If many values are the same, quantiles may collapse → force uniqueness
    qs = np.unique(qs)

    # If uniqueness collapses buckets (rare), fall back to standard bins
    if len(qs) < 5:
        # fallback (avoids failure)
        qs = np.linspace(out['room_size'].min(), out['room_size'].max(), 5)

    out['room_size_bucket'] = pd.cut(
        out['room_size'],
        bins=qs,
        labels=['Q1', 'Q2', 'Q3', 'Q4'],
        include_lowest=True
    )
    # ---------------------------------------------------------------------

    if 'nights_in_stay' in out.columns:
        out['price_per_night_room'] = (
            out['booked_room_total_price'] /
            out['nights_in_stay'].replace(0, np.nan)
        )

    out['price_per_guest'] = (
        out.get('price_per_night_room', np.nan) /
        out['guests_total'].replace(0, np.nan)
    )

    return out

df_booked_rooms = add_booked_room_features(df_booked_rooms_raw)
df_booked_rooms.head()
#%%
df_rooms_raw = con.execute("select * from rooms").fetchdf()
def add_room_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out['log_number_of_rooms'] = np.log1p(out['number_of_rooms'])

    out['capacity_bucket'] = pd.cut(
        out['max_occupancy'],
        bins=[0,1,2,4,999],
        labels=['1','2','3-4','5+'],
        include_lowest=True
    )

    # Optional PCA on structural attributes
    numeric = out[['number_of_rooms','max_occupancy','max_adults']].fillna(0)
    pca = PCA(n_components=2).fit(numeric)
    out[['room_pca1','room_pca2']] = pca.transform(numeric)

    return out

df_rooms = add_room_features(df_rooms_raw)
df_rooms.head()































# # %%
# # Step 1: Feature Engineering & Data Loading (Room-Night Granularity)

# room_nights_query = """
# WITH br AS (
#     SELECT
#         br.id AS booked_room_id,
#         br.booking_id,
#         br.room_id,
#         br.total_adult,
#         br.total_children,
#         br.room_size,
#         br.room_view,
#         br.room_type,
#         br.total_price AS booked_room_total_price,
#         -- Cast to DATE to ensure date math works for expansion
#         CAST(b.arrival_date AS DATE) AS arrival_date,
#         CAST(b.departure_date AS DATE) AS departure_date,
#         b.status AS booking_status,
#         b.total_price AS booking_total_price,
#         b.created_at,
#         b.payment_method,
#         b.source,
#         b.cancelled_by,
#         b.hotel_id
#     FROM booked_rooms br
#     JOIN bookings b
#       ON CAST(br.booking_id AS BIGINT) = b.id
# ),
# expanded AS (
#     SELECT
#         *,
#         (departure_date - arrival_date) AS nights_in_stay
#     FROM br
#     WHERE (departure_date - arrival_date) > 0  -- Ensure positive stay
# ),
# room_nights AS (
#     SELECT
#         e.*,
#         -- This 'stay_date' is the specific night the guest is staying
#         night_date AS stay_date,
#         row_number() OVER (
#             PARTITION BY e.booked_room_id ORDER BY night_date
#         ) AS night_index
#     FROM expanded e
#     -- Explode the date range into individual rows per day
#     -- e.g., Arr: Jan 1, Dep: Jan 3 (2 nights) -> Generates Jan 1, Jan 2
#     CROSS JOIN UNNEST(
#         GENERATE_SERIES(e.arrival_date, e.departure_date - INTERVAL 1 DAY, INTERVAL 1 DAY)
#     ) AS t(night_date)
# ),
# with_room_features AS (
#     SELECT
#         rn.*,
#         rn.stay_date AS date, -- Keep 'date' alias as requested, mirroring stay_date
#         r.number_of_rooms,
#         r.max_occupancy,
#         r.max_adults,
#         r.events_allowed,
#         r.pets_allowed,
#         r.smoking_allowed,
#         r.children_allowed
#     FROM room_nights rn
#     LEFT JOIN rooms r
#       ON rn.room_id = r.id
# ),
# final AS (
#     SELECT
#         wrf.*,
#         hl.id AS hotel_location_id,
#         hl.address,
#         hl.city,
#         hl.zip,
#         hl.country,
#         hl.latitude,
#         hl.longitude,
#         -- Calculate daily price: Total / Nights
#         CASE
#             WHEN nights_in_stay > 0 THEN booked_room_total_price / nights_in_stay
#             ELSE NULL
#         END AS price_per_night
#     FROM with_room_features wrf
#     LEFT JOIN hotel_location hl
#       ON wrf.hotel_id = hl.hotel_id
# )
# SELECT *
# FROM final
# ORDER BY booked_room_id, stay_date;
# """

# print("Executing Room-Night expansion query...")
# df_room_nights = con.execute(room_nights_query).fetchdf()

# # Ensure stay_date is datetime for pandas operations
# df_room_nights['stay_date'] = pd.to_datetime(df_room_nights['stay_date'])
# df_room_nights['created_at'] = pd.to_datetime(df_room_nights['created_at'])

# print(f"Data Loaded. Total Room-Nights: {len(df_room_nights)}")
# df_room_nights.head()
# # %%

# %%
