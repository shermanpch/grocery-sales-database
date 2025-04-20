from collections import Counter

import numpy as np
import pandas as pd
import math

from .feasy.decorator import multiple, single

# --- Define Reference Date ---
REFERENCE_DATE = pd.Timestamp("2018-05-10")


# --- Feature Engineering Functions ---
# == Basic Customer Info ==
@single("CustomerID", str)
def get_customer_id(row) -> str:
    """Gets the customer ID."""
    return row.CustomerID


@single("customer_city", str)
def get_customer_city(row) -> str:
    """Gets the customer's city name."""
    return row.CustomerCityName


@single("customer_country", str)
def get_customer_country(row) -> str:
    """Gets the customer's country name."""
    return row.CustomerCountryName

# Earth's radius in kilometers
EARTH_RADIUS_KM = 6371.0

# Reference coordinate (New York City)
REFERENCE_LAT = 40.7128
REFERENCE_LON = -74.0060

# Coordinates for each city (latitude, longitude)
city_coordinates = {
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
    "Philadelphia": (39.9526, -75.1652),
    "San Antonio": (29.4241, -98.4936),
    "San Diego": (32.7157, -117.1611),
    "Dallas": (32.7767, -96.7970),
    "San Jose": (37.3382, -121.8863),
    "Austin": (30.2672, -97.7431),
    "Jacksonville": (30.3322, -81.6557),
    "Fort Worth": (32.7555, -97.3308),
    "Columbus": (39.9612, -82.9988),
    "Charlotte": (35.2271, -80.8431),
    "San Francisco": (37.7749, -122.4194),
    "Indianapolis": (39.7684, -86.1581),
    "Seattle": (47.6062, -122.3321),
    "Denver": (39.7392, -104.9903),
    "Washington": (38.9072, -77.0369),
    "Boston": (42.3601, -71.0589),
    "El Paso": (31.7619, -106.4850),
    "Detroit": (42.3314, -83.0458),
    "Nashville": (36.1627, -86.7816),
    "Memphis": (35.1495, -90.0490),
    "Portland": (45.5152, -122.6784),
    "Oklahoma": (35.4676, -97.5164),
    "Las Vegas": (36.1699, -115.1398),
    "Louisville": (38.2527, -85.7585),
    "Baltimore": (39.2904, -76.6122),
    "Milwaukee": (43.0389, -87.9065),
    "Albuquerque": (35.0844, -106.6504),
    "Tucson": (32.2226, -110.9747),
    "Fresno": (36.7378, -119.7871),
    "Sacramento": (38.5816, -121.4944),
    "Mesa": (33.4152, -111.8315),
    "Kansas": (39.1142, -94.6275),
    "Atlanta": (33.7490, -84.3880),
    "Colorado": (39.5501, -105.7821),
    "Miami": (25.7617, -80.1918),
    "Tulsa": (36.1540, -95.9928),
    "Oakland": (37.8044, -122.2712),
    "Minneapolis": (44.9778, -93.2650),
    "Arlington": (32.7357, -97.1081),
    "Newark": (40.7357, -74.1724),
    "Anchorage": (61.2181, -149.9003),
    "Honolulu": (21.3069, -157.8583),
    "Wichita": (37.6872, -97.3301),
    "Cleveland": (41.4993, -81.6944),
    "Tampa": (27.9506, -82.4572),
    "Bakersfield": (35.3733, -119.0187),
    "Aurora": (39.7294, -104.8319),
    "St. Louis": (38.6270, -90.1994),
    "Raleigh": (35.7796, -78.6382),
    "Pittsburgh": (40.4406, -79.9959),
    "St. Paul": (44.9537, -93.0900),
    "Cincinnati": (39.1031, -84.5120),
    "Greensboro": (36.0726, -79.7920),
    "Toledo": (41.6528, -83.5379),
    "New Orleans": (29.9511, -90.0715),
    "Lincoln": (40.8136, -96.7026),
    "Buffalo": (42.8864, -78.8784),
    "Norfolk": (36.8508, -76.2859),
    "Rochester": (43.1566, -77.6088),
    "Jersey": (40.7282, -74.0776),
    "Mobile": (30.6954, -88.0399),
    "Little Rock": (34.7465, -92.2896),
    "Dayton": (39.7589, -84.1916),
    "Shreveport": (32.5252, -93.7502),
    "Madison": (43.0731, -89.4012),
    "Richmond": (37.5407, -77.4360),
    "Glendale": (33.5387, -112.1860),
    "Yonkers": (40.9312, -73.8988),
    "Hialeah": (25.8576, -80.2781),
    "Fort Wayne": (41.0793, -85.1394),
    "Garland": (32.9126, -96.6389),
    "Lubbock": (33.5779, -101.8552),
    "Fremont": (37.5483, -121.9886),
    "Santa Ana": (33.7455, -117.8677),
    "Stockton": (37.9577, -121.2908),
    "Des Moines": (41.5868, -93.6250),
    "Anaheim": (33.8353, -117.9145),
    "Birmingham": (33.5437, -86.7796),
    "Montgomery": (32.3617, -86.2792),
    "Grand Rapids": (42.9638, -85.6700),
    "Spokane": (47.6597, -117.4291),
    "Baton Rouge": (30.4712, -91.1474),
    "Jackson": (32.2989, -90.1847),
    "Akron": (41.0818, -81.5115),
    "Tacoma": (47.2458, -122.4594),
    "Virginia Beach": (36.8631, -76.0158),
    "Corpus Christi": (27.8006, -97.3964),
    "Long Beach": (33.7701, -118.1937),
    "Riverside": (33.9534, -117.3962),
    "Omaha": (41.2572, -95.9951),
    "St. Petersburg": (27.7731, -82.6400)
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes the great-circle distance between two points (lat1, lon1) and (lat2, lon2)
    using the Haversine formula.
    """
    # Convert degrees to radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c

@single("distance_from_nyc_km", float)
def get_distance_from_nyc(row) -> float:
    """
    Computes the great-circle distance in km between New York City and the customer's city.
    Uses the Haversine formula. Returns NaN if the city is not in the coordinate list.
    """
    city = getattr(row, "CustomerCityName", None)
    if city not in city_coordinates:
        return np.nan

    lat2, lon2 = city_coordinates[city]
    return haversine_distance(REFERENCE_LAT, REFERENCE_LON, lat2, lon2)


# == Monetary Features ==
@single("total_spend", float)
def get_total_spend(row) -> float:
    """Calculates the total amount spent by the customer."""
    if not hasattr(row, "TotalPrice") or len(row.TotalPrice) == 0:
        return 0.0
    return float(np.sum(row.TotalPrice))


@single("avg_spend_per_transaction", float)
def get_avg_spend_per_transaction(row) -> float:
    """Calculates the average amount spent per transaction."""
    num_transactions = len(row.SalesID) if hasattr(row, "SalesID") else 0
    if num_transactions == 0:
        return 0.0
    total_spend = get_total_spend(row)[-1]
    return float(total_spend / num_transactions)


@single("max_spend_single_transaction", float)
def get_max_spend_single_transaction(row) -> float:
    """Finds the maximum amount spent in a single transaction."""
    if not hasattr(row, "TotalPrice") or len(row.TotalPrice) == 0:
        return 0.0
    return float(np.max(row.TotalPrice))


@single("min_spend_single_transaction", float)
def get_min_spend_single_transaction(row) -> float:
    """Finds the minimum amount spent in a single transaction."""
    if not hasattr(row, "TotalPrice") or len(row.TotalPrice) == 0:
        return 0.0
    meaningful_spends = [p for p in row.TotalPrice if p > 0]
    if not meaningful_spends:
        return 0.0
    return float(np.min(meaningful_spends))


# == Frequency & Recency Features ==
@single("num_transactions", int)
def get_num_transactions(row) -> int:
    """Counts the total number of transactions."""
    if not hasattr(row, "SalesID"):
        return 0
    return len(row.SalesID)


@single("last_purchase_date", "datetime64[ns]")
def get_last_purchase_date(row):
    """Finds the date of the last purchase."""
    if not hasattr(row, "SalesDate") or len(row.SalesDate) == 0:
        return pd.NaT
    try:
        dates = pd.to_datetime(row.SalesDate, errors="coerce")
        valid_dates = dates.dropna()
        if valid_dates.empty:
            return pd.NaT
        return valid_dates.max()
    except Exception as e:
        raise e


@single("first_purchase_date", "datetime64[ns]")
def get_first_purchase_date(row):
    """Finds the date of the first purchase."""
    if not hasattr(row, "SalesDate") or len(row.SalesDate) == 0:
        return pd.NaT
    try:
        dates = pd.to_datetime(row.SalesDate, errors="coerce")
        valid_dates = dates.dropna()
        if valid_dates.empty:
            return pd.NaT
        return valid_dates.min()
    except Exception as e:
        raise e


@single("days_since_last_purchase", float)
def get_days_since_last_purchase(row) -> float:
    """
    Calculates the number of days between the reference date and the customer's last purchase.
    Lower values indicate more recent purchases (higher recency).
    Returns NaN if no valid purchase date is found.
    """
    if not hasattr(row, "SalesDate") or len(row.SalesDate) == 0:
        return np.nan
    try:
        dates = pd.to_datetime(row.SalesDate, errors="coerce")
        valid_dates = dates.dropna()
        if valid_dates.empty:
            return np.nan

        last_date = valid_dates.max()
        delta = REFERENCE_DATE - last_date
        return float(delta.total_seconds() / (24 * 60 * 60))
    except Exception as e:
        raise e


@single("days_since_first_purchase", float)
def get_days_since_first_purchase(row) -> float:
    """
    Calculates the number of days between the reference date and the customer's first purchase.
    Represents the time elapsed since the customer's first interaction (relative to the reference date).
    Returns NaN if no valid purchase date is found.
    """
    if not hasattr(row, "SalesDate") or len(row.SalesDate) == 0:
        return np.nan
    try:
        dates = pd.to_datetime(row.SalesDate, errors="coerce")
        valid_dates = dates.dropna()
        if valid_dates.empty:
            return np.nan

        first_date = valid_dates.min()
        delta = REFERENCE_DATE - first_date
        return float(delta.total_seconds() / (24 * 60 * 60))
    except Exception as e:
        raise


@single("purchase_period_days", float)
def get_purchase_period_days(row) -> float:
    """Calculates the period in days between the first and last purchase."""
    last_date = get_last_purchase_date(row)[-1]
    first_date = get_first_purchase_date(row)[-1]
    if pd.isna(last_date) or pd.isna(first_date) or last_date == first_date:
        return 0.0
    return float((last_date - first_date).days)


@single("avg_days_between_purchases", float)
def get_avg_days_between_purchases(row) -> float:
    """Calculates the average number of days between consecutive purchases."""
    if not hasattr(row, "SalesDate") or len(row.SalesDate) < 2:
        return 0.0
    try:
        dates = pd.to_datetime(row.SalesDate, errors="coerce").dropna().sort_values()
        if len(dates) < 2:
            return 0.0
        diffs = dates.diff().total_seconds() / (24 * 60 * 60)
        avg_diff = np.mean(diffs[1:])
        return float(avg_diff) if not np.isnan(avg_diff) else 0.0
    except Exception as e:
        raise e


# == Product Preference Features ==
@single("num_unique_products", int)
def get_num_unique_products(row) -> int:
    """Counts the number of unique products purchased."""
    if not hasattr(row, "ProductID"):
        return 0
    return len(set(row.ProductID))


@single("num_unique_categories", int)
def get_num_unique_categories(row) -> int:
    """Counts the number of unique product categories purchased."""
    if not hasattr(row, "CategoryID"):
        return 0
    return len(set(row.CategoryID))


@single("most_frequent_category_id", str)
def get_most_frequent_category_id(row) -> str:
    """Finds the most frequently purchased product category ID."""
    if not hasattr(row, "CategoryID") or len(row.CategoryID) == 0:
        return -1
    try:
        category_counts = Counter(row.CategoryID)
        most_common = category_counts.most_common(1)
        return str(most_common[0][0]) if most_common else -1
    except Exception as e:
        raise e


@single("avg_product_price", float)
def get_avg_product_price(row) -> float:
    """Calculates the average price of products purchased (based on unit price)."""
    if not hasattr(row, "Price") or len(row.Price) == 0:
        return 0.0
    positive_prices = [p for p in row.Price if p > 0]
    if not positive_prices:
        return 0.0
    return float(np.mean(positive_prices))


@single("total_quantity", int)
def get_total_quantity(row) -> int:
    """Calculates the total quantity of items purchased."""
    if not hasattr(row, "Quantity") or len(row.Quantity) == 0:
        return 0
    return int(np.sum(row.Quantity))


@single("proportion_high_class", float)
def get_proportion_high_class(row) -> float:
    """Calculates the proportion of purchased items belonging to the 'High' class."""
    if not hasattr(row, "Class") or len(row.Class) == 0:
        return 0.0
    try:
        class_counts = Counter(row.Class)
        total_items = len(row.Class)
        high_count = class_counts.get("High", 0)
        return float(high_count / total_items) if total_items > 0 else 0.0
    except Exception as e:
        raise e


@single("proportion_medium_class", float)
def get_proportion_medium_class(row) -> float:
    """Calculates the proportion of purchased items belonging to the 'Medium' class."""
    if not hasattr(row, "Class") or len(row.Class) == 0:
        return 0.0
    try:
        class_counts = Counter(row.Class)
        total_items = len(row.Class)
        medium_count = class_counts.get("Medium", 0)
        return float(medium_count / total_items) if total_items > 0 else 0.0
    except Exception as e:
        raise e


@single("proportion_low_class", float)
def get_proportion_low_class(row) -> float:
    """Calculates the proportion of purchased items belonging to the 'Low' class."""
    if not hasattr(row, "Class") or len(row.Class) == 0:
        return 0.0
    try:
        class_counts = Counter(row.Class)
        total_items = len(row.Class)
        low_count = class_counts.get("Low", 0)
        return float(low_count / total_items) if total_items > 0 else 0.0
    except Exception as e:
        raise e


@single("proportion_unknown_class", float)
def get_proportion_unknown_class(row) -> float:
    """Calculates the proportion of purchased items belonging to an 'Unknown' or other class."""
    if not hasattr(row, "Class") or len(row.Class) == 0:
        return 0.0
    try:
        class_counts = Counter(row.Class)
        total_items = len(row.Class)
        known_classes = {"High", "Medium", "Low"}
        unknown_count = sum(
            count for cls, count in class_counts.items() if cls not in known_classes
        )
        return float(unknown_count / total_items) if total_items > 0 else 0.0
    except Exception as e:
        raise e


# == Discount Features ==
@single("avg_discount_rate", float)
def get_avg_discount_rate(row) -> float:
    """Calculates the average discount rate across all transactions."""
    if not hasattr(row, "Discount") or len(row.Discount) == 0:
        return 0.0
    return float(np.mean(row.Discount))


@single("proportion_discounted_transactions", float)
def get_proportion_discounted_transactions(row) -> float:
    """Calculates the proportion of transactions where a discount was applied."""
    num_transactions = len(row.SalesID) if hasattr(row, "SalesID") else 0
    if num_transactions == 0 or not hasattr(row, "Discount"):
        return 0.0
    discounted_count = sum(1 for d in row.Discount if d > 0)
    return float(discounted_count / num_transactions)


@single("total_discount_value", float)
def get_total_discount_value(row) -> float:
    """Estimates the total monetary value of discounts received."""
    total_discount = 0.0
    if (
        not hasattr(row, "Quantity")
        or not hasattr(row, "Price")
        or not hasattr(row, "Discount")
        or len(row.Quantity) != len(row.Price)
        or len(row.Quantity) != len(row.Discount)
    ):
        return 0.0

    for q, p, d in zip(row.Quantity, row.Price, row.Discount):
        if q > 0 and p > 0 and d > 0:
            total_discount += q * p * d

    return float(total_discount)


# == Other Product Characteristic Features ==
@single("proportion_allergic_items", float)
def get_proportion_allergic_items(row) -> float:
    """Calculates the proportion of purchased items marked as potentially allergic."""
    num_transactions = len(row.SalesID) if hasattr(row, "SalesID") else 0
    if num_transactions == 0 or not hasattr(row, "IsAllergic"):
        return 0.0
    allergic_count = sum(
        1 for a in row.IsAllergic if isinstance(a, str) and a.lower() == "true"
    )
    return float(allergic_count / num_transactions)


@single("avg_vitality_days", float)
def get_avg_vitality_days(row) -> float:
    """Calculates the average vitality days of products purchased."""
    if not hasattr(row, "VitalityDays") or len(row.VitalityDays) == 0:
        return 0.0
    valid_vitality = [
        v for v in row.VitalityDays if isinstance(v, (int, float)) and v >= 0
    ]
    if not valid_vitality:
        return 0.0
    return float(np.mean(valid_vitality))
