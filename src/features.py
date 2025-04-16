from collections import Counter

import numpy as np
import pandas as pd

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
