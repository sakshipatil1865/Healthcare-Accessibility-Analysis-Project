
# ==============================================================
# ZoyaSofts Internship Project
# LOCAL HEALTHCARE ACCESSIBILITY ANALYSIS
# Tools: Python, Pandas, NumPy, Matplotlib
# ==============================================================

from pathlib import Path
import re
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------
# 1. FILE PATHS AND SETTINGS
# --------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "Local_healthcare.csv"

OUTPUT_DIR = BASE_DIR / "healthcare_outputs"
CHART_DIR = OUTPUT_DIR / "Python_Charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "Healthcare_Analysis_Report.txt"
CLEANED_FILE = OUTPUT_DIR / "Healthcare_Cleaned_Data.csv"

# --------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------

def print_heading(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_chart(filename):
    """Save the current Matplotlib chart as a PNG image."""
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print("Chart saved:", filename)


def get_required_column(data, column_name):
    """Return a column name ignoring capitalization and outer spaces."""
    for col in data.columns:
        if str(col).strip().lower() == column_name.lower():
            return col
    return None


def safe_filename(text):
    """Create a simple filename-safe string."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_")


# Capture printed results so a text report can also be created.
import contextlib
import io

report_buffer = io.StringIO()

# --------------------------------------------------------------
# MAIN ANALYSIS
# --------------------------------------------------------------

with contextlib.redirect_stdout(report_buffer):

    print_heading("ZOYASOFTS INTERNSHIP PROJECT")
    print("LOCAL HEALTHCARE ACCESSIBILITY ANALYSIS")
    print("Tools: Python, Pandas, NumPy, Matplotlib")

    # ----------------------------------------------------------
    # 2. DATA LOADING
    # ----------------------------------------------------------

    print_heading("1. DATA LOADING")

    if not FILE_PATH.exists():
        raise FileNotFoundError(
            "\nCSV file not found!\n"
            f"Expected location: {FILE_PATH}\n"
            "Place Local_healthcare.csv in the same folder as this Python file."
        )

    try:
        df = pd.read_csv(FILE_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(FILE_PATH, encoding="cp1252")

    # Clean column headers.
    df.columns = [str(col).strip() for col in df.columns]

    print("Input file:", FILE_PATH)
    print("Original records:", len(df))
    print("Original columns:", len(df.columns))
    print("Dataset loaded successfully.")

    # ----------------------------------------------------------
    # 3. CHECK REQUIRED COLUMNS
    # ----------------------------------------------------------

    print_heading("2. COLUMN VALIDATION")

    required_columns = [
        "Sr_no",
        "Facility_Type",
        "Facility_Name",
        "Area",
        "Speciality",
        "Opening_hours",
        "Public_rating",
        "Review_count",
        "Publicly_listed_consultation_range",
        "Emergency_availability"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required column(s) missing from CSV: "
            + ", ".join(missing_columns)
            + "\nPlease check the CSV header names."
        )

    print("All 10 required columns are available.")

    # ----------------------------------------------------------
    # 4. DATASET INFORMATION
    # ----------------------------------------------------------

    print_heading("3. DATASET INFORMATION")

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nDataset information:")
    df.info()

    print("\nDataset shape:", df.shape)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # ----------------------------------------------------------
    # 5. DATA CLEANING
    # ----------------------------------------------------------

    print_heading("4. DATA CLEANING")

    print("\nMissing values before cleaning:")
    print(df.isna().sum().to_string())

    # Convert text columns to pandas string type and trim spaces.
    text_columns = [
        "Facility_Type",
        "Facility_Name",
        "Area",
        "Speciality",
        "Opening_hours",
        "Publicly_listed_consultation_range",
        "Emergency_availability"
    ]

    for col in text_columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    # Standardize emergency availability without inventing values.
    emergency_col = "Emergency_availability"
    emergency_clean = df[emergency_col].str.lower()

    df.loc[
        emergency_clean.isin(["yes available", "yes", "available"]),
        emergency_col
    ] = "Yes Available"

    df.loc[
        emergency_clean.isin(["not available", "no", "unavailable"]),
        emergency_col
    ] = "Not Available"

    # Numeric conversion.
    df["Sr_no"] = pd.to_numeric(df["Sr_no"], errors="coerce")
    df["Public_rating"] = pd.to_numeric(
        df["Public_rating"], errors="coerce"
    )
    df["Review_count"] = pd.to_numeric(
        df["Review_count"], errors="coerce"
    )

    # Identify and remove exact duplicate rows only.
    duplicate_count = int(df.duplicated().sum())
    print("\nExact duplicate rows found:", duplicate_count)

    df = df.drop_duplicates().copy()

    print("Rows after exact duplicate removal:", len(df))

    print("\nMissing values after text/numeric standardization:")
    print(df.isna().sum().to_string())

    # Keep missing values as missing; do not invent ratings or fees.
    # ----------------------------------------------------------
    # 6. INVALID VALUES AND POTENTIAL OUTLIERS
    # ----------------------------------------------------------

    print_heading("5. INVALID VALUES AND POTENTIAL OUTLIERS")

    invalid_ratings = df[
        df["Public_rating"].notna()
        & ~df["Public_rating"].between(0, 5)
    ]

    print("\nRatings outside the 0-5 range:")
    if invalid_ratings.empty:
        print("None found.")
    else:
        print(
            invalid_ratings[
                ["Facility_Name", "Public_rating"]
            ].to_string(index=False)
        )

    negative_reviews = df[
        df["Review_count"].notna()
        & (df["Review_count"] < 0)
    ]

    print("\nNegative review counts:")
    if negative_reviews.empty:
        print("None found.")
    else:
        print(
            negative_reviews[
                ["Facility_Name", "Review_count"]
            ].to_string(index=False)
        )

    # Flag invalid values as missing for analysis, but do not delete rows.
    df.loc[
        ~df["Public_rating"].between(0, 5),
        "Public_rating"
    ] = np.nan

    df.loc[df["Review_count"] < 0, "Review_count"] = np.nan

    def iqr_outliers(series):
        values = series.dropna()
        if len(values) < 4:
            return pd.Series(False, index=series.index)

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            return pd.Series(False, index=series.index)

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        return (series < lower) | (series > upper)

    rating_outlier_mask = iqr_outliers(df["Public_rating"])
    review_outlier_mask = iqr_outliers(df["Review_count"])

    print("\nPotential rating outliers (IQR method):")
    rating_outliers = df.loc[
        rating_outlier_mask,
        ["Facility_Name", "Public_rating"]
    ]
    print(
        rating_outliers.to_string(index=False)
        if not rating_outliers.empty else "None found."
    )

    print("\nPotential review-count outliers (IQR method):")
    review_outliers = df.loc[
        review_outlier_mask,
        ["Facility_Name", "Review_count"]
    ]
    print(
        review_outliers.to_string(index=False)
        if not review_outliers.empty else "None found."
    )

    print(
        "\nNote: Potential outliers are flagged for review; "
        "they are not automatically deleted."
    )

    # ----------------------------------------------------------
    # 7. DATA TRANSFORMATION
    # ----------------------------------------------------------

    print_heading("6. DATA TRANSFORMATION")

    # Extract numbers from values such as ₹400 - ₹1,000.
    def extract_fee_numbers(value):
        if pd.isna(value):
            return []

        # Remove commas used as thousands separators.
        text = str(value).replace(",", "")
        return re.findall(r"\d+(?:\.\d+)?", text)

    fee_numbers = df[
        "Publicly_listed_consultation_range"
    ].apply(extract_fee_numbers)

    df["Fee_Min_Extracted"] = fee_numbers.apply(
        lambda values: float(values[0])
        if len(values) >= 2 else np.nan
    )

    df["Fee_Max_Extracted"] = fee_numbers.apply(
        lambda values: float(values[1])
        if len(values) >= 2 else np.nan
    )

    df["Fee_Range_Extracted"] = df.apply(
        lambda row: (
            f"₹{row['Fee_Min_Extracted']:,.0f} - "
            f"₹{row['Fee_Max_Extracted']:,.0f}"
        )
        if pd.notna(row["Fee_Min_Extracted"])
        and pd.notna(row["Fee_Max_Extracted"])
        else "Not Available",
        axis=1
    )

    print("Minimum and maximum consultation fees extracted where possible.")
    print("\nTransformed fee sample:")
    print(
        df[
            [
                "Facility_Name",
                "Publicly_listed_consultation_range",
                "Fee_Min_Extracted",
                "Fee_Max_Extracted"
            ]
        ].head(10).to_string(index=False)
    )

    # ----------------------------------------------------------
    # 8. DESCRIPTIVE STATISTICS
    # ----------------------------------------------------------

    print_heading("7. DESCRIPTIVE STATISTICS")

    print("\nNumeric summary:")
    print(
        df[["Public_rating", "Review_count",
            "Fee_Min_Extracted", "Fee_Max_Extracted"]]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nFacility type-wise count:")
    print(
        df["Facility_Type"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nArea-wise facility count:")
    area_counts = df["Area"].value_counts(dropna=False)
    print(area_counts.to_string())

    print("\nEmergency availability count:")
    print(df["Emergency_availability"].value_counts(dropna=False).to_string())

    # ----------------------------------------------------------
    # 9. BUSINESS QUESTIONS AND INSIGHTS
    # ----------------------------------------------------------

    print_heading("8. BUSINESS QUESTIONS AND INSIGHTS")

    print("\nA. Which facility type has the most records?")
    type_counts = df["Facility_Type"].value_counts()
    if not type_counts.empty:
        print(
            f"{type_counts.index[0]}: {type_counts.iloc[0]} records"
        )
    else:
        print("No facility type data available.")

    print("\nB. Which areas have the most and fewest records?")
    valid_area_counts = df["Area"].dropna().value_counts()
    if not valid_area_counts.empty:
        print(
            "Most records:",
            valid_area_counts.index[0],
            "-",
            int(valid_area_counts.iloc[0])
        )
        print(
            "Fewest records:",
            valid_area_counts.index[-1],
            "-",
            int(valid_area_counts.iloc[-1])
        )
        print(
            "Note: This compares only areas in the collected dataset, "
            "not total real-world healthcare coverage."
        )
    else:
        print("Area information is unavailable.")

    print("\nC. Average public rating:")
    avg_rating = df["Public_rating"].mean()
    print(
        round(avg_rating, 2)
        if pd.notna(avg_rating) else "Not available"
    )

    print("\nD. Highest review count:")
    valid_reviews = df.dropna(subset=["Review_count"])
    if not valid_reviews.empty:
        top_review_row = valid_reviews.loc[
            valid_reviews["Review_count"].idxmax()
        ]
        print(
            top_review_row["Facility_Name"],
            "-",
            int(top_review_row["Review_count"]),
            "reviews"
        )
    else:
        print("Review counts unavailable.")

    print("\nE. Facilities publicly marked as having emergency availability:")
    emergency_counts = df["Emergency_availability"].value_counts()
    print(emergency_counts.to_string())

    print("\nF. Most frequently listed speciality terms:")
    speciality_counts = {}

    for value in df["Speciality"].dropna():
        # Split common separators; these counts are approximate term
        # frequencies because source descriptions may combine specialties.
        terms = re.split(r",|;|&|\band\b", str(value), flags=re.IGNORECASE)
        for term in terms:
            term = term.strip(" .")
            if term and term.lower() not in ["not available", "na", "n/a"]:
                speciality_counts[term] = speciality_counts.get(term, 0) + 1

    if speciality_counts:
        speciality_table = pd.Series(speciality_counts).sort_values(
            ascending=False
        )
        print(speciality_table.head(15).to_string())
    else:
        print("No speciality terms available.")

    print("\nG. Facilities with rating 4.5 or higher:")
    high_rating = df[df["Public_rating"] >= 4.5]
    print("Count:", len(high_rating))
    if not high_rating.empty:
        print(
            high_rating[
                ["Facility_Name", "Area", "Public_rating"]
            ].sort_values(
                "Public_rating", ascending=False
            ).head(15).to_string(index=False)
        )

    print("\nH. Consultation fee summary:")
    print("Rows with extracted minimum fee:", int(df["Fee_Min_Extracted"].notna().sum()))
    print("Rows with extracted maximum fee:", int(df["Fee_Max_Extracted"].notna().sum()))

    # ----------------------------------------------------------
    # 10. CORRELATION ANALYSIS
    # ----------------------------------------------------------

    print_heading("9. CORRELATION ANALYSIS")

    numeric_columns = [
        "Public_rating",
        "Review_count",
        "Fee_Min_Extracted",
        "Fee_Max_Extracted"
    ]

    correlation_data = df[numeric_columns].corr(numeric_only=True)

    print("\nCorrelation matrix:")
    print(correlation_data.round(2).to_string())

    print(
        "\nNote: Correlation indicates association only; it does not "
        "prove that one variable causes another."
    )

    # ----------------------------------------------------------
    # 11. SAVE CLEANED DATASET
    # ----------------------------------------------------------

    print_heading("10. SAVE CLEANED DATASET")

    df.to_csv(CLEANED_FILE, index=False, encoding="utf-8-sig")
    print("Cleaned dataset saved to:")
    print(CLEANED_FILE)
    print("Records in cleaned dataset:", len(df))

    # ----------------------------------------------------------
    # 12. DATA VISUALIZATION - 9 CHARTS
    # ----------------------------------------------------------

    print_heading("11. DATA VISUALIZATION - 9 CHARTS")

    # Chart 1: Area-wise Facility Distribution
    area_plot = df["Area"].fillna("Not Available").value_counts().head(15)

    if not area_plot.empty:
        area_plot.sort_values().plot(kind="barh", figsize=(10, 6))
        plt.title("Area-wise Facility Distribution (Top 15 Areas)")
        plt.xlabel("Number of Facilities")
        plt.ylabel("Area")
        save_chart("01_Area_Wise_Facility_Distribution.png")
    else:
        print("Chart 1 skipped: no area data.")

    # Chart 2: Facility Name-wise Comparison - top 15 by review count
    top_facilities = df.dropna(
        subset=["Facility_Name", "Review_count"]
    ).nlargest(15, "Review_count")

    if not top_facilities.empty:
        plot_data = top_facilities.sort_values("Review_count")
        plt.figure(figsize=(11, 7))
        plt.barh(
            plot_data["Facility_Name"],
            plot_data["Review_count"]
        )
        plt.title("Facility Name-wise Comparison by Review Count (Top 15)")
        plt.xlabel("Review Count")
        plt.ylabel("Facility Name")
        save_chart("02_Facility_Name_Wise_Comparison.png")
    else:
        print("Chart 2 skipped: no review-count data.")

    # Chart 3: Facility Type-wise Count
    type_plot = df["Facility_Type"].fillna("Not Available").value_counts()

    if not type_plot.empty:
        type_plot.plot(kind="bar", figsize=(8, 5))
        plt.title("Facility Type-wise Count")
        plt.xlabel("Facility Type")
        plt.ylabel("Number of Facilities")
        plt.xticks(rotation=20, ha="right")
        save_chart("03_Facility_Type_Wise_Count.png")
    else:
        print("Chart 3 skipped: no facility type data.")

    # Chart 4: Facility Rating Analysis
    rating_values = df["Public_rating"].dropna()

    if not rating_values.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(
            rating_values,
            bins=np.arange(0, 5.6, 0.5),
            edgecolor="black"
        )
        plt.title("Facility Rating Analysis")
        plt.xlabel("Public Rating")
        plt.ylabel("Number of Facilities")
        plt.xticks(np.arange(0, 5.1, 0.5))
        save_chart("04_Facility_Rating_Analysis.png")
    else:
        print("Chart 4 skipped: no valid rating data.")

    # Chart 5: Review Count Analysis - distribution
    review_values = df["Review_count"].dropna()

    if not review_values.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(review_values, bins=10, edgecolor="black")
        plt.title("Review Count Analysis - Distribution")
        plt.xlabel("Review Count")
        plt.ylabel("Number of Facilities")
        save_chart("05_Review_Count_Analysis.png")
    else:
        print("Chart 5 skipped: no review-count data.")

    # Chart 6: Opening Hours Analysis
    hours_plot = df["Opening_hours"].fillna("Not Available").value_counts().head(15)

    if not hours_plot.empty:
        hours_plot.sort_values().plot(kind="barh", figsize=(10, 6))
        plt.title("Opening Hours Analysis (Top 15 Categories)")
        plt.xlabel("Number of Facilities")
        plt.ylabel("Opening Hours")
        save_chart("06_Opening_Hours_Analysis.png")
    else:
        print("Chart 6 skipped: no opening-hours data.")

    # Chart 7: Area-wise Average Rating
    area_rating = (
        df.dropna(subset=["Area", "Public_rating"])
        .groupby("Area")["Public_rating"]
        .mean()
        .sort_values()
        .tail(15)
    )

    if not area_rating.empty:
        area_rating.plot(kind="barh", figsize=(10, 6))
        plt.title("Area-wise Average Rating (Top 15 by Average)")
        plt.xlabel("Average Public Rating")
        plt.ylabel("Area")
        plt.xlim(0, 5)
        save_chart("07_Area_Wise_Average_Rating.png")
    else:
        print("Chart 7 skipped: insufficient area/rating data.")

    # Chart 8: Consultation Range Analysis - most common listed ranges
    fee_range_plot = (
        df["Publicly_listed_consultation_range"]
        .fillna("Not Available")
        .value_counts()
        .head(15)
    )

    if not fee_range_plot.empty:
        fee_range_plot.sort_values().plot(kind="barh", figsize=(10, 6))
        plt.title("Consultation Range Analysis (Most Common Ranges)")
        plt.xlabel("Number of Facilities")
        plt.ylabel("Listed Consultation Range")
        save_chart("08_Consultation_Range_Analysis.png")
    else:
        print("Chart 8 skipped: no consultation-range data.")

    # Chart 9: Emergency Availability Analysis
    emergency_plot = (
        df["Emergency_availability"]
        .fillna("Not Available")
        .value_counts()
    )

    if not emergency_plot.empty:
        emergency_plot.plot(kind="bar", figsize=(8, 5))
        plt.title("Emergency Availability Analysis")
        plt.xlabel("Emergency Availability")
        plt.ylabel("Number of Facilities")
        plt.xticks(rotation=15, ha="right")
        save_chart("09_Emergency_Availability_Analysis.png")
    else:
        print("Chart 9 skipped: no emergency data.")

    # ----------------------------------------------------------
    # 13. FINAL PROJECT SUMMARY
    # ----------------------------------------------------------

    print_heading("12. FINAL PROJECT SUMMARY")

    print("Input records:", len(df))
    print("Total columns including derived fee columns:", len(df.columns))
    print("Output folder:", OUTPUT_DIR)
    print("Charts folder:", CHART_DIR)
    print("Cleaned dataset:", CLEANED_FILE)
    print("\nProject analysis completed.")

# --------------------------------------------------------------
# SAVE REPORT AND SHOW OUTPUT
# --------------------------------------------------------------

report_text = report_buffer.getvalue()

with open(REPORT_FILE, "w", encoding="utf-8") as report:
    report.write(report_text)

print(report_text)

print("\n" + "=" * 70)
print("ALL AVAILABLE ANALYSIS STEPS FINISHED")
print("=" * 70)
print("Report saved to:", REPORT_FILE)
print("Cleaned CSV saved to:", CLEANED_FILE)
print("Charts saved in:", CHART_DIR)
print("\nOpen the Python_Charts folder to view the PNG charts.")
