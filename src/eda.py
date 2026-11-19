"""Download Seattle datasets and produce reproducible CSE 163 EDA outputs."""

from pathlib import Path
import json
import time
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import numpy as np
from scipy.stats import pearsonr, shapiro
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor
)
from statsmodels.stats.diagnostic import het_breuschpagan

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
OUT = BASE / "outputs"
FIGURES = OUT / "figures"

PERMITS_URL = (
    "https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services/"
    "Residential_Building_Permits_Issued_and_Final/FeatureServer/0"
)
TREES_URL = (
    "https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services/"
    "SDOT_Trees_CDL/FeatureServer/0"
)
CANOPY_URL = (
    "https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services/"
    "Seattle_Tree_Canopy_2016_2021_SDOT_Urban_Forestry_Management_Units/"
    "FeatureServer/13"
)

START_DATE = pd.Timestamp("2016-01-01", tz="UTC")
END_DATE = pd.Timestamp("2022-01-01", tz="UTC")


def fetch_layer(url, name, where="1=1", fields=None):
    """Fetch ArcGIS features in batches using selected fields."""
    metadata_response = requests.get(
        url, params={"f": "pjson"}, timeout=30
    )
    metadata_response.raise_for_status()
    metadata = metadata_response.json()

    if fields is None:
        out_fields = "*"
    else:
        available = {
            field["name"].upper(): field["name"]
            for field in metadata.get("fields", [])
        }
        selected = [
            available[field.upper()]
            for field in fields
            if field.upper() in available
        ]
        if not selected:
            raise ValueError("None of the requested fields exist.")
        out_fields = ",".join(selected)

    features = []
    offset = 0
    page_size = 2000

    print("Downloading", name, "...")

    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }

        response = requests.get(
            url + "/query", params=params, timeout=120
        )
        response.raise_for_status()
        payload = response.json()

        if "error" in payload:
            raise RuntimeError(payload["error"])

        page = payload.get("features", [])
        features.extend(page)

        print(name, "records downloaded:", len(features))

        if len(page) < page_size:
            break

        offset += page_size
        time.sleep(0.15)

    if not features:
        raise ValueError("No features returned for " + name)

    path = RAW / (name + ".geojson")
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": features
    }))

    return gpd.read_file(path)

def to_datetime(series):
    """Convert ArcGIS dates or datetime values to UTC datetimes."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce", utc=True)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return pd.to_datetime(
            numeric, unit="ms", errors="coerce", utc=True
        )
    return pd.to_datetime(series, errors="coerce", utc=True)


def date_mask(series):
    """Return a mask for dates in 2016 through 2021, inclusive."""
    dates = to_datetime(series)
    return dates.ge(START_DATE) & dates.lt(END_DATE)


def first_existing(columns, choices):
    """Return the first candidate column present in a dataframe."""
    for choice in choices:
        if choice in columns:
            return choice
    return None


def profile_dataset(name, frame):
    """Record dimensions and missingness for one dataset."""
    rows = []
    for column in frame.columns:
        rows.append({
            "dataset": name,
            "rows": len(frame),
            "columns": len(frame.columns),
            "variable": column,
            "missing_count": int(frame[column].isna().sum()),
            "missing_percent": (
                float(frame[column].isna().mean() * 100)
            ),
            "dtype": str(frame[column].dtype),
        })
    return rows


def summarize_variables(name, frame, columns):
    """Create seven-number summaries and categorical counts."""
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column].dropna()
        if pd.api.types.is_numeric_dtype(series):
            stats = series.describe()
            rows.append({
                "dataset": name,
                "variable": column,
                "kind": "quantitative",
                "count": stats.get("count"),
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "min": stats.get("min"),
                "q1": stats.get("25%"),
                "median": stats.get("50%"),
                "q3": stats.get("75%"),
                "max": stats.get("max"),
                "category": None,
                "category_count": None,
            })
        else:
            counts = series.astype(str).value_counts(dropna=False)
            for category, count in counts.items():
                rows.append({
                    "dataset": name,
                    "variable": column,
                    "kind": "categorical",
                    "count": len(series),
                    "mean": None,
                    "std": None,
                    "min": None,
                    "q1": None,
                    "median": None,
                    "q3": None,
                    "max": None,
                    "category": category,
                    "category_count": int(count),
                })
    return rows


def save_histogram(frame, column, title, filename):
    """Save a histogram for a numeric variable."""
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=20, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Number of records")
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=160)
    plt.close(fig)


def save_bar(counts, title, xlabel, filename):
    """Save a bar chart from a value-count series."""
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of records")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=160)
    plt.close(fig)

def add_housing_density(unit_table, canopy_analysis):
    """Calculate management-unit area and housing density."""
    area_sq_miles = (
        canopy_analysis[["unit_id", "geometry"]]
        .to_crs(epsg=32610)
        .assign(
            area_sq_miles=lambda frame: (
                frame.geometry.area / 2589988.110336
            )
        )
        .set_index("unit_id")["area_sq_miles"]
    )

    result = unit_table.join(area_sq_miles.rename("area_sq_miles"))
    result["net_units_per_sq_mile"] = (
        result["net_housing_units"] / result["area_sq_miles"]
    )

    return result

def run_statistical_analysis(unit_table):
    """Analyze housing density and relative canopy change."""
    data = unit_table[
        [
            "net_units_per_sq_mile",
            "canopy_relative_change",
            "canopy_2016_percent",
        ]
    ].dropna()

    x = data["net_units_per_sq_mile"].to_numpy()
    y = data["canopy_relative_change"].to_numpy()

    observed_r = pearsonr(x, y).statistic

    rng = np.random.default_rng(42)
    permutations = 10000
    permuted_correlations = []

    for _ in range(permutations):
        shuffled_y = rng.permutation(y)
        permuted_correlations.append(
            pearsonr(x, shuffled_y).statistic
        )

    permutation_p = (
        np.sum(np.abs(permuted_correlations) >= abs(observed_r)) + 1
    ) / (permutations + 1)

    predictors = data[
        ["net_units_per_sq_mile", "canopy_2016_percent"]
    ]
    predictors = sm.add_constant(predictors)
    model = sm.OLS(y, predictors).fit(cov_type="HC3")

    residual_check = data.copy()
    residual_check["residual"] = model.resid
    residual_check["absolute_residual"] = (
        residual_check["residual"].abs()
    )

    print("\nManagement units with largest residuals:")
    print(
        residual_check.nlargest(5, "absolute_residual")[
            ["residual", "absolute_residual"]
        ]
    )
    predictor_correlation = data[
        ["net_units_per_sq_mile", "canopy_2016_percent"]
    ].corr().iloc[0, 1]

    vif_values = [
        variance_inflation_factor(predictors.to_numpy(), index)
        for index in range(1, predictors.shape[1])
    ]

    residuals = model.resid
    shapiro_stat, shapiro_p = shapiro(residuals)

    bp_test = het_breuschpagan(
        residuals, model.model.exog
    )
    bp_p = bp_test[1]

    # Plot residuals against fitted values.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(model.fittedvalues, residuals)
    ax.axhline(0, linestyle="--", color="black")
    ax.set_title("Regression Residuals vs. Fitted Values")
    ax.set_xlabel("Fitted relative canopy change")
    ax.set_ylabel("Residuals")
    fig.tight_layout()
    fig.savefig(
        FIGURES / "regression_residuals.png", dpi=160
    )
    plt.close(fig)

    # Create a Q-Q plot of regression residuals.
    fig = sm.qqplot(residuals, line="45", fit=True)
    fig.suptitle("Q-Q Plot of Regression Residuals")
    fig.tight_layout()
    fig.savefig(
        FIGURES / "regression_qq_plot.png", dpi=160
    )
    plt.close(fig)
    # Bootstrap confidence interval for the housing-density coefficient.
    bootstrap_coefficients = []

    for _ in range(5000):
        sample_indices = rng.integers(0, len(data), len(data))
        sample = data.iloc[sample_indices]

        sample_x = sm.add_constant(
            sample[
                ["net_units_per_sq_mile", "canopy_2016_percent"]
            ]
        )

        try:
            bootstrap_model = sm.OLS(
                sample["canopy_relative_change"], sample_x
            ).fit()
            bootstrap_coefficients.append(
                bootstrap_model.params["net_units_per_sq_mile"]
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

    bootstrap_ci = np.percentile(
        bootstrap_coefficients, [2.5, 97.5]
    )

    loo_coefficients = []

    for index in data.index:
        reduced_data = data.drop(index)
        reduced_x = sm.add_constant(
            reduced_data[
                ["net_units_per_sq_mile", "canopy_2016_percent"]
            ]
        )
        reduced_model = sm.OLS(
            reduced_data["canopy_relative_change"], reduced_x
        ).fit()

        loo_coefficients.append(
            reduced_model.params["net_units_per_sq_mile"]
        )

    results = [
                {
            "analysis": "Predictor correlation",
            "statistic": predictor_correlation,
            "p_value": None,
            "lower_ci": None,
            "upper_ci": None,
        },
        {
            "analysis": "Housing density VIF",
            "statistic": vif_values[0],
            "p_value": None,
            "lower_ci": None,
            "upper_ci": None,
        },
        {
            "analysis": "Residual Shapiro-Wilk test",
            "statistic": shapiro_stat,
            "p_value": shapiro_p,
            "lower_ci": None,
            "upper_ci": None,
        },
        {
            "analysis": "Breusch-Pagan heteroskedasticity test",
            "statistic": bp_test[0],
            "p_value": bp_p,
            "lower_ci": None,
            "upper_ci": None,
        },
        {
            "analysis": "Pearson correlation",
            "statistic": observed_r,
            "p_value": pearsonr(x, y).pvalue,
            "lower_ci": None,
            "upper_ci": None,
        },
        {
            "analysis": "Permutation test",
            "statistic": observed_r,
            "p_value": permutation_p,
            "lower_ci": None,
            "upper_ci": None,
        },
        {
            "analysis": "Regression: housing density",
            "statistic": model.params["net_units_per_sq_mile"],
            "p_value": model.pvalues["net_units_per_sq_mile"],
            "lower_ci": model.conf_int().loc[
                "net_units_per_sq_mile", 0
            ],
            "upper_ci": model.conf_int().loc[
                "net_units_per_sq_mile", 1
            ],
        },
        {
            "analysis": "Bootstrap housing-density coefficient",
            "statistic": np.mean(bootstrap_coefficients),
            "p_value": None,
            "lower_ci": bootstrap_ci[0],
            "upper_ci": bootstrap_ci[1],
        },
        {
            "analysis": "Leave-one-out housing coefficient range",
            "statistic": np.mean(loo_coefficients),
            "p_value": None,
            "lower_ci": min(loo_coefficients),
            "upper_ci": max(loo_coefficients),
        },
    ]

    return pd.DataFrame(results), model

def main():
    """Run the complete EDA pipeline."""
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    log = []

    permits = fetch_layer(
        PERMITS_URL,
        "permits",
        fields=[
            "ISS_DATE", "APP_DATE", "NET_UNITS",
            "STGLSTTITL", "DWELTYPE"
        ]
    )

    trees = fetch_layer(
        TREES_URL,
        "trees",
        fields=[
            "OWNERSHIP", "PLANTED_DATE", "DIAM",
            "CONDITION", "BOTANICAL_NAME",
            "EXPEDITE", "REMOVED_DATE", "REMOVAL_DATE"
        ]
    )

    canopy = fetch_layer(CANOPY_URL, "canopy_units")

    log.append("Downloaded permits, SDOT trees, and canopy management units.")
    log.append(
        "The source layers are current public ArcGIS layers; the script "
        "filters date-based records to 2016-2021."
    )

    unit_id = first_existing(
        canopy.columns, ["TC_ID", "MANAGEMENT_UNITS", "OBJECTID"]
    )
    canopy_2016 = first_existing(
        canopy.columns,
        ["TreeCanopy_2016_Percent", "TC_E_P", "Can_P"]
    )
    canopy_change = first_existing(
        canopy.columns,
        ["Change_PercentChange", "TC_R_PC", "RELATIVE_PERCENT_CHANGE"]
    )
    if unit_id is None or canopy_2016 is None or canopy_change is None:
        raise ValueError(
            "Could not identify canopy ID, 2016 canopy, and change fields. "
            "Inspect canopy_units.geojson and update the field candidates."
        )

    canopy[unit_id] = canopy[unit_id].astype(str)
    canopy_analysis = canopy[
        [unit_id, canopy_2016, canopy_change, "geometry"]
    ].copy()
    canopy_analysis = canopy_analysis.rename(columns={
        unit_id: "unit_id",
        canopy_2016: "canopy_2016_percent",
        canopy_change: "canopy_relative_change",
    })

    permit_date = first_existing(
        permits.columns, ["ISS_DATE", "APP_DATE"]
    )
    if permit_date is None:
        raise ValueError("No permit date field found.")
    permits = permits[date_mask(permits[permit_date])].copy()
    net_field = first_existing(permits.columns, ["NET_UNITS"])
    if net_field is None:
        raise ValueError("NET_UNITS field was not found in permit data.")
    permits[net_field] = pd.to_numeric(
        permits[net_field], errors="coerce"
    )
    permits = permits.dropna(subset=[net_field, "geometry"])
    permit_join = gpd.sjoin(
        permits, canopy_analysis[["unit_id", "geometry"]],
        how="left", predicate="within"
    )
    permit_totals = permit_join.groupby("unit_id")[net_field].sum()
    permit_totals.name = "net_housing_units"

    if "OWNERSHIP" in trees.columns:
        trees = trees[trees["OWNERSHIP"].astype(str).str.upper() == "SDOT"]
    planted_field = first_existing(trees.columns, ["PLANTED_DATE"])
    if planted_field is None:
        raise ValueError("PLANTED_DATE field was not found in tree data.")
    trees["planted_in_period"] = date_mask(trees[planted_field])
    tree_join = gpd.sjoin(
        trees, canopy_analysis[["unit_id", "geometry"]],
        how="left", predicate="within"
    )
    planted_totals = (
        tree_join[tree_join["planted_in_period"]]
        .groupby("unit_id")
        .size()
    )
    planted_totals.name = "trees_planted_2016_2021"

    unit_table = canopy_analysis.drop(columns="geometry").copy()
    unit_table = unit_table.set_index("unit_id")
    unit_table = unit_table.join(permit_totals).join(planted_totals)
    unit_table = unit_table.fillna({
        "net_housing_units": 0,
        "trees_planted_2016_2021": 0,
    })

    # Calculate management-unit area and housing density.
    unit_table = add_housing_density(unit_table, canopy_analysis)

    unit_table = unit_table.reset_index()
    unit_table.to_csv(OUT / "management_unit_eda.csv", index=False)

    statistical_results, regression_model = (
        run_statistical_analysis(unit_table)
    )
    statistical_results.to_csv(
        OUT / "statistical_results.csv", index=False
    )

    (OUT / "regression_summary.txt").write_text(
        regression_model.summary().as_text()
    )

    profile_rows = []
    summary_rows = []
    datasets = {
        "permits_2016_2021": permits,
        "sdot_trees_current_snapshot": trees,
        "canopy_management_units": canopy,
        "permits_joined_to_canopy": permit_join,
        "trees_joined_to_canopy": tree_join,
    }
    for name, frame in datasets.items():
        profile_rows.extend(profile_dataset(name, frame))
    profile_frame = pd.DataFrame(profile_rows)
    profile_frame.to_csv(OUT / "dataset_profiles.csv", index=False)

    summary_rows.extend(summarize_variables(
        "permits_2016_2021", permits,
        [net_field, permit_date, "STGLSTTITL", "DWELTYPE"]
    ))
    summary_rows.extend(summarize_variables(
        "sdot_trees_current_snapshot", trees,
        [planted_field, "DIAM", "OWNERSHIP", "CONDITION",
        "BOTANICAL_NAME"]
    ))
    summary_rows.extend(summarize_variables(
        "canopy_management_units", canopy,
        [canopy_2016, canopy_change, unit_id]
    ))
    pd.DataFrame(summary_rows).to_csv(
        OUT / "variable_summaries.csv", index=False
    )

    # two visualizations for each original dataset.
    save_histogram(
        permits, net_field, "Net housing units per permit record",
        "permits_net_units_histogram.png"
    )
    if "STGLSTTITL" in permits.columns:
        save_bar(
            permits["STGLSTTITL"].value_counts(),
            "Permit records by stage", "Permit stage",
            "permits_by_stage.png"
        )
    save_histogram(
        trees, "DIAM", "Street tree diameter distribution",
        "trees_diameter_histogram.png"
    )
    if "CONDITION" in trees.columns:
        save_bar(
            trees["CONDITION"].value_counts().head(15),
            "Most common street tree conditions", "Condition",
            "trees_condition_counts.png"
        )
    save_histogram(
        canopy, canopy_2016, "2016 tree canopy percentage by unit",
        "canopy_2016_histogram.png"
    )
    save_histogram(
        canopy, canopy_change, "Relative tree canopy change, 2016-2021",
        "canopy_change_histogram.png"
    )

    # Joined-dataset visualization: housing density and canopy change.
    joined = unit_table.dropna(
        subset=["canopy_relative_change", "net_units_per_sq_mile"]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        joined["net_units_per_sq_mile"],
        joined["canopy_relative_change"]
    )
    ax.set_title("Housing density and relative canopy change by unit")
    ax.set_xlabel("Net housing units per square mile (2016-2021)")
    ax.set_ylabel("Relative canopy change (%)")
    fig.tight_layout()
    fig.savefig(FIGURES / "housing_vs_canopy_change.png", dpi=160)
    plt.close(fig)

    joined_tree = unit_table.dropna(
        subset=["canopy_relative_change", "trees_planted_2016_2021"]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        joined_tree["canopy_relative_change"],
        joined_tree["trees_planted_2016_2021"]
    )
    ax.set_title("Street tree planting and canopy change by unit")
    ax.set_xlabel("Relative canopy change (%)")
    ax.set_ylabel("Trees planted (2016-2021)")
    fig.tight_layout()
    fig.savefig(FIGURES / "planting_vs_canopy_change.png", dpi=160)
    plt.close(fig)

    # Explicitly flag the unresolved removal-field mismatch.
    removal_field = first_existing(
        trees.columns, ["EXPEDITE", "REMOVED_DATE", "REMOVAL_DATE"]
    )
    if removal_field is None:
        log.append(
            "STOP: no verified tree-removal date field was found. "
            "The published schema contains EXPDATE (Expiration Date), "
            "which is not treated as a removal date. Question 4 and "
            "removal-rate analyses are pending data-dictionary verification."
        )
    else:
        log.append(
            "A candidate removal field exists: " + removal_field +
            ". Verify its definition and historical completeness before use."
        )

    log.append(
        "No historical 2016 tree inventory denominator was reconstructed. "
        "Do not calculate removal rates per 1,000 baseline trees until "
        "historical inventory completeness is verified."
    )
    (OUT / "eda_run_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    print("EDA outputs saved to:", OUT)


if __name__ == "__main__":
    main()
