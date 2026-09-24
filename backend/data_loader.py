"""
FinSight AI — Data Loader
Loads all CSVs into pandas DataFrames at startup.
Joined once, shared across all agentic tools — no repeated disk reads.

The deployment ZIP excludes gl_data/, master_data/, planning_data/ (100MB+,
kept out of the App Service package). In production those files are pulled
from Azure Blob Storage on first access and cached to local disk; local dev
just reads the files that are already sitting in the project root.
"""

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Resolve paths relative to project root (one level up from backend/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files DataLoader needs, relative to project root — used for blob fallback
_REQUIRED_FILES = [
    "master_data/dim_responsibility_center.csv",
    "planning_data/fact_planning_combined.csv",
    "gl_data/fact_gl_transactions.csv",
]


def _path(*parts: str) -> str:
    return os.path.join(_PROJECT_ROOT, *parts)


def _ensure_local_files():
    """Download any missing required CSVs from Blob Storage into the project root."""
    missing = [f for f in _REQUIRED_FILES if not os.path.exists(_path(*f.split("/")))]
    if not missing:
        return

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError(
            f"Missing local data files {missing} and AZURE_STORAGE_CONNECTION_STRING "
            "is not set — cannot fall back to Blob Storage."
        )

    from azure.storage.blob import BlobServiceClient

    logger.info(f"Downloading {len(missing)} data file(s) from Blob Storage...")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "financial-data")
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container = blob_service.get_container_client(container_name)

    for rel_path in missing:
        local_path = _path(*rel_path.split("/"))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(container.download_blob(rel_path).readall())
        logger.info(f"Downloaded {rel_path}")


class DataLoader:
    """
    Singleton-style loader. Call DataLoader.get() to get the shared instance.
    Loads on first access, cached after that.
    """

    _instance: "DataLoader | None" = None

    def __init__(self):
        _ensure_local_files()
        logger.info("Loading financial data CSVs...")

        # ── Dimension tables ──────────────────────────────────────────────────
        self.dim_rc = pd.read_csv(_path("master_data", "dim_responsibility_center.csv"))

        # ── Planning data (budget / actuals / forecast) ───────────────────────
        planning_raw = pd.read_csv(_path("planning_data", "fact_planning_combined.csv"))

        # Join Department from dim_rc (planning only has RC_Code)
        rc_map = self.dim_rc[["RC_Code", "Department"]].drop_duplicates()
        self.planning = planning_raw.merge(rc_map, on="RC_Code", how="left")

        # Normalise column names used downstream
        self.planning["Fiscal_Year"] = self.planning["Fiscal_Year"].astype(str)
        self.planning["Quarter"] = self.planning["Quarter"].astype(str)

        # ── GL Transactions ───────────────────────────────────────────────────
        self.gl = pd.read_csv(_path("gl_data", "fact_gl_transactions.csv"))
        self.gl["Fiscal_Year"] = self.gl["Fiscal_Year"].astype(str)
        self.gl["Quarter"] = self.gl["Quarter"].astype(str)

        logger.info(
            f"Data loaded — planning: {len(self.planning):,} rows | "
            f"GL: {len(self.gl):,} rows"
        )

    @classmethod
    def get(cls) -> "DataLoader":
        if cls._instance is None:
            cls._instance = DataLoader()
        return cls._instance

    # ── Convenience accessors ─────────────────────────────────────────────────

    def departments(self) -> list[str]:
        """Distinct department names present in planning data."""
        return sorted(self.planning["Department"].dropna().unique().tolist())

    def fiscal_years(self) -> list[str]:
        return sorted(self.planning["Fiscal_Year"].unique().tolist())

    def latest_actuals_period(self) -> tuple[str, str]:
        """(fiscal_year, quarter) of the most recent quarter with any actuals booked."""
        if not hasattr(self, "_latest_actuals"):
            by_q = self.planning.groupby(["Fiscal_Year", "Quarter"])["Actuals_USD"].sum()
            self._latest_actuals = max(by_q[by_q > 0].index)
        return self._latest_actuals

    def has_actuals(self, fiscal_year: str, quarter: str) -> bool:
        return (fiscal_year, quarter) <= self.latest_actuals_period()
