"""
Ingest Citi Bike trip files (CSV or CSV.GZ) into SQLite.

Input:  data/raw/citibike/*.csv or *.csv.gz
Output: data/processed/citibike.sqlite (table: trips)

Handles both legacy and current schemas by best-effort column mapping.
Computes duration (minutes) and great-circle distance (miles) when lat/lng present.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw/citibike")
DB_PATH = Path("data/processed/citibike.sqlite")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Haversine distance (km)
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower()
          .str.replace(r"[^\w\s]+", "", regex=True)
          .str.replace(r"\s+", "_", regex=True)
    )
    return df

def load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)

    # Map possible column names across schema versions
    started = next((c for c in ["started_at","starttime","start_time"] if c in df.columns), None)
    ended   = next((c for c in ["ended_at","stoptime","stop_time","endtime"] if c in df.columns), None)
    if not started or not ended:
        raise ValueError(f"Missing start/stop time columns in {path.name}")

    start_id  = next((c for c in ["start_station_id","from_station_id","startstationid"] if c in df.columns), None)
    end_id    = next((c for c in ["end_station_id","to_station_id","endstationid"] if c in df.columns), None)
    start_nm  = next((c for c in ["start_station_name","from_station_name"] if c in df.columns), None)
    end_nm    = next((c for c in ["end_station_name","to_station_name"] if c in df.columns), None)

    slat = next((c for c in ["start_lat","start station latitude","startlatitude"] if c in df.columns), None)
    slng = next((c for c in ["start_lng","start station longitude","startlongitude"] if c in df.columns), None)
    elat = next((c for c in ["end_lat","end station latitude","endlatitude"] if c in df.columns), None)
    elng = next((c for c in ["end_lng","end station longitude","endlongitude"] if c in df.columns), None)

    member = next((c for c in ["member_casual","usertype"] if c in df.columns), None)

    # Parse datetimes
    df["started_at"] = pd.to_datetime(df[started], errors="coerce")
    df["ended_at"]   = pd.to_datetime(df[ended], errors="coerce")

    # Duration (minutes)
    df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60.0

    # Distance (miles) if coords exist
    if all(col is not None for col in [slat, slng, elat, elng]):
        lat1 = pd.to_numeric(df[slat], errors="coerce")
        lon1 = pd.to_numeric(df[slng], errors="coerce")
        lat2 = pd.to_numeric(df[elat], errors="coerce")
        lon2 = pd.to_numeric(df[elng], errors="coerce")
        km = haversine_km(lat1, lon1, lat2, lon2)
        df["distance_miles"] = km * 0.621371
    else:
        df["distance_miles"] = np.nan

    # Member type
    if member:
        s = df[member].astype(str).str.lower()
        df["member_type"] = np.where(
            s.str.contains("member|subscriber"), "member",
            np.where(s.str.contains("casual|customer"), "casual", "unknown")
        )
    else:
        df["member_type"] = "unknown"

    # Keep tidy columns
    keep = {
        "started_at","ended_at","duration_min","distance_miles","member_type",
        start_id, end_id, start_nm, end_nm
    } - {None}
    out = df[list(keep)].copy()

    # Rename for consistency
    # Keep tidy columns + coords if present
    keep = {
        "started_at","ended_at","duration_min","distance_miles","member_type",
        start_id, end_id, start_nm, end_nm, slat, slng, elat, elng
    } - {None}
    out = df[list(keep)].copy()
    
    # Standardize names
    if start_id: out.rename(columns={start_id:"start_station_id"}, inplace=True)
    if end_id:   out.rename(columns={end_id:"end_station_id"}, inplace=True)
    if start_nm: out.rename(columns={start_nm:"start_station_name"}, inplace=True)
    if end_nm:   out.rename(columns={end_nm:"end_station_name"}, inplace=True)
    if slat:     out.rename(columns={slat:"start_lat"}, inplace=True)
    if slng:     out.rename(columns={slng:"start_lng"}, inplace=True)
    if elat:     out.rename(columns={elat:"end_lat"}, inplace=True)
    if elng:     out.rename(columns={elng:"end_lng"}, inplace=True)
    
    return out

def main():
    files = sorted(list(RAW_DIR.glob("*.csv"))) + sorted(list(RAW_DIR.glob("*.csv.gz"))) + sorted(list(RAW_DIR.glob("*.zip")))
    if not files:
        raise SystemExit(f"No Citi Bike files found in {RAW_DIR}. Put monthly trip CSVs there (e.g., 2024-08-citibike-tripdata.csv.gz).")

    with sqlite3.connect(DB_PATH) as con:
        con.execute("DROP TABLE IF EXISTS trips;")
        total_rows = 0
        for f in files:
            df = load_one(f)
            df.to_sql("trips", con, if_exists="append", index=False)
            total_rows += len(df)
            print(f"Ingested {len(df):,} rows from {f.name}")
        con.executescript("""
            CREATE INDEX IF NOT EXISTS idx_trips_started ON trips(started_at);
            CREATE INDEX IF NOT EXISTS idx_trips_member ON trips(member_type);
            CREATE INDEX IF NOT EXISTS idx_trips_start_station ON trips(start_station_id);
        """)
        print(f"Wrote table trips to {DB_PATH} with {total_rows:,} rows")

if __name__ == "__main__":
    main()
