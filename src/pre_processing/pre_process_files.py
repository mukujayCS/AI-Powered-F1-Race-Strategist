import pandas as pd
import json
from pathlib import Path
import numpy as np

# Paths
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def process_circuits(races_df, circuits_df):
    # Get circuitIds from 2022–2024 races
    circuit_ids = races_df["circuitId"].unique()
    filtered = circuits_df[circuits_df["circuitId"].isin(circuit_ids)]

    # Select only needed columns
    circuits_out = filtered[["circuitId", "name"]].to_dict(orient="records")

    # Save to JSON
    with open(PROCESSED_DATA_DIR / "22-24_circuits.json", "w") as f:
        json.dump(circuits_out, f, indent=4)

    print(f"Saved {len(circuits_out)} circuits to data/processed/22-24_circuits.json")


def process_drivers(races_df, results_df, drivers_df):
    # Get raceIds for 2022–2024
    race_ids = races_df["raceId"].unique()

    # Get driverIds from results for those races
    driver_ids = results_df[results_df["raceId"].isin(race_ids)]["driverId"].unique()
    filtered = drivers_df[drivers_df["driverId"].isin(driver_ids)]

    # Select only needed columns
    drivers_out = filtered[["driverId", "driverRef", "number", "code"]].to_dict(orient="records")

    # Save to JSON
    with open(PROCESSED_DATA_DIR / "22-24_drivers.json", "w") as f:
        json.dump(drivers_out, f, indent=4)

    print(f"Saved {len(drivers_out)} drivers to data/processed/22-24_drivers.json")


def process_lap_times(races_df, driver_ids, lap_times_df):
    race_ids = races_df["raceId"].unique()
    # Filter lap times by race and driver
    # filtered = lap_times_df[
    #     (lap_times_df["raceId"].isin(race_ids)) &
    #     (lap_times_df["driverId"].isin(driver_ids))
    # ]

    filtered = lap_times_df[
        (lap_times_df["raceId"].isin(race_ids)) &
        (lap_times_df["driverId"].isin(driver_ids))
    ].copy()
    
    # Convert milliseconds to seconds
    filtered.loc[:, "lapTime_seconds"] = filtered["milliseconds"] / 1000.0

    # Convert milliseconds to seconds
    filtered["lapTime_seconds"] = filtered["milliseconds"] / 1000.0

    # Keep only required columns
    lap_times_out = filtered[["raceId", "driverId", "lap", "position", "time", "lapTime_seconds"]] \
        .to_dict(orient="records")

    with open(PROCESSED_DATA_DIR / "22-24_lap_times.json", "w") as f:
        json.dump(lap_times_out, f, indent=4)

    print(f"Saved {len(lap_times_out)} lap time records to data/processed/22-24_lap_times.json")


def process_races(races_df):
    races_out = races_df[["raceId", "year", "round", "name", "circuitId"]].to_dict(orient="records")

    with open(PROCESSED_DATA_DIR / "22-24_races.json", "w") as f:
        json.dump(races_out, f, indent=4)

    print(f"Saved {len(races_out)} races to data/processed/22-24_races.json")


def process_qualifying(races_filtered, driver_ids, qualifying_df):
    # Filter relevant races + drivers
    qualifying_filtered = qualifying_df[
        (qualifying_df["raceId"].isin(races_filtered["raceId"])) &
        (qualifying_df["driverId"].isin(driver_ids))
    ].copy()
    
    # Helper function to convert lap times from "M:SS.mmm" to seconds
    def convert_lap_time(time_str):
        if pd.isna(time_str):
            return np.nan
        try:
            # Split on colon to get minutes and seconds
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return np.nan
        except:
            return np.nan
    
    # Replace '\N' with np.nan so pandas can parse them properly
    qualifying_filtered.replace("\\N", np.nan, inplace=True)
    
    # Convert q1, q2, q3 to numeric lap times in seconds
    for col in ["q1", "q2", "q3"]:
        qualifying_filtered[col] = qualifying_filtered[col].apply(convert_lap_time)
    
    # Compute best lap (fastest/minimum time, ignores NaN automatically)
    qualifying_filtered["bestQualiTime"] = qualifying_filtered[["q1", "q2", "q3"]].min(axis=1)
    
    # Keep only required columns
    qualifying_processed = qualifying_filtered[["raceId", "driverId", "position", "bestQualiTime"]]
    
    # Handle missing qualifying times with smart penalty strategy
    for race_id in qualifying_processed["raceId"].unique():
        race_mask = qualifying_processed["raceId"] == race_id
        race_data = qualifying_processed[race_mask].copy()
        
        # Get drivers with valid qualifying times, sorted by position
        valid_qualifiers = race_data.dropna(subset=["bestQualiTime"]).sort_values("position")
        
        if len(valid_qualifiers) == 0:
            # Edge case: no valid times in this race, skip
            continue
            
        # Determine penalty time based on available data
        penalty_time = None
        
        if len(valid_qualifiers) >= 19:
            # Use P19 time + 2% penalty (realistic for DNQ drivers)
            p19_time = valid_qualifiers.iloc[18]["bestQualiTime"]  # 19th position (0-indexed)
            penalty_time = p19_time * 1.02
        elif len(valid_qualifiers) >= 15:
            # If less than 19 cars, use slowest time + 3% penalty
            slowest_time = valid_qualifiers["bestQualiTime"].max()
            penalty_time = slowest_time * 1.03
        else:
            # Very few cars qualified - use mean + 1 standard deviation
            valid_times = valid_qualifiers["bestQualiTime"]
            penalty_time = valid_times.mean() + valid_times.std()
        
        # Apply penalty to drivers with missing qualifying times
        if penalty_time:
            nan_mask = race_mask & qualifying_processed["bestQualiTime"].isna()
            qualifying_processed.loc[nan_mask, "bestQualiTime"] = penalty_time
    
    # Remove any remaining NaN values (should be very rare)
    initial_count = len(qualifying_processed)
    qualifying_processed = qualifying_processed.dropna(subset=["bestQualiTime"])
    final_count = len(qualifying_processed)
    
    if initial_count > final_count:
        print(f"Warning: Dropped {initial_count - final_count} entries with unresolvable missing times")
    
    # Convert to list of dicts for JSON
    qualy_out = qualifying_processed.to_dict(orient="records")
    
    # Save in pretty JSON format
    output_file = PROCESSED_DATA_DIR / "22-24_qualifying.json"
    with open(output_file, "w") as f:
        json.dump(qualy_out, f, indent=4)
    
    print(f"Saved {len(qualy_out)} qualifying entries to {output_file}")
    print(f"Applied smart penalty strategy for missing qualifying times")


# processing the pitstops
def process_pitstops(races_file, drivers_file, pitstops_df):
    # Load races and drivers from the already-processed JSONs
    with open(races_file, "r") as f:
        races_data = json.load(f)
    with open(drivers_file, "r") as f:
        drivers_data = json.load(f)

    # Extract raceIds and driverIds
    race_ids = [r["raceId"] for r in races_data]
    driver_ids = [d["driverId"] for d in drivers_data]

    # Filter pit stops to only relevant races + drivers
    filtered = pitstops_df[
        (pitstops_df["raceId"].isin(race_ids)) &
        (pitstops_df["driverId"].isin(driver_ids))
    ].copy()

    # Convert milliseconds → seconds
    filtered["stop_time_seconds"] = filtered["milliseconds"] / 1000.0

    # Select required fields
    pitstops_out = filtered[["raceId", "driverId", "stop", "lap", "stop_time_seconds"]] \
        .to_dict(orient="records")

    # Save to JSON
    output_file = PROCESSED_DATA_DIR / "22-24_pitstops.json"
    with open(output_file, "w") as f:
        json.dump(pitstops_out, f, indent=4)

    print(f"Saved {len(pitstops_out)} pit stop records to {output_file}")


def main():
    # Load raw CSVs
    races_df = pd.read_csv(RAW_DATA_DIR / "races.csv")
    circuits_df = pd.read_csv(RAW_DATA_DIR / "circuits.csv")
    drivers_df = pd.read_csv(RAW_DATA_DIR / "drivers.csv")
    results_df = pd.read_csv(RAW_DATA_DIR / "results.csv")
    lap_times_df = pd.read_csv(RAW_DATA_DIR / "lap_times.csv")
    qualifying_df = pd.read_csv(RAW_DATA_DIR / "qualifying.csv")
    pitstops_df = pd.read_csv(RAW_DATA_DIR / "pit_stops.csv")

    # Filter races from 2022–2024
    races_filtered = races_df[races_df["year"].between(2022, 2024)]

    driver_ids = (
        results_df[results_df["raceId"].isin(races_df["raceId"].unique())]["driverId"].unique().tolist() 
    )

    # Process and save outputs
    process_circuits(races_filtered, circuits_df)
    process_drivers(races_filtered, results_df, drivers_df)
    process_lap_times(races_filtered, driver_ids, lap_times_df)
    process_races(races_filtered)
    process_qualifying(races_filtered, driver_ids, qualifying_df)
    process_pitstops(PROCESSED_DATA_DIR / "22-24_races.json",
                     PROCESSED_DATA_DIR / "22-24_drivers.json",
                     pitstops_df)


if __name__ == "__main__":
    main()
