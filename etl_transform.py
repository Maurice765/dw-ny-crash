import argparse
import os
from datetime import datetime, timedelta
import re

import numpy as np
import pandas as pd


def _normalize_str(x: object) -> object:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return s if s else np.nan


def _normalize_col_name(name: object) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^0-9a-z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    used: set[str] = set()
    mapping: dict[object, str] = {}
    for c in df.columns:
        nc = _normalize_col_name(c)
        if not nc:
            nc = "col"
        if nc in used:
            i = 2
            while f"{nc}_{i}" in used:
                i += 1
            nc = f"{nc}_{i}"
        used.add(nc)
        mapping[c] = nc
    return df.rename(columns=mapping)


def _col_or_na(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([np.nan] * len(df))


def _normalize_join_id(series: pd.Series) -> pd.Series:
    s = series.copy()
    s = s.map(_normalize_str)
    s = s.astype(object)
    return s


def _parse_maybe_timestamp(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series([], dtype="datetime64[ns]")
    s = series.copy()

    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().mean() >= 0.8:
        return dt

    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().sum() == 0:
        return dt

    median = float(s_num.dropna().median())
    unit = None
    if median > 1e12:
        unit = "ms"
    elif median > 1e9:
        unit = "s"

    if unit is not None:
        dt2 = pd.to_datetime(s_num, errors="coerce", unit=unit, utc=True).dt.tz_convert(None)
        if dt2.notna().mean() > dt.notna().mean():
            return dt2

    return dt


def _parse_time(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series([], dtype=object)
    s = series.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "NaN": np.nan, "": np.nan})

    dt = pd.to_datetime(s, errors="coerce", format="%H:%M")
    if dt.notna().mean() >= 0.5:
        return dt.dt.time

    dt2 = pd.to_datetime(s, errors="coerce")
    return dt2.dt.time


def _safe_float(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series([], dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _safe_int(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series([], dtype="Int64")
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def _load_boundaries(path: str):
    try:
        import geopandas as gpd
    except ImportError as e:
        raise RuntimeError(
            "Geo boundary file specified, but geopandas is not installed. Install dependencies or run without --borough-boundaries/--precinct-boundaries."
        ) from e

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    else:
        gdf = gdf.to_crs(4326)
    return gdf


def _detect_field_case_insensitive(gdf, candidates: list[str]) -> str | None:
    cols = list(gdf.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        cl = str(c).lower()
        if cl in lower_map:
            return lower_map[cl]
    return None


def _infer_from_boundaries(crashes: pd.DataFrame, borough_boundaries: str | None, precinct_boundaries: str | None,
                           borough_field: str | None, precinct_id_field: str | None, precinct_name_field: str | None,
                           precinct_borough_field: str | None) -> pd.DataFrame:

    # Coordinates must ALWAYS be float for downstream merges to work
    df = crashes.copy()
    df["latitude"] = _safe_float(df.get("latitude"))
    df["longitude"] = _safe_float(df.get("longitude"))

    if borough_boundaries is None and precinct_boundaries is None:
        return df

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as e:
        raise RuntimeError(
            "Geo boundary file specified, but geopandas/shapely are not installed. Install dependencies oder run without boundaries."
        ) from e

    mask = df["latitude"].notna() & df["longitude"].notna()
    if mask.sum() == 0:
        return df

    pts = gpd.GeoDataFrame(
        df.loc[mask].copy(),
        geometry=[Point(xy) for xy in zip(df.loc[mask, "longitude"], df.loc[mask, "latitude"])],
        crs=4326,
    )

    if borough_boundaries is not None:
        boros = _load_boundaries(borough_boundaries)
        if borough_field is None:
            candidates = ["borough", "boro_name", "BoroName", "boro_nm", "BoroName", "name", "NAME"]
            borough_field = _detect_field_case_insensitive(boros, candidates)
        if borough_field is None:
            raise RuntimeError("Could not detect borough name field. Provide it via --borough-field.")

        boros_join = boros[[borough_field, "geometry"]].rename(columns={borough_field: "_bnd_borough_name"})
        joined = gpd.sjoin(pts, boros_join, how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]
        df.loc[joined.index, "_inferred_borough"] = joined["_bnd_borough_name"].values

    if precinct_boundaries is not None:
        prec = _load_boundaries(precinct_boundaries)

        if precinct_id_field is None:
            candidates = ["precinct", "Precinct", "pct", "PCT", "precinct_id", "PrecinctID"]
            precinct_id_field = _detect_field_case_insensitive(prec, candidates)
        if precinct_id_field is None:
            raise RuntimeError("Could not detect precinct id field. Provide it via --precinct-id-field.")

        if precinct_name_field is None:
            candidates = ["precinct", "Precinct", "name", "NAME", "pct", "PCT"]
            precinct_name_field = _detect_field_case_insensitive(prec, candidates) or precinct_id_field

        cols = [precinct_id_field, precinct_name_field]
        if precinct_borough_field is None:
            candidates = ["borough", "boro_name", "BoroName", "boro_nm", "BORO_NM"]
            precinct_borough_field = _detect_field_case_insensitive(prec, candidates)
        if precinct_borough_field is not None:
            cols.append(precinct_borough_field)

        src_cols: list[str] = []
        for c in [precinct_id_field, precinct_name_field, precinct_borough_field]:
            if c is not None and c not in src_cols:
                src_cols.append(c)

        prec_join = prec[src_cols + ["geometry"]].copy()
        prec_join["_bnd_precinct_id"] = prec[precinct_id_field]
        prec_join["_bnd_precinct_name"] = prec[precinct_name_field]
        if precinct_borough_field is not None:
            prec_join["_bnd_precinct_borough"] = prec[precinct_borough_field]

        keep_cols = ["_bnd_precinct_id", "_bnd_precinct_name"]
        if precinct_borough_field is not None:
            keep_cols.append("_bnd_precinct_borough")
        prec_join = prec_join[keep_cols + ["geometry"]]
        joined = gpd.sjoin(pts, prec_join, how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]
        df.loc[joined.index, "_inferred_precinct_id"] = joined["_bnd_precinct_id"].values
        df.loc[joined.index, "_inferred_precinct_name"] = joined["_bnd_precinct_name"].values
        if precinct_borough_field is not None:
            df.loc[joined.index, "_inferred_precinct_borough"] = joined["_bnd_precinct_borough"].values

    return df


def _build_dimension_ids(values: pd.Series, start: int = 1) -> tuple[pd.DataFrame, dict[object, int]]:
    uniq = pd.Series(values.dropna().unique()).sort_values(kind="stable")
    ids = pd.Series(range(start, start + len(uniq)), index=uniq.values)
    mapping = ids.to_dict()
    dim = pd.DataFrame({"_value": uniq.values, "_id": [mapping[v] for v in uniq.values]})
    return dim, mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Transform NY crash/person/vehicle/weather CSVs into normalized CSVs matching tables-creation.sql")
    parser.add_argument("--crashes", required=True, help="Path to crashes.csv")
    parser.add_argument("--persons", required=True, help="Path to person.csv")
    parser.add_argument("--vehicles", required=True, help="Path to vehicles.csv")
    parser.add_argument("--weather", required=True, help="Path to weather.csv")
    parser.add_argument("--out", required=True, help="Output directory for generated CSVs")

    parser.add_argument("--borough-boundaries", default=None, help="Optional GeoJSON/Shapefile path with borough polygons")
    parser.add_argument("--borough-field", default=None, help="Column name in borough boundaries that contains borough name")

    parser.add_argument("--precinct-boundaries", default=None, help="Optional GeoJSON/Shapefile path with precinct polygons")
    parser.add_argument("--precinct-id-field", default=None, help="Column name in precinct boundaries that contains precinct id")
    parser.add_argument("--precinct-name-field", default=None, help="Column name in precinct boundaries that contains precinct name")
    parser.add_argument("--precinct-borough-field", default=None, help="Optional column in precinct boundaries that contains borough name")

    parser.add_argument("--encoding", default="utf-8", help="CSV encoding")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    crashes = _normalize_columns(pd.read_csv(args.crashes, encoding=args.encoding, low_memory=False))
    persons = _normalize_columns(pd.read_csv(args.persons, encoding=args.encoding, low_memory=False))
    vehicles = _normalize_columns(pd.read_csv(args.vehicles, encoding=args.encoding, low_memory=False))
    weather_raw = _normalize_columns(pd.read_csv(args.weather, encoding=args.encoding, low_memory=False))

    for col in crashes.columns:
        if crashes[col].dtype == object:
            crashes[col] = crashes[col].map(_normalize_str)
    for col in persons.columns:
        if persons[col].dtype == object:
            persons[col] = persons[col].map(_normalize_str)
    for col in vehicles.columns:
        if vehicles[col].dtype == object:
            vehicles[col] = vehicles[col].map(_normalize_str)

    crashes = _infer_from_boundaries(
        crashes,
        borough_boundaries=args.borough_boundaries,
        precinct_boundaries=args.precinct_boundaries,
        borough_field=args.borough_field,
        precinct_id_field=args.precinct_id_field,
        precinct_name_field=args.precinct_name_field,
        precinct_borough_field=args.precinct_borough_field,
    )

    crashes["_crash_dt"] = _parse_maybe_timestamp(_col_or_na(crashes, "crash_date"))
    crashes["_crash_time"] = _parse_time(_col_or_na(crashes, "crash_time"))

    crashes["Crash_Date"] = pd.to_datetime(crashes["_crash_dt"], errors="coerce").dt.date
    crashes["Crash_Time"] = crashes["_crash_time"]

    if crashes.get("_inferred_precinct_id") is not None:
        crashes["Precinct_ID"] = _safe_int(crashes["_inferred_precinct_id"]).astype("Int64")
    else:
        crashes["Precinct_ID"] = pd.Series([pd.NA] * len(crashes), dtype="Int64")

    borough_series = _col_or_na(crashes, "borough")

    inferred_boro = crashes.get("_inferred_borough")
    inferred_precinct_boro = crashes.get("_inferred_precinct_borough")

    borough_final = borough_series.copy()
    if inferred_boro is not None:
        borough_final = borough_final.fillna(inferred_boro)
    if inferred_precinct_boro is not None:
        borough_final = borough_final.fillna(inferred_precinct_boro)

    borough_dim, borough_map = _build_dimension_ids(borough_final)
    borough_out = pd.DataFrame(
        {
            "Borough_ID": borough_dim["_id"].astype(int),
            "Borough_Name": borough_dim["_value"],
        }
    )
    borough_out.to_csv(os.path.join(args.out, "Borough.csv"), index=False)

    precinct_id = crashes.get("_inferred_precinct_id")
    precinct_name = crashes.get("_inferred_precinct_name")

    precinct_df = pd.DataFrame({"Precinct_ID": precinct_id, "Precinct_Name": precinct_name, "Borough_Name": borough_final})
    precinct_df = precinct_df.dropna(subset=["Precinct_ID"]).copy()

    if len(precinct_df) > 0:
        precinct_df["Precinct_ID"] = _safe_int(precinct_df["Precinct_ID"])
        precinct_df = precinct_df.dropna(subset=["Precinct_ID"])
        precinct_df["Precinct_Name"] = precinct_df["Precinct_Name"].fillna(precinct_df["Precinct_ID"].astype(str))

        precinct_unique = (
            precinct_df.groupby("Precinct_ID", as_index=False)
            .agg({"Precinct_Name": "first", "Borough_Name": lambda x: x.dropna().mode().iloc[0] if x.dropna().size else np.nan})
        )
        precinct_unique["Borough_ID"] = precinct_unique["Borough_Name"].map(borough_map).astype("Int64")

        precinct_out = precinct_unique[["Precinct_ID", "Precinct_Name", "Borough_ID"]].copy()
        precinct_out["Precinct_ID"] = precinct_out["Precinct_ID"].astype(int)
    else:
        precinct_out = pd.DataFrame({"Precinct_ID": pd.Series(dtype=int), "Precinct_Name": pd.Series(dtype=str), "Borough_ID": pd.Series(dtype="Int64")})

    precinct_out.to_csv(os.path.join(args.out, "Precinct.csv"), index=False)

    location_cols = {
        "Longitude": crashes["longitude"],  # Already safe float from _infer_from_boundaries
        "Latitude": crashes["latitude"],    # Already safe float from _infer_from_boundaries
        "Zip_Code": _col_or_na(crashes, "zip_code"),
        "Precinct_ID": crashes["Precinct_ID"],
    }
    location_tmp = pd.DataFrame(location_cols)

    loc_unique = location_tmp.drop_duplicates().reset_index(drop=True)
    loc_unique.insert(0, "Location_ID", pd.Series(range(1, len(loc_unique) + 1), dtype=int))

    location_out = loc_unique[["Location_ID", "Longitude", "Latitude", "Zip_Code", "Precinct_ID"]]
    location_out.to_csv(os.path.join(args.out, "Location.csv"), index=False)

    crashes = crashes.merge(
        location_out,
        left_on=["longitude", "latitude", "zip_code", "Precinct_ID"],
        right_on=["Longitude", "Latitude", "Zip_Code", "Precinct_ID"],
        how="left",
    )

    weather = weather_raw.copy()
    weather_dt = pd.to_datetime(_col_or_na(weather, "date"), errors="coerce")
    weather["Measure_Date"] = weather_dt.dt.date
    weather["Measure_Time"] = weather_dt.dt.floor("h").dt.time

    temp_f = _safe_float(_col_or_na(weather, "hourlydrybulbtempf"))
    weather["Temp_Celsius"] = (temp_f - 32) * (5.0 / 9.0)

    weather["Visibility_Miles"] = _safe_float(_col_or_na(weather, "hourlyvisibility"))
    weather["Precipitation_Inches"] = _safe_float(_col_or_na(weather, "hourlyprecip"))

    weather["Snow_Depth_Inches"] = _safe_float(_col_or_na(weather, "hourlysnowdepth")) if "hourlysnowdepth" in weather.columns else np.nan
    weather["Wind_Gust_Speed_MPH"] = _safe_float(_col_or_na(weather, "hourlywindspeed"))

    weather["Weather_Condition_Text"] = _col_or_na(weather, "hourlyweathertype") if "hourlyweathertype" in weather.columns else np.nan

    weather_key_cols = [
        "Measure_Date",
        "Measure_Time",
        "Temp_Celsius",
        "Visibility_Miles",
        "Precipitation_Inches",
        "Snow_Depth_Inches",
        "Wind_Gust_Speed_MPH",
        "Weather_Condition_Text",
    ]
    weather_unique = weather[weather_key_cols].drop_duplicates().reset_index(drop=True)
    weather_unique.insert(0, "Weather_ID", pd.Series(range(1, len(weather_unique) + 1), dtype=int))

    weather_out = weather_unique[
        [
            "Weather_ID",
            "Measure_Date",
            "Measure_Time",
            "Temp_Celsius",
            "Visibility_Miles",
            "Precipitation_Inches",
            "Snow_Depth_Inches",
            "Wind_Gust_Speed_MPH",
            "Weather_Condition_Text",
        ]
    ]
    weather_out.to_csv(os.path.join(args.out, "Weather.csv"), index=False)

    crash_date = pd.to_datetime(crashes["Crash_Date"], errors="coerce")
    crash_time_dt = pd.to_datetime(crashes["Crash_Time"].astype(str), errors="coerce")
    crash_time_delta = pd.to_timedelta(crash_time_dt.dt.hour.fillna(0).astype(int), unit="h") + pd.to_timedelta(
        crash_time_dt.dt.minute.fillna(0).astype(int), unit="m"
    )

    crash_dt = crash_date + crash_time_delta
    crash_floor_time = crash_dt.dt.floor("h")

    crashes["_wx_date"] = crash_floor_time.dt.date
    crashes["_wx_time"] = crash_floor_time.dt.time

    crashes = crashes.merge(
        weather_out,
        left_on=["_wx_date", "_wx_time"],
        right_on=["Measure_Date", "Measure_Time"],
        how="left",
        suffixes=("", "_wx"),
    )

    crash_out = pd.DataFrame(
        {
            "Collision_ID": _safe_int(_col_or_na(crashes, "collision_id")),
            "Crash_Date": crashes["Crash_Date"],
            "Crash_Time": crashes["Crash_Time"],
            "Location_ID": crashes["Location_ID"].astype("Int64"),
            "Weather_ID": crashes["Weather_ID"].astype("Int64"),
        }
    )
    crash_out = crash_out.dropna(subset=["Collision_ID"]).copy()
    crash_out["Collision_ID"] = crash_out["Collision_ID"].astype(int)
    crash_out.to_csv(os.path.join(args.out, "Crash.csv"), index=False)

    vehicle_type_values = []
    for c in [
        "vehicle_type_code1",
        "vehicle_type_code2",
        "vehicle_type_code_3",
        "vehicle_type_code_4",
        "vehicle_type_code_5",
    ]:
        if c in crashes.columns:
            vehicle_type_values.append(crashes[c])
    if "vehicle_type" in vehicles.columns:
        vehicle_type_values.append(vehicles["vehicle_type"])

    if vehicle_type_values:
        vehicle_type_series = pd.concat(vehicle_type_values, ignore_index=True).map(_normalize_str)
    else:
        vehicle_type_series = pd.Series([], dtype=object)

    vt_dim, vt_map = _build_dimension_ids(vehicle_type_series)
    vehicle_type_out = pd.DataFrame(
        {
            "Vehicle_Type_ID": vt_dim["_id"].astype(int),
            "Vehicle_Type_Name": vt_dim["_value"],
            "Vehicle_Type_Category": pd.Series([np.nan] * len(vt_dim)),
        }
    )
    vehicle_type_out.to_csv(os.path.join(args.out, "Vehicle_Type.csv"), index=False)

    factor_values = []
    for c in [
        "contributing_factor_vehicle_1",
        "contributing_factor_vehicle_2",
        "contributing_factor_vehicle_3",
        "contributing_factor_vehicle_4",
        "contributing_factor_vehicle_5",
    ]:
        if c in crashes.columns:
            factor_values.append(crashes[c])

    for c in ["contributing_factor_1", "contributing_factor_2"]:
        if c in vehicles.columns:
            factor_values.append(vehicles[c])
        if c in persons.columns:
            factor_values.append(persons[c])

    if factor_values:
        factor_series = pd.concat(factor_values, ignore_index=True).map(_normalize_str)
    else:
        factor_series = pd.Series([], dtype=object)

    cf_dim, cf_map = _build_dimension_ids(factor_series)
    contributing_factor_out = pd.DataFrame(
        {
            "Factor_ID": cf_dim["_id"].astype(int),
            "Factor_Name": cf_dim["_value"],
            "Factor_Category": pd.Series([np.nan] * len(cf_dim)),
        }
    )
    contributing_factor_out.to_csv(os.path.join(args.out, "Contributing_Factor.csv"), index=False)

    vehicle_out = pd.DataFrame(
        {
            "Vehicle_ID": _safe_int(_col_or_na(vehicles, "unique_id")),
            "Collision_ID": _safe_int(_col_or_na(vehicles, "collision_id")),
            "State_Registration": _col_or_na(vehicles, "state_registration"),
            "Vehicle_Year": _safe_int(_col_or_na(vehicles, "vehicle_year")),
            "Vehicle_Type_ID": vehicles.get("vehicle_type").map(vt_map).astype("Int64") if "vehicle_type" in vehicles.columns else pd.Series([pd.NA] * len(vehicles), dtype="Int64"),
        }
    )
    vehicle_out = vehicle_out.dropna(subset=["Vehicle_ID"]).copy()
    vehicle_out["Vehicle_ID"] = vehicle_out["Vehicle_ID"].astype(int)
    if "Collision_ID" in vehicle_out.columns:
        vehicle_out["Collision_ID"] = vehicle_out["Collision_ID"].astype("Int64")
    vehicle_out.to_csv(os.path.join(args.out, "Vehicle.csv"), index=False)

    vf_rows = []
    if len(vehicles) > 0:
        vid = _safe_int(vehicles.get("unique_id"))
        for col in ["contributing_factor_1", "contributing_factor_2"]:
            if col in vehicles.columns:
                fac = vehicles[col].map(_normalize_str)
                for v, f in zip(vid, fac):
                    if pd.isna(v) or pd.isna(f):
                        continue
                    fid = cf_map.get(f)
                    if fid is None:
                        continue
                    vf_rows.append((int(v), int(fid)))

    vehicle_factors_out = pd.DataFrame(vf_rows, columns=["Vehicle_ID", "Factor_ID"]).drop_duplicates()
    vehicle_factors_out.to_csv(os.path.join(args.out, "Vehicle_Factors.csv"), index=False)

    vehicles_key = vehicles[[c for c in ["collision_id", "vehicle_id", "unique_id"] if c in vehicles.columns]].copy()
    if set(["collision_id", "vehicle_id", "unique_id"]).issubset(vehicles_key.columns):
        vehicles_key["collision_id"] = _safe_int(vehicles_key["collision_id"])
        vehicles_key["unique_id"] = _safe_int(vehicles_key["unique_id"])
        vehicles_key["vehicle_id_str"] = _normalize_join_id(vehicles_key["vehicle_id"])

    # Fallback für person_id, da NYPD in manchen Datensätzen person_id statt unique_id verwendet
    persons_id_series = _col_or_na(persons, "person_id") if "person_id" in persons.columns else _col_or_na(persons, "unique_id")

    persons_out = pd.DataFrame(
        {
            "Person_ID": _safe_int(persons_id_series),
            "Collision_ID": _safe_int(_col_or_na(persons, "collision_id")),
            "Vehicle_ID": pd.Series([pd.NA] * len(persons), dtype="Int64"),
            "Person_Type": _col_or_na(persons, "person_type"),
            "Person_Role": persons.get("ped_role") if "ped_role" in persons.columns else _col_or_na(persons, "position_in_vehicle"),
            "Person_Injury": _col_or_na(persons, "person_injury"),
            "Person_Age": _safe_int(_col_or_na(persons, "person_age")),
            "Person_Sex": _col_or_na(persons, "person_sex"),
        }
    )

    if set(["collision_id", "vehicle_id", "unique_id"]).issubset(vehicles.columns) and "vehicle_id" in persons.columns:
        tmp = persons[["collision_id", "vehicle_id"]].copy()
        tmp["_orig_index"] = tmp.index  # WICHTIG: Sichert den ursprünglichen Index für sauberes Alignment
        tmp["collision_id"] = _safe_int(tmp["collision_id"])
        tmp["vehicle_id_str"] = _normalize_join_id(tmp["vehicle_id"])

        vk = vehicles_key.rename(columns={"unique_id": "_vehicle_unique_id"})
        vk_unique = vk[["collision_id", "vehicle_id_str", "_vehicle_unique_id"]].drop_duplicates(subset=["collision_id", "vehicle_id_str"])

        tmp = tmp.merge(
            vk_unique,
            on=["collision_id", "vehicle_id_str"],
            how="left",
        )
        tmp = tmp.set_index("_orig_index")  # Index wiederherstellen
        persons_out["Vehicle_ID"] = _safe_int(tmp["_vehicle_unique_id"]).astype("Int64")

    persons_out = persons_out.dropna(subset=["Person_ID"]).copy()
    persons_out["Person_ID"] = persons_out["Person_ID"].astype(int)
    persons_out.to_csv(os.path.join(args.out, "Person.csv"), index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())