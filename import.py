import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import requests
import json
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore') # Unterdrückt Warnungen bei der Entfernungsberechnung

# --- Konfiguration ---
CRASHES_FILE = 'data/crashes.csv'
VEHICLES_FILE = 'data/vehicles.csv'
PERSONS_FILE = 'data/persons.csv'
PRECINCTS_FILE = 'data/precincts.geojson'  
OUTPUT_DIR = 'output_tables/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Wetterstationen (Die 3 echten Sensoren) ---
STATIONS = [
    {"Weather_Station": "KLGA", "name": "LaGuardia Airport", "lat": 40.7769, "lon": -73.8740},
    {"Weather_Station": "KJFK", "name": "JFK Airport", "lat": 40.6413, "lon": -73.7781},
    {"Weather_Station": "KEWR", "name": "Newark Airport", "lat": 40.6895, "lon": -74.1745}
]

# --- Hilfsfunktionen ---
def get_fixed_vehicle_category(v_type):
    if not isinstance(v_type, str): return 'Other / Unknown'
    v = v_type.upper().strip()
    if any(k in v for k in ['AMBULANCE', 'AMB', 'EMS', 'EMT', 'FIRE', 'FDNY', 'NYFD', 'POLICE', 'NYPD', 'EMERGENCY', 'AMU', 'RESCUE', 'PATROL', 'RMP', 'GOV', 'ARMY', 'LADDER']): return 'Emergency / Medical / Police'
    if any(k in v for k in ['BUS', 'MTA', 'OMNIBUS', 'SCH', 'SCL', 'COACH', 'ACCESS', 'SHUTTLE', 'TRANSI']): return 'Bus / School Bus'
    if any(k in v for k in ['TRUCK', 'TRK', 'TRU', 'TRACT', 'TRAC', 'TRAIL', 'BOX', 'DUMP', 'PICK', 'P/U', 'DELIV', 'DELV', 'COURIER', 'FEDEX', 'FED E', 'FEDERAL EX', 'UPS', 'MAIL', 'POST', 'USPS', 'COMMERCIAL', 'COM', 'FREIGHT', 'GARB', 'GARGAGE', 'SANITA', 'U-HAUL', 'U HAUL', 'UHAUL', 'U-HAL', 'VENDOR', 'FORK', 'CRANE', 'FLAT', 'TOW', 'CHASSIS', 'UTIL', 'BOBCAT', 'BACKHOE', 'SWEEP', 'BROOM', 'LIFT', 'ICE CREAM']): return 'Truck / Commercial / Delivery'
    if any(k in v for k in ['TAXI', 'CAB', 'LIMO', 'LIVERY', 'UBER', 'LYFT']): return 'Taxi / Livery'
    if any(k in v for k in ['MOTOR', 'MOPED', 'MOP', 'SCOOT', 'SCOT', 'SCO', 'DIRT', 'MOTO', 'E-SCO', 'VESPA', 'E-SKA']): return 'Motorcycle / Moped / Scooter'
    if any(k in v for k in ['BIKE', 'BICYCLE', 'E-BIKE', 'CYCL', 'ELECTRIC', 'ELETRIC', 'E-BI']): return 'Bicycle / E-Bike'
    if any(k in v for k in ['VAN', 'VAHN', 'MINI', 'TRANSIT', 'SPRIN', 'ECONO']): return 'Van / Minivan'
    if any(k in v for k in ['SUV', 'SPORT', 'STATION', 'SUBUR', 'SUBN', 'JEEP']): return 'SUV / Station Wagon'
    if any(k in v for k in ['PASS', 'SEDAN', 'SEDN', '4 DR', '2 DR', 'COUPE', 'CONV', 'CAR', 'AUTO', '4D', '2D', '4S', 'SDN']): return 'Passenger Vehicle'
    return 'Other / Unknown'

def get_wmo_condition_text(code):
    if pd.isna(code): return None
    if code == 0: return 'Clear sky'
    if code in [1, 2, 3]: return 'Partly cloudy / Overcast'
    if code in [45, 48]: return 'Fog'
    if code in [51, 53, 55, 56, 57]: return 'Drizzle'
    if code in [61, 63, 65, 66, 67]: return 'Rain'
    if code in [71, 73, 75, 77]: return 'Snow'
    if code in [80, 81, 82]: return 'Rain showers'
    if code in [85, 86]: return 'Snow showers'
    if code in [95, 96, 99]: return 'Thunderstorm'
    return 'Unknown / Other'

# ==============================================================================

print("1. Lese CSV-Daten ein...")
crashes_df = pd.read_csv(CRASHES_FILE, low_memory=False)
vehicles_df = pd.read_csv(VEHICLES_FILE, low_memory=False)
persons_df = pd.read_csv(PERSONS_FILE, low_memory=False)

crashes_df.columns = crashes_df.columns.str.lower().str.replace(' ', '_')
vehicles_df.columns = vehicles_df.columns.str.lower().str.replace(' ', '_')
persons_df.columns = persons_df.columns.str.lower().str.replace(' ', '_')

print("2. Erstelle Borough und Precinct Dimensionstabellen...")
boroughs_list = ['MANHATTAN', 'BRONX', 'BROOKLYN', 'QUEENS', 'STATEN ISLAND']
borough_df = pd.DataFrame({'Borough_Name': boroughs_list})
borough_df['Borough_ID'] = range(1, len(borough_df) + 1)
borough_df.to_csv(f'{OUTPUT_DIR}Borough.csv', index=False)

def get_borough_id_from_precinct(p_id):
    if 1 <= p_id <= 39: return 1 # Manhattan
    elif 40 <= p_id <= 59: return 2 # Bronx
    elif 60 <= p_id <= 99: return 3 # Brooklyn
    elif 100 <= p_id <= 119: return 4 # Queens
    elif 120 <= p_id <= 139: return 5 # Staten Island
    return None

precincts_gdf = gpd.read_file(PRECINCTS_FILE)
precincts_gdf.columns = precincts_gdf.columns.str.lower()
precinct_col = 'precinct' if 'precinct' in precincts_gdf.columns else 'precinctnumber'

precincts_gdf['Precinct_ID'] = pd.to_numeric(precincts_gdf[precinct_col], errors='coerce').fillna(0).astype(int)
precincts_gdf['Precinct_Name'] = 'Precinct ' + precincts_gdf['Precinct_ID'].astype(str)
precincts_gdf['Borough_ID'] = precincts_gdf['Precinct_ID'].apply(get_borough_id_from_precinct)

precinct_out = precincts_gdf[['Precinct_ID', 'Precinct_Name', 'Borough_ID']].drop_duplicates(subset=['Precinct_ID'])
precinct_out = precinct_out[precinct_out['Precinct_ID'] > 0]
precinct_out['Borough_ID'] = precinct_out['Borough_ID'].astype('Int64') 
precinct_out.to_csv(f'{OUTPUT_DIR}Precinct.csv', index=False)

print("3. Filtere Unfälle & berechne räumliche Nähe zu Precincts und Wetterstationen...")
crashes_df = crashes_df.dropna(subset=['latitude', 'longitude'])
crashes_df['clean_date'] = crashes_df['crash_date'].astype(str).str[:10]
crashes_df['crash_datetime'] = pd.to_datetime(crashes_df['clean_date'] + ' ' + crashes_df['crash_time'].astype(str), errors='coerce')
crashes_df = crashes_df.dropna(subset=['crash_datetime']).sort_values('crash_datetime')

# Unfälle als Geodaten
geometry = [Point(xy) for xy in zip(crashes_df['longitude'], crashes_df['latitude'])]
crashes_gdf = gpd.GeoDataFrame(crashes_df, geometry=geometry, crs="EPSG:4326")

if precincts_gdf.crs is None: precincts_gdf.set_crs(epsg=4326, inplace=True)
else: precincts_gdf = precincts_gdf.to_crs(epsg=4326)

# Welcher Precinct?
crashes_mapped = gpd.sjoin(crashes_gdf, precincts_gdf[['Precinct_ID', 'Borough_ID', 'geometry']], how="left", predicate="within")
crashes_mapped = crashes_mapped.dropna(subset=['Borough_ID']) 
crashes_mapped['Borough_ID'] = crashes_mapped['Borough_ID'].astype(int)

# ---> NEU: Lösche die Hilfsspalte vom ersten Join <---
if 'index_right' in crashes_mapped.columns:
    crashes_mapped = crashes_mapped.drop(columns=['index_right'])

# --- NEU: Finde die absolut nächste Wetterstation für jeden Unfall ---
stations_df = pd.DataFrame(STATIONS)

# --- NEU: Finde die absolut nächste Wetterstation für jeden Unfall ---
# Stationen als Geodaten anlegen
stations_df = pd.DataFrame(STATIONS)
stations_geom = [Point(xy) for xy in zip(stations_df['lon'], stations_df['lat'])]
stations_gdf = gpd.GeoDataFrame(stations_df, geometry=stations_geom, crs="EPSG:4326")

# Um genaue Distanzen in Metern zu berechnen, projizieren wir temporär auf das lokale NYC Koordinatensystem (EPSG:2263)
crashes_proj = crashes_mapped.to_crs(epsg=2263)
stations_proj = stations_gdf.to_crs(epsg=2263)

# Räumlicher Join: Finde für jeden Unfallpunkt die exakt nächste Wetterstation
crashes_with_station = gpd.sjoin_nearest(crashes_proj, stations_proj[['Weather_Station', 'geometry']], how='left')
# Übertrage die zugeordnete Station auf unsere Haupttabelle
crashes_mapped['Weather_Station'] = crashes_with_station['Weather_Station'].values


print("4. Lade Wetterdaten von Open-Meteo für die 3 Stationen herunter...")

# Hole echtes Datumsobjekt (verhindert MM/DD/YYYY Fehler)
start_dt = crashes_mapped['crash_datetime'].min()
end_dt = crashes_mapped['crash_datetime'].max()

# Open-Meteo Archive hat Daten nur bis ca. 3-5 Tage in der Vergangenheit
max_archive_date = datetime.now() - timedelta(days=5)

if end_dt > max_archive_date:
    print(f"   -> Info: Setze Enddatum von {end_dt.date()} auf {max_archive_date.date()} zurück (API-Limit)")
    end_dt = max_archive_date

# Formatiere strikt als YYYY-MM-DD für die API
start_date = start_dt.strftime('%Y-%m-%d')
end_date = end_dt.strftime('%Y-%m-%d')

print(f"   -> Abfrage-Zeitraum: {start_date} bis {end_date}")

all_weather_data = []

for station in STATIONS:
    station_id = station['Weather_Station']
    print(f"   -> Lade Wetter für Station {station_id} ({station['name']})...")
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={station['lat']}&longitude={station['lon']}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,precipitation,snow_depth,weather_code&timezone=America%2FNew_York"    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        hourly = data['hourly']
        
        temp_df = pd.DataFrame({
            'weather_datetime': pd.to_datetime(hourly['time']),
            'Temp_Celsius': hourly['temperature_2m'],
            'Precipitation_mm': hourly['precipitation'],
            'Snow_Depth_m': hourly['snow_depth'],
            'Visibility_m': None, # Gibt es im historischen Archiv nicht
            'Wind_Gust_kmh': None, # Gibt es im historischen Archiv nicht
            'Weather_Code': hourly['weather_code'],
            'Weather_Station': station_id  
        })
        all_weather_data.append(temp_df)
    else:
        # NEU: Falls es wieder crasht, drucken wir den genauen Grund der API aus!
        print(f"Fehler bei Open-Meteo API für {station_id}: {response.status_code}")
        print(f"Details vom Server: {response.text}")

if not all_weather_data:
    raise ValueError("Es konnten absolut keine Wetterdaten geladen werden. Skript abgebrochen!")

weather_raw_df = pd.concat(all_weather_data, ignore_index=True)
weather_raw_df = weather_raw_df.sort_values('weather_datetime')

weather_df = pd.DataFrame()
weather_df['Weather_ID'] = range(1, len(weather_raw_df) + 1)
weather_df['Weather_Station'] = weather_raw_df['Weather_Station'] 
weather_df['Measure_Date'] = weather_raw_df['weather_datetime'].dt.date
weather_df['Measure_Time'] = weather_raw_df['weather_datetime'].dt.time
weather_df['Temp_Celsius'] = pd.to_numeric(weather_raw_df['Temp_Celsius'], errors='coerce').round(2)
weather_df['Precipitation_Inches'] = (pd.to_numeric(weather_raw_df['Precipitation_mm'], errors='coerce') * 0.0393701).round(2)
weather_df['Snow_Depth_Inches'] = (pd.to_numeric(weather_raw_df['Snow_Depth_m'], errors='coerce') * 39.3701).fillna(0).round(2)

# Da es in den historischen Daten keine Sichtweite und Windböen gibt, setzen wir sie für die Datenbank direkt auf "leer" (None)
weather_df['Visibility_Miles'] = None
weather_df['Wind_Gust_Speed_MPH'] = None

weather_df.to_csv(f'{OUTPUT_DIR}Weather.csv', index=False)
weather_raw_df['Weather_ID'] = weather_df['Weather_ID'].values


print("5. Verbinde Unfälle mit dem Wetter der nächstgelegenen Station...")
crashes_mapped = crashes_mapped.sort_values('crash_datetime')
weather_raw_df = weather_raw_df.sort_values('weather_datetime')

crashes_final = pd.merge_asof(
    crashes_mapped, 
    weather_raw_df[['weather_datetime', 'Weather_ID', 'Weather_Station']], 
    left_on='crash_datetime', 
    right_on='weather_datetime', 
    by='Weather_Station', # Vergleicht nur Stationen, die im Schritt 3 gematcht wurden
    direction='nearest',
    tolerance=pd.Timedelta('2 hours')
)

print("6. Erstelle restliche Dimensionstabellen...")
fixed_types = [
    'Passenger Vehicle', 'SUV / Station Wagon', 'Taxi / Livery', 'Bus / School Bus', 
    'Truck / Commercial / Delivery', 'Emergency / Medical / Police', 
    'Motorcycle / Moped / Scooter', 'Bicycle / E-Bike', 'Van / Minivan', 'Other / Unknown'
]
vehicle_type_df = pd.DataFrame({'Vehicle_Type_Name': fixed_types})
vehicle_type_df['Vehicle_Type_ID'] = range(1, len(vehicle_type_df) + 1)
vehicle_type_df['Vehicle_Type_Category'] = 'Standardized'
vehicle_type_df[['Vehicle_Type_ID', 'Vehicle_Type_Name', 'Vehicle_Type_Category']].to_csv(f'{OUTPUT_DIR}Vehicle_Type.csv', index=False)

all_factors = pd.concat([vehicles_df['contributing_factor_1'].dropna(), vehicles_df['contributing_factor_2'].dropna()]).unique()
factor_df = pd.DataFrame({'Factor_Name': all_factors})
factor_df['Factor_ID'] = range(1, len(factor_df) + 1)
factor_df['Factor_Category'] = None
factor_df[['Factor_ID', 'Factor_Name', 'Factor_Category']].to_csv(f'{OUTPUT_DIR}Contributing_Factor.csv', index=False)

locations = crashes_final[['latitude', 'longitude', 'zip_code', 'Precinct_ID']].drop_duplicates().reset_index(drop=True)
locations['Location_ID'] = range(1, len(locations) + 1)
locations['Precinct_ID'] = locations['Precinct_ID'].astype('Int64')
locations[['Location_ID', 'longitude', 'latitude', 'zip_code', 'Precinct_ID']].to_csv(f'{OUTPUT_DIR}Location.csv', index=False)

print("7. Erstelle Faktentabellen...")
crash_merge = pd.merge(crashes_final, locations, on=['latitude', 'longitude', 'zip_code', 'Precinct_ID'], how='left')
crash_out = crash_merge[['collision_id', 'clean_date', 'crash_time', 'Location_ID', 'Weather_ID']].copy()
crash_out.rename(columns={'collision_id': 'Collision_ID', 'clean_date': 'Crash_Date', 'crash_time': 'Crash_Time'}, inplace=True)
crash_out['Weather_ID'] = crash_out['Weather_ID'].astype('Int64') 
crash_out[['Collision_ID', 'Crash_Date', 'Crash_Time', 'Location_ID', 'Weather_ID']].to_csv(f'{OUTPUT_DIR}Crash.csv', index=False)

valid_collision_ids = crash_out['Collision_ID'].unique()
vehicles_filtered = vehicles_df[vehicles_df['collision_id'].isin(valid_collision_ids)].copy()
persons_filtered = persons_df[persons_df['collision_id'].isin(valid_collision_ids)].copy()

vehicles_filtered['clean_type'] = vehicles_filtered['vehicle_type'].apply(get_fixed_vehicle_category)
vehicle_merge = pd.merge(vehicles_filtered, vehicle_type_df, left_on='clean_type', right_on='Vehicle_Type_Name', how='left')
vehicle_out = vehicle_merge[['vehicle_id', 'collision_id', 'state_registration', 'vehicle_year', 'Vehicle_Type_ID']].copy()
vehicle_out.rename(columns={'vehicle_id': 'Vehicle_ID', 'collision_id': 'Collision_ID', 'state_registration': 'State_Registration', 'vehicle_year': 'Vehicle_Year'}, inplace=True)
vehicle_out.dropna(subset=['Vehicle_ID'], inplace=True)
vehicle_out[['Vehicle_ID', 'Collision_ID', 'State_Registration', 'Vehicle_Year', 'Vehicle_Type_ID']].to_csv(f'{OUTPUT_DIR}Vehicle.csv', index=False)

factor_map = factor_df.set_index('Factor_Name')['Factor_ID'].to_dict()
vf_list = []
for index, row in vehicles_filtered.iterrows():
    vid = row['vehicle_id']
    if pd.isna(vid): continue
    f1, f2 = row['contributing_factor_1'], row['contributing_factor_2']
    if pd.notna(f1) and f1 in factor_map: vf_list.append({'Vehicle_ID': vid, 'Factor_ID': factor_map[f1]})
    if pd.notna(f2) and f2 in factor_map and f1 != f2: vf_list.append({'Vehicle_ID': vid, 'Factor_ID': factor_map[f2]})

pd.DataFrame(vf_list).drop_duplicates().to_csv(f'{OUTPUT_DIR}Vehicle_Factors.csv', index=False)

person_out = persons_filtered[['person_id', 'collision_id', 'vehicle_id', 'person_type', 'ped_role', 'person_injury', 'person_age', 'person_sex']].copy()
person_out.rename(columns={'person_id': 'Person_ID', 'collision_id': 'Collision_ID', 'vehicle_id': 'Vehicle_ID', 'person_type': 'Person_Type', 'ped_role': 'Person_Role', 'person_injury': 'Person_Injury', 'person_age': 'Person_Age', 'person_sex': 'Person_Sex'}, inplace=True)
person_out.dropna(subset=['Person_ID'], inplace=True)
person_out[['Person_ID', 'Collision_ID', 'Vehicle_ID', 'Person_Type', 'Person_Role', 'Person_Injury', 'Person_Age', 'Person_Sex']].to_csv(f'{OUTPUT_DIR}Person.csv', index=False)

print(f"Fertig! Die hochpräzisen Tabellen liegen bereit im Ordner '{OUTPUT_DIR}'.")