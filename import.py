import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import re
import warnings
warnings.filterwarnings('ignore')

# --- Konfiguration ---
CRASHES_FILE = 'data/crashes.csv'
VEHICLES_FILE = 'data/vehicles.csv'
PERSONS_FILE = 'data/persons.csv'
WEATHER_FILE = 'data/jfk_weather_cleaned.csv' 
PRECINCTS_FILE = 'data/precincts.geojson'  
OUTPUT_DIR = 'output_tables/'
PREFIX = 'st_' 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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

def clean_kaggle_numeric(val):
    if pd.isna(val): return 0.0
    val_str = str(val).strip().upper()
    if val_str == 'T': return 0.0 
    cleaned = re.sub(r'[^\d.]', '', val_str)
    try: return float(cleaned) if cleaned else 0.0
    except: return 0.0

# ==============================================================================

print("1. Lese CSV-Daten ein...")
crashes_df = pd.read_csv(CRASHES_FILE, low_memory=False)
vehicles_df = pd.read_csv(VEHICLES_FILE, low_memory=False)
persons_df = pd.read_csv(PERSONS_FILE, low_memory=False)
weather_raw_df = pd.read_csv(WEATHER_FILE, low_memory=False)

crashes_df.columns = crashes_df.columns.str.lower().str.replace(' ', '_')
vehicles_df.columns = vehicles_df.columns.str.lower().str.replace(' ', '_')
persons_df.columns = persons_df.columns.str.lower().str.replace(' ', '_')

print("2. Erstelle Borough und Precinct Dimensionstabellen...")
boroughs_list = ['MANHATTAN', 'BRONX', 'BROOKLYN', 'QUEENS', 'STATEN ISLAND']
borough_df = pd.DataFrame({'Borough_Name': boroughs_list})
borough_df['Borough_ID'] = range(1, len(borough_df) + 1)
borough_df.to_csv(f'{OUTPUT_DIR}{PREFIX}Borough.csv', index=False)

def get_borough_id_from_precinct(p_id):
    if 1 <= p_id <= 39: return 1 
    elif 40 <= p_id <= 59: return 2 
    elif 60 <= p_id <= 99: return 3 
    elif 100 <= p_id <= 119: return 4 
    elif 120 <= p_id <= 139: return 5 
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
precinct_out.to_csv(f'{OUTPUT_DIR}{PREFIX}Precinct.csv', index=False)


print("3. Filtere Unfälle & berechne räumliche Nähe zu Precincts...")
crashes_df = crashes_df.dropna(subset=['latitude', 'longitude'])
crashes_df['clean_date'] = crashes_df['crash_date'].astype(str).str[:10]
crashes_df['crash_datetime'] = pd.to_datetime(crashes_df['clean_date'] + ' ' + crashes_df['crash_time'].astype(str), errors='coerce')
crashes_df = crashes_df.dropna(subset=['crash_datetime'])

# ---> NEU: Filter auf das Jahr 2017 <---
crashes_df = crashes_df[crashes_df['crash_datetime'].dt.year == 2017]

crashes_df = crashes_df.sort_values('crash_datetime')

geometry = [Point(xy) for xy in zip(crashes_df['longitude'], crashes_df['latitude'])]
crashes_gdf = gpd.GeoDataFrame(crashes_df, geometry=geometry, crs="EPSG:4326")

if precincts_gdf.crs is None: precincts_gdf.set_crs(epsg=4326, inplace=True)
else: precincts_gdf = precincts_gdf.to_crs(epsg=4326)

crashes_mapped = gpd.sjoin(crashes_gdf, precincts_gdf[['Precinct_ID', 'Borough_ID', 'geometry']], how="left", predicate="within")
crashes_mapped = crashes_mapped.dropna(subset=['Borough_ID']) 
crashes_mapped['Borough_ID'] = crashes_mapped['Borough_ID'].astype(int)
if 'index_right' in crashes_mapped.columns: crashes_mapped = crashes_mapped.drop(columns=['index_right'])


print("4. Verarbeite die JFK Kaggle Wetterdaten...")
weather_raw_df['weather_datetime'] = pd.to_datetime(weather_raw_df['DATE'], errors='coerce')
weather_raw_df = weather_raw_df.dropna(subset=['weather_datetime']).sort_values('weather_datetime')

weather_df = pd.DataFrame()
weather_df['Weather_ID'] = range(1, len(weather_raw_df) + 1)
weather_df['Weather_Station'] = 'KJFK' 
weather_df['Measure_Date'] = weather_raw_df['weather_datetime'].dt.date
weather_df['Measure_Time'] = weather_raw_df['weather_datetime'].dt.time

temp_f = weather_raw_df['HOURLYDRYBULBTEMPF'].apply(clean_kaggle_numeric)
weather_df['Temp_Celsius'] = ((temp_f - 32) * 5.0/9.0).round(2)
weather_df['Precipitation_Inches'] = weather_raw_df['HOURLYPrecip'].apply(clean_kaggle_numeric).round(2)
weather_df['Visibility_Miles'] = weather_raw_df['HOURLYVISIBILITY'].apply(clean_kaggle_numeric).round(2)
weather_df['Wind_Gust_Speed_MPH'] = weather_raw_df['HOURLYWindSpeed'].apply(clean_kaggle_numeric).round(2)
weather_df['Snow_Depth_Inches'] = 0.0
weather_df['Weather_Condition_Text'] = None

weather_df.to_csv(f'{OUTPUT_DIR}{PREFIX}Weather.csv', index=False)
weather_raw_df['Weather_ID'] = weather_df['Weather_ID'].values


print("5. Verbinde Unfälle zeitlich mit dem JFK-Wetter...")
crashes_mapped = crashes_mapped.sort_values('crash_datetime')
weather_raw_df = weather_raw_df.sort_values('weather_datetime')

crashes_final = pd.merge_asof(
    crashes_mapped, 
    weather_raw_df[['weather_datetime', 'Weather_ID']], 
    left_on='crash_datetime', 
    right_on='weather_datetime', 
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
# ---> NEU: Vehicle_Type_Category komplett entfernt <---
vehicle_type_df[['Vehicle_Type_ID', 'Vehicle_Type_Name']].to_csv(f'{OUTPUT_DIR}{PREFIX}Vehicle_Type.csv', index=False)

all_factors = pd.concat([vehicles_df['contributing_factor_1'].dropna(), vehicles_df['contributing_factor_2'].dropna()]).unique()
factor_df = pd.DataFrame({'Factor_Name': all_factors})
factor_df['Factor_ID'] = range(1, len(factor_df) + 1)
factor_df['Factor_Category'] = None
factor_df[['Factor_ID', 'Factor_Name', 'Factor_Category']].to_csv(f'{OUTPUT_DIR}{PREFIX}Contributing_Factor.csv', index=False)

locations = crashes_final[['latitude', 'longitude', 'zip_code', 'Precinct_ID']].drop_duplicates().reset_index(drop=True)
locations['Location_ID'] = range(1, len(locations) + 1)
locations['Precinct_ID'] = locations['Precinct_ID'].astype('Int64')
locations[['Location_ID', 'longitude', 'latitude', 'zip_code', 'Precinct_ID']].to_csv(f'{OUTPUT_DIR}{PREFIX}Location.csv', index=False)


print("7. Erstelle Faktentabellen...")
crash_merge = pd.merge(crashes_final, locations, on=['latitude', 'longitude', 'zip_code', 'Precinct_ID'], how='left')
crash_out = crash_merge[['collision_id', 'clean_date', 'crash_time', 'Location_ID', 'Weather_ID']].copy()
crash_out.rename(columns={'collision_id': 'Collision_ID', 'clean_date': 'Crash_Date', 'crash_time': 'Crash_Time'}, inplace=True)
crash_out['Weather_ID'] = crash_out['Weather_ID'].astype('Int64') 
crash_out[['Collision_ID', 'Crash_Date', 'Crash_Time', 'Location_ID', 'Weather_ID']].to_csv(f'{OUTPUT_DIR}{PREFIX}Crash.csv', index=False)

valid_collision_ids = crash_out['Collision_ID'].unique()
vehicles_filtered = vehicles_df[vehicles_df['collision_id'].isin(valid_collision_ids)].copy()
persons_filtered = persons_df[persons_df['collision_id'].isin(valid_collision_ids)].copy()

vehicles_filtered['clean_type'] = vehicles_filtered['vehicle_type'].apply(get_fixed_vehicle_category)
vehicle_merge = pd.merge(vehicles_filtered, vehicle_type_df, left_on='clean_type', right_on='Vehicle_Type_Name', how='left')

# --- FAHRZEUGE (Nutzt jetzt unique_id als Vehicle_ID) ---
vehicle_out = vehicle_merge[['unique_id', 'collision_id', 'state_registration', 'vehicle_year', 'Vehicle_Type_ID']].copy()
vehicle_out.rename(columns={
    'unique_id': 'Vehicle_ID', 
    'collision_id': 'Collision_ID', 
    'state_registration': 'State_Registration', 
    'vehicle_year': 'Vehicle_Year'
}, inplace=True)
vehicle_out.dropna(subset=['Vehicle_ID'], inplace=True)
vehicle_out[['Vehicle_ID', 'Collision_ID', 'State_Registration', 'Vehicle_Year', 'Vehicle_Type_ID']].to_csv(f'{OUTPUT_DIR}{PREFIX}Vehicle.csv', index=False)

# --- FAHRZEUG-FAKTOREN ---
factor_map = factor_df.set_index('Factor_Name')['Factor_ID'].to_dict()
vf_list = []
for index, row in vehicles_filtered.iterrows():
    vid = row['unique_id'] 
    if pd.isna(vid): continue
    f1, f2 = row['contributing_factor_1'], row['contributing_factor_2']
    if pd.notna(f1) and f1 in factor_map: vf_list.append({'Vehicle_ID': vid, 'Factor_ID': factor_map[f1]})
    if pd.notna(f2) and f2 in factor_map and f1 != f2: vf_list.append({'Vehicle_ID': vid, 'Factor_ID': factor_map[f2]})

pd.DataFrame(vf_list).drop_duplicates().to_csv(f'{OUTPUT_DIR}{PREFIX}Vehicle_Factors.csv', index=False)

# --- PERSONEN (Nutzt unique_id als Person_ID und vehicle_id bleibt Vehicle_ID für den Join) ---
person_out = persons_filtered[['unique_id', 'collision_id', 'vehicle_id', 'person_type', 'ped_role', 'person_injury', 'person_age', 'person_sex']].copy()
person_out.rename(columns={
    'unique_id': 'Person_ID', 
    'collision_id': 'Collision_ID', 
    'vehicle_id': 'Vehicle_ID', 
    'person_type': 'Person_Type', 
    'ped_role': 'Person_Role', 
    'person_injury': 'Person_Injury', 
    'person_age': 'Person_Age', 
    'person_sex': 'Person_Sex'
}, inplace=True)
person_out.dropna(subset=['Person_ID'], inplace=True)
person_out[['Person_ID', 'Collision_ID', 'Vehicle_ID', 'Person_Type', 'Person_Role', 'Person_Injury', 'Person_Age', 'Person_Sex']].to_csv(f'{OUTPUT_DIR}{PREFIX}Person.csv', index=False)

print(f"Bäm! Fertig! Alle sauberen Tabellen (für 2017) liegen in '{OUTPUT_DIR}'.")