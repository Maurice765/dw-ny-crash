import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import re
import warnings
import pint
import numpy as np

warnings.filterwarnings('ignore')

# --- Konfiguration ---
CRASHES_FILE = 'data/crashes.csv'
VEHICLES_FILE = 'data/vehicles.csv'
PERSONS_FILE = 'data/persons.csv'
WEATHER_FILE = 'data/jfk_weather.csv' 
PRECINCTS_FILE = 'data/precincts.geojson'  
OUTPUT_DIR = Path('output_tables')
PREFIX = 'st_' 
FILTER_YEAR = 2017

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ureg = pint.UnitRegistry()

# --- Hilfsfunktionen ---
def clean_kaggle_numeric(val):
    if pd.isna(val): return 0.0
    val_str = str(val).strip().upper()
    if val_str == 'T': return 0.0 
    cleaned = re.sub(r'[^\d.]', '', val_str)
    try: return float(cleaned) if cleaned else 0.0
    except: return 0.0

# ==============================================================================

print("1. Lese CSV-Daten ein...")
crashes_df = pd.read_csv(CRASHES_FILE, engine='pyarrow')
vehicles_df = pd.read_csv(VEHICLES_FILE, engine='pyarrow')
persons_df = pd.read_csv(PERSONS_FILE, engine='pyarrow')
weather_raw_df = pd.read_csv(WEATHER_FILE, engine='pyarrow')

crashes_df.columns = crashes_df.columns.str.lower().str.replace(' ', '_')
vehicles_df.columns = vehicles_df.columns.str.lower().str.replace(' ', '_')
persons_df.columns = persons_df.columns.str.lower().str.replace(' ', '_')

print("2. Erstelle Borough und Precinct Dimensionstabellen...")
boroughs_list = ['MANHATTAN', 'BRONX', 'BROOKLYN', 'QUEENS', 'STATEN ISLAND']
borough_df = pd.DataFrame({'Borough_Name': boroughs_list})
borough_df['Borough_ID'] = pd.Series(range(1, len(borough_df) + 1), dtype='Int64')
borough_df.to_csv(OUTPUT_DIR / f'{PREFIX}Borough.csv', index=False)

def get_borough_id_from_precinct(p_id):
    if pd.isna(p_id) or p_id == 0: return None
    if 1 <= p_id <= 39: return 1 
    elif 40 <= p_id <= 59: return 2 
    elif 60 <= p_id <= 99: return 3 
    elif 100 <= p_id <= 119: return 4 
    elif 120 <= p_id <= 139: return 5 
    return None

precincts_gdf = gpd.read_file(PRECINCTS_FILE)
precincts_gdf.columns = precincts_gdf.columns.str.lower()
precinct_col = 'precinct' if 'precinct' in precincts_gdf.columns else 'precinctnumber'

precincts_gdf['Precinct_ID'] = pd.to_numeric(precincts_gdf[precinct_col], errors='coerce').fillna(0).astype('Int64')
precincts_gdf['Precinct_Name'] = 'Precinct ' + precincts_gdf['Precinct_ID'].astype(str)
precincts_gdf['Borough_ID'] = precincts_gdf['Precinct_ID'].apply(get_borough_id_from_precinct).astype('Int64')

precinct_out = precincts_gdf[['Precinct_ID', 'Precinct_Name', 'Borough_ID']].drop_duplicates(subset=['Precinct_ID'])
precinct_out = precinct_out[precinct_out['Precinct_ID'] > 0]
precinct_out.to_csv(OUTPUT_DIR / f'{PREFIX}Precinct.csv', index=False)


print("3. Filtere Unfälle & berechne räumliche Nähe zu Precincts...")
crashes_df = crashes_df.dropna(subset=['latitude', 'longitude'])
crashes_df['clean_date'] = crashes_df['crash_date'].astype(str).str[:10]
crashes_df['crash_datetime'] = pd.to_datetime(crashes_df['clean_date'] + ' ' + crashes_df['crash_time'].astype(str), errors='coerce')
crashes_df = crashes_df.dropna(subset=['crash_datetime'])

crashes_df = crashes_df[crashes_df['crash_datetime'].dt.year == FILTER_YEAR]
crashes_df = crashes_df.sort_values('crash_datetime')

geometry = [Point(xy) for xy in zip(crashes_df['longitude'], crashes_df['latitude'])]
crashes_gdf = gpd.GeoDataFrame(crashes_df, geometry=geometry, crs="EPSG:4326")

if precincts_gdf.crs is None: precincts_gdf.set_crs(epsg=4326, inplace=True)
else: precincts_gdf = precincts_gdf.to_crs(epsg=4326)

crashes_mapped = gpd.sjoin(crashes_gdf, precincts_gdf[['Precinct_ID', 'Borough_ID', 'geometry']], how="left", predicate="within")
crashes_mapped = crashes_mapped.dropna(subset=['Borough_ID']) 
crashes_mapped['Borough_ID'] = crashes_mapped['Borough_ID'].astype('Int64')
crashes_mapped['Precinct_ID'] = crashes_mapped['Precinct_ID'].astype('Int64')
if 'index_right' in crashes_mapped.columns: crashes_mapped = crashes_mapped.drop(columns=['index_right'])


print("4. Verarbeite die JFK Kaggle Wetterdaten...")
weather_raw_df['weather_datetime'] = pd.to_datetime(weather_raw_df['DATE'], errors='coerce')
weather_raw_df = weather_raw_df.dropna(subset=['weather_datetime']).sort_values('weather_datetime')

weather_df = pd.DataFrame()
weather_df['Weather_ID'] = pd.Series(range(1, len(weather_raw_df) + 1), dtype='Int64')

# Station direkt aus den Daten (auf 10 Zeichen begrenzt für VARCHAR(10))
weather_df['Weather_Station'] = weather_raw_df['STATION'].astype(str).str[:10] 

weather_df['Measure_Date'] = weather_raw_df['weather_datetime'].dt.date
weather_df['Measure_Time'] = weather_raw_df['weather_datetime'].dt.time

# Temperatur: Direkt die Celsius-Spalte nutzen!
weather_df['Temp_Celsius'] = weather_raw_df['HOURLYDRYBULBTEMPC'].apply(clean_kaggle_numeric).round(2)

# Sicht und Niederschlag
weather_df['Visibility_Miles'] = weather_raw_df['HOURLYVISIBILITY'].apply(clean_kaggle_numeric).round(2)
weather_df['Precipitation_Inches'] = weather_raw_df['HOURLYPrecip'].apply(clean_kaggle_numeric).round(2)

# Windböen: Wir nutzen Gust (Böen). Wenn leer, Fallback auf normale Windgeschwindigkeit
gust = weather_raw_df['HOURLYWindGustSpeed'].apply(clean_kaggle_numeric)
speed = weather_raw_df['HOURLYWindSpeed'].apply(clean_kaggle_numeric)
weather_df['Wind_Gust_Speed_MPH'] = np.where(gust > 0, gust, speed).round(2)

# Schneehöhe: Ist ein täglicher Wert in den Rohdaten. Wir füllen ihn "vorwärts" (ffill)
# für die Stunden des Tages auf, bevor wir bereinigen.
snow_depth_raw = weather_raw_df['DAILYSnowDepth'].replace(r'^\s*$', np.nan, regex=True).ffill()
weather_df['Snow_Depth_Inches'] = snow_depth_raw.apply(clean_kaggle_numeric).round(2)

# Wetter-Bedingungen (Text): Direkt aus der Datei
# Priorität: 1. PRESENTWEATHERTYPE, 2. SKYCONDITIONS, 3. 'Clear' (als Default)
cond_text = weather_raw_df['HOURLYPRSENTWEATHERTYPE'].fillna('').astype(str).str.strip()
sky_text = weather_raw_df['HOURLYSKYCONDITIONS'].fillna('').astype(str).str.strip()

weather_df['Weather_Condition_Text'] = np.where(
    cond_text != '', cond_text, 
    np.where(sky_text != '', sky_text, 'Clear')
)

weather_df['Weather_Condition_Text'] = weather_df['Weather_Condition_Text'].str[:255]

weather_df.to_csv(OUTPUT_DIR / f'{PREFIX}Weather.csv', index=False)
weather_raw_df['Weather_ID'] = weather_df['Weather_ID'].values

print("5. Verbinde Unfälle zeitlich mit dem JFK-Wetter...")
crashes_mapped = crashes_mapped.sort_values('crash_datetime')
weather_raw_df = weather_raw_df.sort_values('weather_datetime')

crashes_mapped['crash_datetime'] = crashes_mapped['crash_datetime'].astype('datetime64[us]')
weather_raw_df['weather_datetime'] = weather_raw_df['weather_datetime'].astype('datetime64[us]')

crashes_final = pd.merge_asof(
    crashes_mapped, 
    weather_raw_df[['weather_datetime', 'Weather_ID']], 
    left_on='crash_datetime', 
    right_on='weather_datetime', 
    direction='nearest',
    tolerance=pd.Timedelta('2 hours')
)

print("6. Erstelle restliche Dimensionstabellen...")
# --- Vehicle Types ---
vehicle_types_data = [
    {'Type': 'Passenger', 'Category': 'Auto'},
    {'Type': 'SUV', 'Category': 'Auto'},
    {'Type': 'Wagon', 'Category': 'Auto'},
    
    {'Type': 'Taxi', 'Category': 'Hire'},
    {'Type': 'Livery', 'Category': 'Hire'},
    
    {'Type': 'Bus', 'Category': 'Transit'},
    {'Type': 'Schoolbus', 'Category': 'Transit'},
    
    {'Type': 'Truck', 'Category': 'Commercial'},
    {'Type': 'Commercial', 'Category': 'Commercial'},
    {'Type': 'Delivery', 'Category': 'Commercial'},
    {'Type': 'Sanitation', 'Category': 'Commercial'},
    
    {'Type': 'Emergency', 'Category': 'Emergency'},
    {'Type': 'Ambulance', 'Category': 'Emergency'},
    {'Type': 'Police', 'Category': 'Emergency'},
    {'Type': 'Firetruck', 'Category': 'Emergency'},
    
    {'Type': 'Motorcycle', 'Category': 'Motorbike'},
    {'Type': 'Moped', 'Category': 'Motorbike'},
    {'Type': 'ATV', 'Category': 'Motorbike'},
    
    {'Type': 'Bicycle', 'Category': 'Micromobility'},
    {'Type': 'Ebike', 'Category': 'Micromobility'},
    {'Type': 'Scooter', 'Category': 'Micromobility'},
    {'Type': 'Escooter', 'Category': 'Micromobility'},
    {'Type': 'Pedicab', 'Category': 'Micromobility'},
    
    {'Type': 'Van', 'Category': 'Van'},
    {'Type': 'Minivan', 'Category': 'Van'},
    
    {'Type': 'Other', 'Category': 'Unknown'},
    {'Type': 'Unknown', 'Category': 'Unknown'}
]

vehicle_type_df = pd.DataFrame(vehicle_types_data)
vehicle_type_df.rename(columns={'Type': 'Vehicle_Type_Name', 'Category': 'Vehicle_Category'}, inplace=True)
vehicle_type_df['Vehicle_Type_ID'] = pd.Series(range(1, len(vehicle_type_df) + 1), dtype='Int64')
vehicle_type_df[['Vehicle_Type_ID', 'Vehicle_Type_Name', 'Vehicle_Category']].to_csv(OUTPUT_DIR / f'{PREFIX}Vehicle_Type.csv', index=False)

# --- Contributing Factors (Bereinigt & Kategorisiert) ---
factor_mapping = {
    'driver inattention/distraction': ['Driver Inattention/Distraction', 'Driver Distraction'],
    'passenger distraction': ['Passenger Distraction', 'Driver Distraction'],
    'outside car distraction': ['Outside Car Distraction', 'Driver Distraction'],
    'other electronic device': ['Other Electronic Device', 'Driver Distraction'],
    'cell phone (hand-held)': ['Cell Phone (hand-held)', 'Driver Distraction'],
    'cell phone (hands-free)': ['Cell Phone (hands-free)', 'Driver Distraction'],
    'using on board navigation device': ['Using On Board Navigation Device', 'Driver Distraction'],
    'eating or drinking': ['Eating or Drinking', 'Driver Distraction'],
    'texting': ['Texting', 'Driver Distraction'],
    'listening/using headphones': ['Listening/Using Headphones', 'Driver Distraction'],
    'lost consciousness': ['Lost Consciousness', 'Driver Condition'],
    'physical disability': ['Physical Disability', 'Driver Condition'],
    'fatigued/drowsy': ['Fatigued/Drowsy', 'Driver Condition'],
    'prescription medication': ['Prescription Medication', 'Driver Condition'],
    'illness': ['Illness', 'Driver Condition'],
    'illnes': ['Illness', 'Driver Condition'], 
    'alcohol involvement': ['Alcohol Involvement', 'Driver Condition'],
    'fell asleep': ['Fell Asleep', 'Driver Condition'],
    'drugs (illegal)': ['Drugs (Illegal)', 'Driver Condition'], 
    'traffic control disregarded': ['Traffic Control Disregarded', 'Traffic Violation'],
    'unsafe lane changing': ['Unsafe Lane Changing', 'Traffic Violation'],
    'backing unsafely': ['Backing Unsafely', 'Traffic Violation'],
    'failure to yield right-of-way': ['Failure to Yield Right-of-Way', 'Traffic Violation'],
    'following too closely': ['Following Too Closely', 'Traffic Violation'],
    'turning improperly': ['Turning Improperly', 'Traffic Violation'],
    'passing too closely': ['Passing Too Closely', 'Traffic Violation'],
    'passing or lane usage improper': ['Passing or Lane Usage Improper', 'Traffic Violation'],
    'unsafe speed': ['Unsafe Speed', 'Traffic Violation'],
    'aggressive driving/road rage': ['Aggressive Driving/Road Rage', 'Traffic Violation'],
    'failure to keep right': ['Failure to Keep Right', 'Traffic Violation'],
    'driver inexperience': ['Driver Inexperience', 'Traffic Violation'],
    'other vehicular': ['Other Vehicular Defect', 'Vehicle Defect'],
    'brakes defective': ['Brakes Defective', 'Vehicle Defect'],
    'oversized vehicle': ['Oversized Vehicle', 'Vehicle Defect'],
    'steering failure': ['Steering Failure', 'Vehicle Defect'],
    'tire failure/inadequate': ['Tire Failure/Inadequate', 'Vehicle Defect'],
    'accelerator defective': ['Accelerator Defective', 'Vehicle Defect'],
    'driverless/runaway vehicle': ['Driverless/Runaway Vehicle', 'Vehicle Defect'],
    'other lighting defects': ['Other Lighting Defects', 'Vehicle Defect'],
    'headlights defective': ['Headlights Defective', 'Vehicle Defect'],
    'tow hitch defective': ['Tow Hitch Defective', 'Vehicle Defect'],
    'windshield inadequate': ['Windshield Inadequate', 'Vehicle Defect'],
    'tinted windows': ['Tinted Windows', 'Vehicle Defect'],
    'pavement slippery': ['Pavement Slippery', 'Environmental'],
    'glare': ['Glare', 'Environmental'],
    'pavement defective': ['Pavement Defective', 'Environmental'],
    'obstruction/debris': ['Obstruction/Debris', 'Environmental'],
    'lane marking improper/inadequate': ['Lane Marking Improper/Inadequate', 'Environmental'],
    'traffic control device improper/non-working': ['Traffic Control Device Improper/Non-Working', 'Environmental'],
    'shoulders defective/improper': ['Shoulders Defective/Improper', 'Environmental'],
    'animals action': ['Animals Action', 'Environmental'],
    'reaction to uninvolved vehicle': ['Reaction to Uninvolved Vehicle', 'Other'],
    'view obstructed/limited': ['View Obstructed/Limited', 'Other'],
    'reaction to other uninvolved vehicle': ['Reaction to Other Uninvolved Vehicle', 'Other'],
    'pedestrian/bicyclist/other pedestrian error/confusion': ['Pedestrian/Bicyclist Error', 'Other'],
    'vehicle vandalism': ['Vehicle Vandalism', 'Other'],
    'unspecified': ['Unspecified', 'Unknown']
}

raw_factors = pd.concat([vehicles_df['contributing_factor_1'], vehicles_df['contributing_factor_2']]).dropna()
raw_factors = raw_factors.str.strip().str.lower()

mapped_data = []
for raw_val in raw_factors.unique():
    if raw_val in factor_mapping:
        clean_name, category = factor_mapping[raw_val]
        mapped_data.append({'Factor_Name': clean_name, 'Factor_Category': category})

factor_df = pd.DataFrame(mapped_data).drop_duplicates().reset_index(drop=True)
factor_df['Factor_ID'] = pd.Series(range(1, len(factor_df) + 1), dtype='Int64')
factor_df[['Factor_ID', 'Factor_Name', 'Factor_Category']].to_csv(OUTPUT_DIR / f'{PREFIX}Contributing_Factor.csv', index=False)

# --- Locations ---
locations = crashes_final[['latitude', 'longitude', 'zip_code', 'Precinct_ID']].drop_duplicates().reset_index(drop=True)
locations['Location_ID'] = pd.Series(range(1, len(locations) + 1), dtype='Int64')
locations[['Location_ID', 'longitude', 'latitude', 'zip_code', 'Precinct_ID']].to_csv(OUTPUT_DIR / f'{PREFIX}Location.csv', index=False)


print("7. Erstelle Faktentabellen...")
crash_merge = pd.merge(crashes_final, locations, on=['latitude', 'longitude', 'zip_code', 'Precinct_ID'], how='left')
crash_out = crash_merge[['collision_id', 'clean_date', 'crash_time', 'Location_ID', 'Weather_ID']].copy()
crash_out.rename(columns={'collision_id': 'Collision_ID', 'clean_date': 'Crash_Date', 'crash_time': 'Crash_Time'}, inplace=True)

crash_out['Collision_ID'] = crash_out['Collision_ID'].astype('Int64')
crash_out['Weather_ID'] = crash_out['Weather_ID'].astype('Int64') 
crash_out['Location_ID'] = crash_out['Location_ID'].astype('Int64') 
crash_out[['Collision_ID', 'Crash_Date', 'Crash_Time', 'Location_ID', 'Weather_ID']].to_csv(OUTPUT_DIR / f'{PREFIX}Crash.csv', index=False)

valid_collision_ids = crash_out['Collision_ID'].unique()
vehicles_filtered = vehicles_df[vehicles_df['collision_id'].isin(valid_collision_ids)].copy()
persons_filtered = persons_df[persons_df['collision_id'].isin(valid_collision_ids)].copy()

# --- FAHRZEUGE (Vektorisiert & Granular) ---
v_type = vehicles_filtered['vehicle_type'].str.upper().fillna('')

# 1. Alle Conditions sauber definieren (Priorität von oben nach unten!)
cond_school_bus = v_type.str.contains('SCH|SCL', regex=True)
cond_bus = v_type.str.contains('BUS|MTA|OMNIBUS|COACH|ACCESS|SHUTTLE|TRANSI', regex=True)

cond_fire = v_type.str.contains('FIRE|FDNY|NYFD|LADDER', regex=True)
cond_amb = v_type.str.contains('AMBULANCE|AMB|EMS|EMT', regex=True)
cond_pol = v_type.str.contains('POLICE|NYPD|PATROL|RMP', regex=True)
cond_emg = v_type.str.contains('EMERGENCY|AMU|RESCUE|GOV|ARMY', regex=True)

cond_garbage = v_type.str.contains('GARB|GARGAGE|SANITA|DSNY|TRASH|SWEEP|BROOM', regex=True)
cond_delivery = v_type.str.contains('DELIV|DELV|COURIER|FEDEX|FED E|FEDERAL EX|UPS|MAIL|POST|USPS|U-HAUL|U HAUL|UHAUL|U-HAL', regex=True)
cond_commercial = v_type.str.contains('COMMERCIAL|COM|FREIGHT|VENDOR|BOBCAT|BACKHOE|LIFT|ICE CREAM|FORK|CRANE', regex=True)
cond_truck = v_type.str.contains('TRUCK|TRK|TRU|TRACT|TRAC|TRAIL|BOX|DUMP|PICK|P/U|FLAT|TOW|CHASSIS', regex=True)

cond_taxi = v_type.str.contains('TAXI|CAB', regex=True)
cond_livery = v_type.str.contains('LIMO|LIVERY|UBER|LYFT', regex=True)

cond_pedicab = v_type.str.contains('PEDICAB|PEDI CAB|RICKSHAW', regex=True)
cond_escooter = v_type.str.contains('E-SCO|E-SKA|HOVER|BOARD|SEGWAY|ONEWHEEL', regex=True)
cond_ebike = v_type.str.contains('E-BIKE|ELECTRIC|ELETRIC|E-BI', regex=True)
cond_bike = v_type.str.contains('BIKE|BICYCLE|CYCL', regex=True)

cond_moped = v_type.str.contains('MOPED|MOP|REVEL', regex=True)
cond_scooter = v_type.str.contains('SCOOT|SCOT|SCO|VESPA', regex=True)
cond_atv = v_type.str.contains('ATV|DIRT', regex=True)
cond_moto = v_type.str.contains('MOTOR|MOTO', regex=True)

cond_minivan = v_type.str.contains('MINI', regex=True)
cond_van = v_type.str.contains('VAN|VAHN|TRANSIT|SPRIN|ECONO', regex=True)

cond_wagon = v_type.str.contains('STATION', regex=True)
cond_suv = v_type.str.contains('SUV|SPORT|SUBUR|SUBN|JEEP', regex=True)

cond_pass = v_type.str.contains('PASS|SEDAN|SEDN|4 DR|2 DR|COUPE|CONV|CAR|AUTO|4D|2D|4S|SDN', regex=True)

cond_unknown = (v_type == '') | (v_type == 'UNKNOWN')

# 2. Conditions in die Liste packen
conditions = [
    cond_school_bus, cond_bus, 
    cond_fire, cond_amb, cond_pol, cond_emg, 
    cond_garbage, cond_delivery, cond_commercial, cond_truck, 
    cond_taxi, cond_livery, 
    cond_pedicab, cond_escooter, cond_ebike, cond_bike, 
    cond_moped, cond_scooter, cond_atv, cond_moto, 
    cond_minivan, cond_van, 
    cond_wagon, cond_suv, 
    cond_pass, 
    cond_unknown
]

# 3. Zugehörige Ein-Wort-Choices
choices = [
    'Schoolbus', 'Bus', 
    'Firetruck', 'Ambulance', 'Police', 'Emergency', 
    'Sanitation', 'Delivery', 'Commercial', 'Truck', 
    'Taxi', 'Livery', 
    'Pedicab', 'Escooter', 'Ebike', 'Bicycle', 
    'Moped', 'Scooter', 'ATV', 'Motorcycle', 
    'Minivan', 'Van', 
    'Wagon', 'SUV', 
    'Passenger', 
    'Unknown'
]

# 4. Anwenden und Mergen
vehicles_filtered['clean_type'] = np.select(conditions, choices, default='Other')
vehicle_merge = pd.merge(vehicles_filtered, vehicle_type_df, left_on='clean_type', right_on='Vehicle_Type_Name', how='left')

# 5. Finale Fahrzeugtabelle bauen
vehicle_out = vehicle_merge[['unique_id', 'collision_id', 'state_registration', 'vehicle_year', 'Vehicle_Type_ID']].copy()
vehicle_out.rename(columns={'unique_id': 'Vehicle_ID', 'collision_id': 'Collision_ID', 'state_registration': 'State_Registration', 'vehicle_year': 'Vehicle_Year'}, inplace=True)
vehicle_out.dropna(subset=['Vehicle_ID'], inplace=True)
vehicle_out['Vehicle_ID'] = vehicle_out['Vehicle_ID'].astype('Int64')
vehicle_out['Collision_ID'] = vehicle_out['Collision_ID'].astype('Int64')
vehicle_out[['Vehicle_ID', 'Collision_ID', 'State_Registration', 'Vehicle_Year', 'Vehicle_Type_ID']].to_csv(OUTPUT_DIR / f'{PREFIX}Vehicle.csv', index=False)


# --- FAHRZEUG-FAKTOREN (Vektorisiert & Bereinigt) ---
factors_melted = vehicles_filtered[['unique_id', 'contributing_factor_1', 'contributing_factor_2']].melt(
    id_vars=['unique_id'], value_vars=['contributing_factor_1', 'contributing_factor_2'], value_name='Raw_Factor'
).dropna(subset=['Raw_Factor'])

# Mappe die Rohdaten auf unsere sauberen Namen
factors_melted['clean_key'] = factors_melted['Raw_Factor'].str.strip().str.lower()
factors_melted['Factor_Name'] = factors_melted['clean_key'].map(lambda x: factor_mapping[x][0] if x in factor_mapping else None)

# Alle unlogischen oder undefinierten fallen hier als NaN raus
factors_melted = factors_melted.dropna(subset=['Factor_Name'])

factors_mapped = pd.merge(factors_melted, factor_df, on='Factor_Name', how='inner')
vehicle_factors_out = factors_mapped[['unique_id', 'Factor_ID']].drop_duplicates()
vehicle_factors_out.rename(columns={'unique_id': 'Vehicle_ID'}, inplace=True)

vehicle_factors_out['Vehicle_ID'] = vehicle_factors_out['Vehicle_ID'].astype('Int64')
vehicle_factors_out['Factor_ID'] = vehicle_factors_out['Factor_ID'].astype('Int64')
vehicle_factors_out.to_csv(OUTPUT_DIR / f'{PREFIX}Vehicle_Factors.csv', index=False)


# --- PERSONEN ---
person_out = persons_filtered[['unique_id', 'collision_id', 'vehicle_id', 'person_type', 'ped_role', 'person_injury', 'person_age', 'person_sex']].copy()
person_out.rename(columns={
    'unique_id': 'Person_ID', 'collision_id': 'Collision_ID', 'vehicle_id': 'Vehicle_ID', 
    'person_type': 'Person_Type', 'ped_role': 'Person_Role', 'person_injury': 'Person_Injury', 
    'person_age': 'Person_Age', 'person_sex': 'Person_Sex'
}, inplace=True)
person_out.dropna(subset=['Person_ID'], inplace=True)

person_out['Person_ID'] = person_out['Person_ID'].astype('Int64')
person_out['Collision_ID'] = person_out['Collision_ID'].astype('Int64')
person_out['Vehicle_ID'] = person_out['Vehicle_ID'].astype('Int64')
person_out[['Person_ID', 'Collision_ID', 'Vehicle_ID', 'Person_Type', 'Person_Role', 'Person_Injury', 'Person_Age', 'Person_Sex']].to_csv(OUTPUT_DIR / f'{PREFIX}Person.csv', index=False)

print(f"Tabellen (für {FILTER_YEAR}) liegen in '{OUTPUT_DIR}'.")