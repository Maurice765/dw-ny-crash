-- ==============================================================================
-- 1. UNABHÄNGIGE DIMENSIONEN
-- ==============================================================================

INSERT INTO Borough (Borough_ID, Borough_Name)
SELECT 
    TRY_CONVERT(INT, CAST(Borough_ID AS VARCHAR(255))), 
    CAST(Borough_Name AS VARCHAR(100))
FROM st_Borough;

INSERT INTO Weather (Weather_ID, Weather_Station, Measure_Date, Measure_Time, Temp_Celsius, Visibility_Miles, Precipitation_Inches, Snow_Depth_Inches, Wind_Gust_Speed_MPH, Weather_Condition_Text)
SELECT 
    TRY_CONVERT(INT, CAST(Weather_ID AS VARCHAR(255))), 
    CAST(Weather_Station AS VARCHAR(10)), 
    TRY_CONVERT(DATE, CAST(Measure_Date AS VARCHAR(255))), 
    TRY_CONVERT(TIME, CAST(Measure_Time AS VARCHAR(255))), 
    TRY_CONVERT(DECIMAL(5,2), CAST(Temp_Celsius AS VARCHAR(255))), 
    TRY_CONVERT(DECIMAL(5,2), CAST(Visibility_Miles AS VARCHAR(255))), 
    TRY_CONVERT(DECIMAL(5,2), CAST(Precipitation_Inches AS VARCHAR(255))), 
    TRY_CONVERT(DECIMAL(5,2), CAST(Snow_Depth_Inches AS VARCHAR(255))), 
    TRY_CONVERT(DECIMAL(5,2), CAST(Wind_Gust_Speed_MPH AS VARCHAR(255))), 
    CAST(Weather_Condition_Text AS VARCHAR(255))
FROM st_Weather;

INSERT INTO Vehicle_Type (Vehicle_Type_ID, Vehicle_Type_Name)
SELECT 
    TRY_CONVERT(INT, CAST(Vehicle_Type_ID AS VARCHAR(255))), 
    CAST(Vehicle_Type_Name AS VARCHAR(100))
FROM st_Vehicle_Type;

INSERT INTO Contributing_Factor (Factor_ID, Factor_Name, Factor_Category)
SELECT 
    TRY_CONVERT(INT, CAST(Factor_ID AS VARCHAR(255))), 
    CAST(Factor_Name AS VARCHAR(255)), 
    CAST(Factor_Category AS VARCHAR(100))
FROM st_Contributing_Factor;


-- ==============================================================================
-- 2. ABHÄNGIGE DIMENSIONEN
-- ==============================================================================

INSERT INTO Precinct (Precinct_ID, Precinct_Name, Borough_ID)
SELECT 
    TRY_CONVERT(INT, CAST(Precinct_ID AS VARCHAR(255))), 
    CAST(Precinct_Name AS VARCHAR(100)), 
    TRY_CONVERT(INT, CAST(Borough_ID AS VARCHAR(255)))
FROM st_Precinct;

INSERT INTO Location (Location_ID, Longitude, Latitude, Zip_Code, Precinct_ID)
SELECT 
    TRY_CONVERT(INT, CAST(Location_ID AS VARCHAR(255))), 
    TRY_CONVERT(DECIMAL(9,6), CAST(longitude AS VARCHAR(255))), 
    TRY_CONVERT(DECIMAL(9,6), CAST(latitude AS VARCHAR(255))), 
    CAST(zip_code AS VARCHAR(20)), 
    TRY_CONVERT(INT, CAST(Precinct_ID AS VARCHAR(255)))
FROM st_Location;


-- ==============================================================================
-- 3. FAKTENTABELLEN (Mit der unzerstörbaren COALESCE / TRY_CONVERT Logik)
-- ==============================================================================

INSERT INTO Crash (Collision_ID, Crash_Date, Crash_Time, Location_ID, Weather_ID)
SELECT 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Collision_ID AS VARCHAR(255)))), 
    TRY_CONVERT(DATE, CAST(Crash_Date AS VARCHAR(255))), 
    TRY_CONVERT(TIME, CAST(Crash_Time AS VARCHAR(255))), 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Location_ID AS VARCHAR(255)))), 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Weather_ID AS VARCHAR(255))))
FROM st_Crash;

INSERT INTO Vehicle (Vehicle_ID, Collision_ID, State_Registration, Vehicle_Year, Vehicle_Type_ID)
SELECT 
    COALESCE(CAST(TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Vehicle_ID AS VARCHAR(255)))) AS VARCHAR(255)), CAST(Vehicle_ID AS VARCHAR(255))), 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Collision_ID AS VARCHAR(255)))), 
    CAST(State_Registration AS VARCHAR(50)), 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Vehicle_Year AS VARCHAR(255)))), 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Vehicle_Type_ID AS VARCHAR(255))))
FROM st_Vehicle;

INSERT INTO Vehicle_Factors (Vehicle_ID, Factor_ID)
SELECT 
    v.Vehicle_ID,
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(vf.Factor_ID AS VARCHAR(255))))
FROM st_Vehicle_Factors vf
INNER JOIN Vehicle v 
    ON COALESCE(CAST(TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(vf.Vehicle_ID AS VARCHAR(255)))) AS VARCHAR(255)), CAST(vf.Vehicle_ID AS VARCHAR(255))) = v.Vehicle_ID;

INSERT INTO Person (Person_ID, Collision_ID, Vehicle_ID, Person_Type, Person_Role, Person_Injury, Person_Age, Person_Sex)
SELECT 
    COALESCE(CAST(TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(p.Person_ID AS VARCHAR(255)))) AS VARCHAR(255)), CAST(p.Person_ID AS VARCHAR(255))), 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(p.Collision_ID AS VARCHAR(255)))), 
    v.Vehicle_ID,
    CAST(p.Person_Type AS VARCHAR(100)), 
    CAST(p.Person_Role AS VARCHAR(100)), 
    CAST(p.Person_Injury AS VARCHAR(100)), 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(p.Person_Age AS VARCHAR(255)))), 
    CAST(p.Person_Sex AS VARCHAR(10))
FROM st_Person p
LEFT JOIN Vehicle v 
    ON COALESCE(CAST(TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(p.Vehicle_ID AS VARCHAR(255)))) AS VARCHAR(255)), CAST(p.Vehicle_ID AS VARCHAR(255))) = v.Vehicle_ID;