-- 1. Borough Dummy
INSERT INTO Borough (Borough_ID, Borough_Name) 
VALUES (-1, 'Unknown Borough');

-- 2. Precinct Dummy
INSERT INTO Precinct (Precinct_ID, Precinct_Name, Borough_ID) 
VALUES (-1, 'Unknown Precinct', -1);

-- 3. Location Dummy
INSERT INTO Location (Location_ID, Longitude, Latitude, Zip_Code, Precinct_ID) 
VALUES (-1, 0.0, 0.0, 'Unknown', -1);

-- 4. Weather Dummy
INSERT INTO Weather (Weather_ID, Weather_Station, Measure_Date, Measure_Time, Temp_Celsius, Visibility_Miles, Precipitation_Inches, Snow_Depth_Inches, Wind_Gust_Speed_MPH, Weather_Condition_Text)
VALUES (-1, 'UNKNOWN', '1900-01-01', '00:00:00', 0.0, 0.0, 0.0, 0.0, 0.0, 'Wetterdaten fehlen');

-- 5. Vehicle Type Dummy
INSERT INTO Vehicle_Type (Vehicle_Type_ID, Vehicle_Type_Name, Vehicle_Category)
VALUES (-1, 'Unknown Type', 'Unknown Category');

-- 6. Contributing Factor Dummy
INSERT INTO Contributing_Factor (Factor_ID, Factor_Name, Factor_Category)
VALUES (-1, 'Unspecified Factor', 'Unknown Category');

-- 7. Vehicle Dummy
INSERT INTO Vehicle (Vehicle_ID, Collision_ID, State_Registration, Vehicle_Year, Vehicle_Type_ID)
VALUES (0, NULL, 'Kein Fahrzeug', NULL, -1);

-- 8. Person Dummy
INSERT INTO Person (Person_ID, Collision_ID, Vehicle_ID, Person_Type, Person_Role, Person_Injury, Person_Age, Person_Sex)
VALUES (-1, NULL, 0, 'Unknown Person', 'Unknown Role', 'Unknown', NULL, 'U');
GO

-- 9. Crash Dummy
INSERT INTO Crash (Collision_ID, Crash_Date, Crash_Time, Location_ID, Weather_ID)
VALUES (-1, '1900-01-01', '00:00:00', -1, -1);
GO

-- 1. Unabhängige Dimensionen
INSERT INTO Borough (Borough_ID, Borough_Name)
SELECT 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Borough_ID AS VARCHAR(255)))), 
    CAST(Borough_Name AS VARCHAR(100))
FROM st_Borough;

INSERT INTO Weather (Weather_ID, Weather_Station, Measure_Date, Measure_Time, Temp_Celsius, Visibility_Miles, Precipitation_Inches, Snow_Depth_Inches, Wind_Gust_Speed_MPH, Weather_Condition_Text)
SELECT 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Weather_ID AS VARCHAR(255)))), 
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

INSERT INTO Vehicle_Type (Vehicle_Type_ID, Vehicle_Type_Name, Vehicle_Category)
SELECT 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Vehicle_Type_ID AS VARCHAR(255)))), 
    CAST(Vehicle_Type_Name AS VARCHAR(100)),
    CAST(Vehicle_Category AS VARCHAR(100))
FROM st_Vehicle_Type;

INSERT INTO Contributing_Factor (Factor_ID, Factor_Name, Factor_Category)
SELECT 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Factor_ID AS VARCHAR(255)))), 
    CAST(Factor_Name AS VARCHAR(255)), 
    CAST(Factor_Category AS VARCHAR(100))
FROM st_Contributing_Factor;


-- 2. Abhängige Dimensionen
INSERT INTO Precinct (Precinct_ID, Precinct_Name, Borough_ID)
SELECT 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Precinct_ID AS VARCHAR(255)))), 
    CAST(Precinct_Name AS VARCHAR(100)), 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Borough_ID AS VARCHAR(255))))
FROM st_Precinct;

INSERT INTO Location (Location_ID, Longitude, Latitude, Zip_Code, Precinct_ID)
SELECT 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Location_ID AS VARCHAR(255)))), 
    TRY_CONVERT(DECIMAL(9,6), CAST(longitude AS VARCHAR(255))), 
    TRY_CONVERT(DECIMAL(9,6), CAST(latitude AS VARCHAR(255))), 
    CAST(zip_code AS VARCHAR(20)), 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Precinct_ID AS VARCHAR(255))))
FROM st_Location;


-- 3. Faktentabelle
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
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(Vehicle_ID AS VARCHAR(255)))), 
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
    ON TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(vf.Vehicle_ID AS VARCHAR(255)))) = v.Vehicle_ID;

INSERT INTO Person (Person_ID, Collision_ID, Vehicle_ID, Person_Type, Person_Role, Person_Injury, Person_Age, Person_Sex)
SELECT 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(p.Person_ID AS VARCHAR(255)))), 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(p.Collision_ID AS VARCHAR(255)))), 
    v.Vehicle_ID,
    CAST(p.Person_Type AS VARCHAR(100)), 
    CAST(p.Person_Role AS VARCHAR(100)), 
    CAST(p.Person_Injury AS VARCHAR(100)), 
    TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(p.Person_Age AS VARCHAR(255)))), 
    CAST(p.Person_Sex AS VARCHAR(10))
FROM st_Person p
LEFT JOIN Vehicle v 
    ON TRY_CONVERT(INT, TRY_CONVERT(FLOAT, CAST(p.Vehicle_ID AS VARCHAR(255)))) = v.Vehicle_ID;