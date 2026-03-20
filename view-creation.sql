-- =============================================
-- 1. Dimension: Location
-- =============================================
CREATE OR ALTER VIEW vw_Dim_Location AS
SELECT 
    l.Location_ID,
    b.Borough_Name,
    p.Precinct_Name,
    l.Zip_Code,
    CAST(l.Longitude AS FLOAT) AS Longitude,
    CAST(l.Latitude AS FLOAT) AS Latitude
FROM Location l
LEFT JOIN Precinct p ON l.Precinct_ID = p.Precinct_ID
LEFT JOIN Borough b ON p.Borough_ID = b.Borough_ID;
GO

-- =============================================
-- 2. Dimension: Time & Weather (Kombiniert)
-- =============================================
CREATE OR ALTER VIEW vw_Dim_Time_Weather AS
SELECT 
    w.Weather_ID AS Time_Weather_ID,
    w.Measure_Date AS [Date],
    YEAR(w.Measure_Date) AS [Year],
    DATEPART(QUARTER, w.Measure_Date) AS [Quarter],
    MONTH(w.Measure_Date) AS [Month],
    DAY(w.Measure_Date) AS [Day],
    DATEPART(HOUR, w.Measure_Time) AS [Hour],
    
    CAST(w.Temp_Celsius AS FLOAT) AS Temp_Celsius,
    CAST(w.Visibility_Miles AS FLOAT) AS Visibility_Miles,
    CAST(w.Precipitation_Inches AS FLOAT) AS Precipitation_Inches,
    CAST(w.Snow_Depth_Inches AS FLOAT) AS Snow_Depth_Inches,
    CAST(w.Wind_Gust_Speed_MPH AS FLOAT) AS Wind_Gust_Speed_MPH,
    
    w.Weather_Condition_Text
FROM Weather w;
GO

-- =============================================
-- 3. Dimension: Vehicle
-- =============================================
CREATE OR ALTER VIEW vw_Dim_Vehicle AS
SELECT 
    v.Vehicle_ID,
    v.State_Registration,
    CAST(v.Vehicle_Year AS INT) AS Vehicle_Year,
    vt.Vehicle_Type_Name AS Vehicle_Type
FROM Vehicle v
LEFT JOIN Vehicle_Type vt ON v.Vehicle_Type_ID = vt.Vehicle_Type_ID;
GO

-- =============================================
-- 4. Dimension: Person
-- =============================================
CREATE OR ALTER VIEW vw_Dim_Person AS
SELECT 
    Person_ID,
    Person_Type,
    Person_Role,
    Person_Injury,
    CAST(Person_Age AS INT) AS Person_Age,
    Person_Sex
FROM Person;
GO

-- =============================================
-- 5. Dimension: Contributing Factor
-- =============================================
CREATE OR ALTER VIEW vw_Dim_Contributing_Factor AS
SELECT 
    Factor_ID,
    Factor_Name,
    Factor_Category
FROM Contributing_Factor;
GO

-- =============================================
-- 6. Bridge: Vehicle Factor (Angehängt an Vehicle)
-- =============================================
CREATE OR ALTER VIEW vw_Bridge_Vehicle_Factor AS
SELECT 
    Vehicle_ID,
    Factor_ID
FROM Vehicle_Factors;
GO

-- =============================================
-- 7. Bridge: Involvement (Verknüpft Crash, Vehicle, Person)
-- =============================================
CREATE OR ALTER VIEW vw_Bridge_Involvement AS
SELECT 
    ROW_NUMBER() OVER (ORDER BY Collision_ID, Person_ID) AS Involvement_ID,
    
    Collision_ID AS Crash_ID,
    Vehicle_ID,
    Person_ID
FROM Person;
GO

-- =============================================
-- 8. Bridge: Crash Factor (Neu - Verknüpft Crash und Factor über Vehicle)
-- =============================================
CREATE OR ALTER VIEW vw_Bridge_Crash_Factor AS
SELECT DISTINCT 
    bi.Crash_ID, 
    vf.Factor_ID
FROM vw_Bridge_Involvement bi
JOIN vw_Bridge_Vehicle_Factor vf ON bi.Vehicle_ID = vf.Vehicle_ID
WHERE bi.Crash_ID IS NOT NULL AND vf.Factor_ID IS NOT NULL;
GO

-- =============================================
-- 9. Dimension: Severity (Statisch generiert)
-- =============================================
CREATE OR ALTER VIEW vw_Dim_Severity AS
SELECT CAST(1 AS INT) AS Severity_ID, 'Fatal' AS Damage_Category, 'High' AS Severity_Class
UNION ALL
SELECT CAST(2 AS INT) AS Severity_ID, 'Injury' AS Damage_Category, 'Medium' AS Severity_Class
UNION ALL
SELECT CAST(3 AS INT) AS Severity_ID, 'Property Damage Only' AS Damage_Category, 'Low' AS Severity_Class;
GO

-- =============================================
-- 10. Fact: Crashes (Aggregiert die Metriken)
-- =============================================
CREATE OR ALTER VIEW vw_Fact_Crashes AS
WITH CrashSeverity AS (
    SELECT 
        c.Collision_ID,
        MAX(CASE 
            WHEN p.Person_Injury = 'Killed' THEN 1
            WHEN p.Person_Injury = 'Injured' THEN 2
            ELSE 3 
        END) AS Computed_Severity_ID
    FROM Crash c
    LEFT JOIN Person p ON c.Collision_ID = p.Collision_ID
    GROUP BY c.Collision_ID
)
SELECT 
    c.Collision_ID AS Crash_ID,
    CAST(1 AS INT) AS Crash_Count,
    
    CAST(ISNULL(SUM(CASE WHEN p.Person_Injury = 'Injured' THEN 1 ELSE 0 END), 0) AS INT) AS Persons_Injured,
    CAST(ISNULL(SUM(CASE WHEN p.Person_Injury = 'Killed' THEN 1 ELSE 0 END), 0) AS INT) AS Persons_Killed,
    
    CAST(ISNULL(SUM(CASE WHEN p.Person_Type LIKE '%Cyclist%' AND p.Person_Injury = 'Injured' THEN 1 ELSE 0 END), 0) AS INT) AS Cyclists_Injured,
    CAST(ISNULL(SUM(CASE WHEN p.Person_Type LIKE '%Cyclist%' AND p.Person_Injury = 'Killed' THEN 1 ELSE 0 END), 0) AS INT) AS Cyclists_Killed,
    
    CAST(ISNULL(SUM(CASE WHEN p.Person_Type LIKE '%Motorist%' AND p.Person_Injury = 'Injured' THEN 1 ELSE 0 END), 0) AS INT) AS Motorists_Injured,
    CAST(ISNULL(SUM(CASE WHEN p.Person_Type LIKE '%Motorist%' AND p.Person_Injury = 'Killed' THEN 1 ELSE 0 END), 0) AS INT) AS Motorists_Killed,
    
    CAST(ISNULL(SUM(CASE WHEN vt.Vehicle_Type_Name LIKE '%Truck%' AND p.Person_Injury = 'Injured' THEN 1 ELSE 0 END), 0) AS INT) AS Heavy_Vehicle_Injured,
    CAST(ISNULL(SUM(CASE WHEN vt.Vehicle_Type_Name LIKE '%Truck%' AND p.Person_Injury = 'Killed' THEN 1 ELSE 0 END), 0) AS INT) AS Heavy_Vehicle_Killed,
    
    CAST(c.Location_ID AS INT) AS Location_ID,
    CAST(c.Weather_ID AS INT) AS Time_Weather_ID,
    CAST(ISNULL(cs.Computed_Severity_ID, 3) AS INT) AS Severity_ID

FROM Crash c
LEFT JOIN Person p ON c.Collision_ID = p.Collision_ID
LEFT JOIN Vehicle v ON p.Vehicle_ID = v.Vehicle_ID
LEFT JOIN Vehicle_Type vt ON v.Vehicle_Type_ID = vt.Vehicle_Type_ID
LEFT JOIN CrashSeverity cs ON c.Collision_ID = cs.Collision_ID
GROUP BY 
    c.Collision_ID,
    c.Location_ID,
    c.Weather_ID,
    cs.Computed_Severity_ID;
GO