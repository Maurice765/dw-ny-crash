CREATE TABLE Borough (
    Borough_ID INT PRIMARY KEY,
    Borough_Name VARCHAR(100)
);

CREATE TABLE Weather (
    Weather_ID INT PRIMARY KEY,
    Measure_Date DATE,
    Measure_Time TIME,
    Temp_Celsius DECIMAL(5,2),
    Visibility_Miles DECIMAL(5,2),
    Precipitation_Inches DECIMAL(5,2),
    Snow_Depth_Inches DECIMAL(5,2),
    Wind_Gust_Speed_MPH DECIMAL(5,2),
    Weather_Condition_Text VARCHAR(255)
);

CREATE TABLE Vehicle_Type (
    Vehicle_Type_ID INT PRIMARY KEY,
    Vehicle_Type_Name VARCHAR(100),
    Vehicle_Type_Category VARCHAR(100)
);

CREATE TABLE Contributing_Factor (
    Factor_ID INT PRIMARY KEY,
    Factor_Name VARCHAR(255),
    Factor_Category VARCHAR(100)
);

CREATE TABLE Precinct (
    Precinct_ID INT PRIMARY KEY,
    Precinct_Name VARCHAR(100),
    Borough_ID INT,
    FOREIGN KEY (Borough_ID) REFERENCES Borough(Borough_ID)
);

CREATE TABLE Location (
    Location_ID INT PRIMARY KEY,
    Longitude DECIMAL(9,6),
    Latitude DECIMAL(9,6),
    Zip_Code VARCHAR(20),
    Precinct_ID INT,
    FOREIGN KEY (Precinct_ID) REFERENCES Precinct(Precinct_ID)
);

CREATE TABLE Crash (
    Collision_ID INT PRIMARY KEY,
    Crash_Date DATE,
    Crash_Time TIME,
    Location_ID INT,
    Weather_ID INT,
    FOREIGN KEY (Location_ID) REFERENCES Location(Location_ID),
    FOREIGN KEY (Weather_ID) REFERENCES Weather(Weather_ID)
);

CREATE TABLE Vehicle (
    Vehicle_ID INT PRIMARY KEY,
    Collision_ID INT,
    State_Registration VARCHAR(50),
    Vehicle_Year INT,
    Vehicle_Type_ID INT,
    FOREIGN KEY (Collision_ID) REFERENCES Crash(Collision_ID),
    FOREIGN KEY (Vehicle_Type_ID) REFERENCES Vehicle_Type(Vehicle_Type_ID)
);

-- Tabelle zur Auflösung der m:n-Beziehung zwischen Vehicle und Contributing_Factor
CREATE TABLE Vehicle_Factors (
    Vehicle_ID INT,
    Factor_ID INT,
    PRIMARY KEY (Vehicle_ID, Factor_ID),
    FOREIGN KEY (Vehicle_ID) REFERENCES Vehicle(Vehicle_ID),
    FOREIGN KEY (Factor_ID) REFERENCES Contributing_Factor(Factor_ID)
);

CREATE TABLE Person (
    Person_ID INT PRIMARY KEY,
    Collision_ID INT,
    Vehicle_ID INT,
    Person_Type VARCHAR(100),
    Person_Role VARCHAR(100),
    Person_Injury VARCHAR(100),
    Person_Age INT,
    Person_Sex VARCHAR(10),
    FOREIGN KEY (Collision_ID) REFERENCES Crash(Collision_ID),
    FOREIGN KEY (Vehicle_ID) REFERENCES Vehicle(Vehicle_ID)
);