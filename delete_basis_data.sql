-- 1. Faktentabellen leeren
DELETE FROM Person;
DELETE FROM Vehicle_Factors;
DELETE FROM Vehicle;
DELETE FROM Crash;

-- 2. Abhängige Dimensionen leeren
DELETE FROM Location;
DELETE FROM Precinct;

-- 3. Unabhängige Dimensionen leeren
DELETE FROM Contributing_Factor;
DELETE FROM Vehicle_Type;
DELETE FROM Weather;
DELETE FROM Borough;