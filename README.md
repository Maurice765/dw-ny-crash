# DW-NY-CRASH
Dieses Repository enthält die ETL-Pipeline und die SQL-Skripte zum Aufbau eines relationalen Data Warehouses (Kimball-Modell) zur multidimensionalen Analyse von NYPD-Fahrzeugunfällen (Crashes) in Kombination mit JFK-Wetterdaten.

## 📂 Projektstruktur

* **`data/`**: Ordner für die Rohdaten (NYPD Crashes, Vehicles, Persons, JFK Weather, Precinct GeoJSON).
* **`output_tables/`**: Zielordner, in dem das Python-Skript die bereinigten und transformierten CSV-Dateien für den Datenbank-Import ablegt.
* **`etl-transform.py`**: Das zentrale ETL-Skript. Liest Rohdaten, bereinigt sie, führt räumliche Zuweisungen durch (Spatial Join) und exportiert die bereiten Staging-Tabellen.

### 🗄️ SQL Skripte

1. **`create-basis-tables.sql`**: Erstellt das physische Tabellenschema in der SQL-Datenbank
2. **`import_basis_data.sql`**: Legt die Dummy-Datensätze an und importiert die Staging-Daten fehlerfrei und typensicher in die Basistabellen.
3. **`create-mdm-tables.sql`**: Erstellt die optimierten Views als direkte Lese-Schicht für das SSAS-Cube.
4. **`delete_basis_data.sql`**: (Optional) Skript zum sauberen Leeren der Tabellen.

## 🚀 Setup & Ausführung

**1. Python-Umgebung:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Daten aufbereiten:**
```bash
python etl-transform.py
```
