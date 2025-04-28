import numpy as np
import pandas as pd
from src.configurations.config import FIRE_STATISTICS_CSV

file_path = FIRE_STATISTICS_CSV

def extract():

    fire_df = pd.read_csv(file_path, low_memory=False)

    fire_df.drop(['X', 'Y', 'OBJECTID', 'FIRE_NAME', 'LOCAL_FIRE_NUMBER', 'LOCATION', 'TOWNSHIP', 'RANGE', 'SECTION', 'SUB_SECTION', 'REPORT_UNIT', 'DISTRICT',
                  'FIRE_NUMBER', 'ADMIN_UNIT', 'PROTECTION_UNIT', 'PROTECTION_UNIT_NAME', 'OWNERSHIP_UNIT', 'OWNERSHIP_UNIT_NAME', 'TOPO_LANDFORM_ORIGIN',
                  'STATE_CODE', 'COUNTY', 'COUNTY_NAME', 'COUNTY_STATE_CODE', 'FIRE_MANAGEMENT_CODE', 'DISCOVERED_BY_DESCR', 'OBJECTIVES',
                  'COMPLEX_FIRE', 'COMPLEX_NAME', 'CONTAINED', 'AGENCY_ACRES', 'OTHER_ACRES_INSIDE', 'OTHER_ACRES_OUTSIDE', 'FIRE_SIZE_CLASS',
                  'PRESCRIBED_ACRES', 'WUI_FIRE', 'WUI_ACRES', 'STATION_TYPE', 'STATION_NAME', 'WIND_SPEED', 'OTHER_FUEL_MODEL', 'COVER_CLASS', 'SUBMITTED_DATE', 'APPROVED_DATE',                                                                                                                                                                                                                                                       "PRINCIPAL_MERIDIAN",
                  "ADMIN_UNIT_NAME", "PROTECTING_AGENCY_AT_ORIGIN", "OWNERSHIP_AGENCY_AT_ORIGIN", "LAT_DEG", "LAT_MIN", "LAT_SEC",
                  "LONG_DEG", "LONG_MIN", "LONG_SEC", "INITIAL_RESPONSE", "INITIAL_STRATEGY", "STRATEGY_MET", "FIRE_OUT", "TOTAL_ACRES_BURNED", "RECORD_ENTRY_DATE", "CREATED_DATE",
                  "LAST_MODIFIED_DATE", "FIRE_DETECTOR", "NFDRS_FUEL_MODEL", "REP_WX_STATION", "LOCAL_TIMEZONE"], axis=1, inplace=True)

    fire_df.rename(columns={"POO_LATITUDE": "LATITUDE", "POO_LONGITUDE" : "LONGITUDE"}, inplace=True)

    fire_df["STATE_NAME"] = fire_df["STATE_NAME"].str.split('-').str[-1].str.strip().astype(str)
    fire_df["STATE_NAME"] = fire_df["STATE_NAME"].fillna('N/A').astype(str)

    fire_df["LATITUDE"] = fire_df["LATITUDE"].fillna(0.0).astype(float)

    fire_df["LONGITUDE"] = fire_df["LONGITUDE"].fillna(0.0).astype(float)

    fire_df["DISCOVER_YEAR"] = pd.to_numeric(fire_df["DISCOVER_YEAR"], errors='coerce')
    fire_df["DISCOVER_YEAR"] = fire_df["DISCOVER_YEAR"].fillna(0).astype(int)

    fire_df["IGNITION"] = fire_df["IGNITION"].fillna('N/A').astype(str)
    fire_df["DISCOVERY"] = fire_df["DISCOVERY"].fillna('N/A').astype(str)

    fire_df["STATISTICAL_CAUSE"] = fire_df["STATISTICAL_CAUSE"].str.split('-').str[-1].str.strip()
    fire_df["STATISTICAL_CAUSE"] = fire_df.STATISTICAL_CAUSE.fillna('N/A').astype(str)

    fire_df["PRESCRIBED_FIRE"] = fire_df.PRESCRIBED_FIRE.fillna('N/A').astype(str)

    fire_df["FIRE_INTENSITY_LEVEL"] = fire_df["FIRE_INTENSITY_LEVEL"].str.split(' - ').str[1].str.strip()
    fire_df["FIRE_INTENSITY_LEVEL"] = fire_df["FIRE_INTENSITY_LEVEL"].replace('', np.nan)
    fire_df["FIRE_INTENSITY_LEVEL"] = fire_df["FIRE_INTENSITY_LEVEL"].fillna('N/A').astype(str)

    fire_df["SLOPE"] = fire_df["SLOPE"].fillna(0.0).astype(float)

    fire_df["ASPECT"] = fire_df["ASPECT"].str.split(' ').str[1].str.strip()
    fire_df["ASPECT"] = fire_df["ASPECT"].replace('', np.nan)
    fire_df["ASPECT"] = fire_df["ASPECT"].fillna('N/A').astype(str)

    fire_df["ELEVATION"] = fire_df["ELEVATION"].fillna(0.0).astype(float)

    intensity_map = {
        "Flame Length 0-2'": 1, "Flame Length >2-4'": 2, "Flame Length >4-6'": 3,
        "Flame Length >6-8'": 4, "Flame Length >8-12'": 5, "Flame Length >12'": 6,
        "HISTORICAL": 0, "N/A": 0
    }
    fire_df["FIRE_INTENSITY_LEVEL"] = fire_df["FIRE_INTENSITY_LEVEL"].map(intensity_map)
    #
    # # fire_df["FIRE_OCCURRENCE"] = fire_df["FIRE_INTENSITY_LEVEL"].apply(lambda x: 1 if x > 0 else 0)

    fire_df = fire_df.iloc[50000:60000]

    return fire_df
