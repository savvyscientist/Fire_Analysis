import matplotlib.pyplot as plt


EARTH_RADIUS = 6371000
KM_NEG_2TOM_NEG_2 = 10**-6
KM_TO_METERS = 10**6
DAYS_TO_SECONDS = 60 * 60 * 24
SCRIPTS_ENV_VARIABLES = "utilityEnvVarBA.json"
MONTHLIST = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
]
DISTINCT_COLORS = [
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FFFF00",
    "#FF00FF",
    "#00FFFF",
    "#FFA500",
    "#008000",
    "#800080",
    "#008080",
    "#800000",
    "#000080",
    "#808000",
    "#800080",
    "#FF6347",
    "#00CED1",
    "#FF4500",
    "#DA70D6",
    "#32CD32",
    "#FF69B4",
    "#8B008B",
    "#7FFF00",
    "#FFD700",
    "#20B2AA",
    "#B22222",
    "#FF7F50",
    "#00FA9A",
    "#4B0082",
    "#ADFF2F",
    "#F08080",
]
MASK_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
MONTHS_NUM = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
GFED_COVER_LABELS = {
    0: "Ocean",
    1: "BONA",
    2: "TENA",
    3: "CEAM",
    4: "NHSA",
    5: "SHSA",
    6: "EURO",
    7: "MIDE",
    8: "NHAF",
    9: "SHAF",
    10: "BOAS",
    11: "CEAS",
    12: "SEAS",
    13: "EQAS",
    14: "AUST",
    15: "Total",
}

LAND_COVER_LABELS = {
    0: "Water",
    1: "Boreal forest",
    2: "Tropical forest",
    3: "Temperate forest",
    4: "Temperate mosaic",
    5: "Tropical shrublands",
    6: "Temperate shrublands",
    7: "Temperate grasslands",
    8: "Woody savanna",
    9: "Open savanna",
    10: "Tropical grasslands",
    11: "Wetlands",
    12: "",
    13: "Urban",
    14: "",
    15: "Snow and Ice",
    16: "Barren",
    17: "Sparse boreal forest",
    18: "Tundra",
    19: "",
}
NUM_MONTHS = len(MONTHLIST)
MARKER = "o"
SECONDS_IN_A_YEAR = 60.0 * 60.0 * 24.0 * 365.0
KILOGRAMS_TO_GRAMS = 10.0**3
COLOR_MAP = plt.get_cmap("tab20")
SQM_TO_SQHA = 1e-4
