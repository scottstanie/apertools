from math import pi

SENTINEL_WAVELENGTH = 5.5465763  # cm
PHASE_TO_CM_S1 = SENTINEL_WAVELENGTH / (4 * pi)
P2MM_S1 = PHASE_TO_CM_S1 * 10 * 365  # (cm / day) -> (mm / yr)

UAVSAR_WAVELENGTH = 23.8403545  # cm
PHASE_TO_CM_UA = UAVSAR_WAVELENGTH / (4 * pi)
P2MM_UA = PHASE_TO_CM_UA * 10 * 365


# Dfault: Sentinel1
PHASE_TO_CM = PHASE_TO_CM_S1
P2MM = P2MM_S1

PLATFORM_ABBREVIATIONS = {
    "s1": ["sentinel", "s1a", "sentinel1"],
    "ua": ["uavsar"],
}

PHASE_TO_CM_MAP = {
    "ua": PHASE_TO_CM_UA,
    "s1": PHASE_TO_CM_S1,
}
WAVELENGTH_MAP = {
    "ua": UAVSAR_WAVELENGTH,
    "s1": SENTINEL_WAVELENGTH,
}

PLATFORM_CHOICES = list(PHASE_TO_CM_MAP.keys())
COORDINATES_CHOICES = ["geo", "rdr"]
