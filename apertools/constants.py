from numpy import pi

SENTINEL_WAVELENGTH = 5.5465763  # cm
PHASE_TO_CM = SENTINEL_WAVELENGTH / (4 * pi)
P2MM = PHASE_TO_CM * 10 * 365  # (cm / day) -> (mm / yr)
