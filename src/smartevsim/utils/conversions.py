
"""Electrical power and current conversion helpers."""

import math

PF = 1.0  # Power factor, assumed to be 1 for simplicity
V_LL = 400.0  # Line-to-line voltage in volts for a 3-phase system

def convert_kw_to_A(p_kw: float) -> float:
    """Convert three-phase active power in kilowatts to amperes.

    Args:
        p_kw: Active power in kilowatts.

    Returns:
        Line current in amperes, assuming the module voltage and power factor.
    """
    return p_kw * 1000.0 / (math.sqrt(3) * V_LL * PF)

def convert_A_to_kw(i_a: float) -> float:
    """Convert three-phase current in amperes to active power in kilowatts.

    Args:
        i_a: Line current in amperes.

    Returns:
        Active power in kilowatts, assuming the module voltage and power factor.
    """
    return i_a * math.sqrt(3) * V_LL * PF / 1000.0
