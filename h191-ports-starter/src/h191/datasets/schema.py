from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

# Canonical dtypes for the processed panel
DTYPES: Dict[str, Any] = {
    'port_id': 'category',
    'date': 'datetime64[ns]',
    'year': 'int32',
    'month': 'int8',
    'teu': 'Int64',
    'moves': 'Int64',
    'vessel_calls': 'Int64',
    'turnaround_avg_hr': 'float64',
    'wait_avg_hr': 'float64',
    'employees': 'Int64',
    'hours_worked': 'float64',
    'kl': 'float64',
    'sts_cranes': 'Int64',
    'yard_cranes': 'Int64',
    'berth_depth_m': 'float64',
    'berth_length_m': 'float64',
    'competition_active': 'Int8',
    'privatized': 'Int8',
    'peer_competition_active': 'Int8',
    'holiday_intensity': 'float64',
    'weather_shock': 'float64',
    'container_share_transship': 'float64',
    'notes': 'string',
}
