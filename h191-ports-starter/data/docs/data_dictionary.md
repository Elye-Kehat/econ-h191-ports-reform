# Data Dictionary — Canonical Port-Month Panel

This file documents the **canonical analysis table** we will build: `data/processed/ports_panel_month.parquet`.

| Column | Type | Description |
|---|---|---|
| port_id | string | 'haifa' or 'ashdod' (lowercase) |
| date | date (month-start) | Month index, e.g., 2021-09-01 |
| year | int | Calendar year |
| month | int | Calendar month (1..12) |
| teu | int | Loaded + empty TEU handled in month |
| moves | int | Total container moves in month (if available) |
| vessel_calls | int | Number of container vessel calls |
| turnaround_avg_hr | float | Avg vessel turnaround time (hours) |
| wait_avg_hr | float | Avg vessel waiting time at anchor (hours) |
| employees | int | Headcount in container ops |
| hours_worked | float | Total container-ops hours |
| kl | float | Computed K/L index (see `features.py`) |
| sts_cranes | int | Ship-to-shore cranes available in month |
| yard_cranes | int | RTG/RMG yard cranes available |
| berth_depth_m | float | Max usable depth for container berth (m) |
| berth_length_m | float | Total deep-water berth length available (m) |
| competition_active | int (0/1) | 1 if deep-water competing terminal operating this month |
| privatized | int (0/1) | 1 for months after Haifa legacy privatization date |
| peer_competition_active | int (0/1) | 1 if the *other* port has competition active (spillover) |
| holiday_intensity | float | Share of days affected by major holidays (0..1) |
| weather_shock | float | Optional: storm/wave proxy (z-score) |
| container_share_transship | float | Share of transshipment in throughput |
| notes | string | Free-form notes for data quirks |

> Only a subset is required for the first event-study; we will fill what we can, document gaps, and expand.
