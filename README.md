# SmartEVSim

SmartEVSim is a Python simulation framework for testing control strategies for
coordinated electric-vehicle charging. It models a point of common coupling,
central and local controllers, chargers, and EVs in a deterministic,
time-stepped simulation.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) (recommended) or another Python package manager
- A valid Gurobi installation and license when using optimization-based controllers

## Installation

Clone the repository and install the project with its development dependencies:

```bash
uv sync --extra dev
```

To install only the runtime dependencies, use `uv sync`.

## Running a simulation

The included runner loads `centralized_config.yaml`, executes the simulation,
saves its records as CSV, and opens an interactive Plotly figure:

```bash
uv run python run.py
```

Results are written to `results/centralized_config.csv`. Existing files are
preserved by adding a numeric suffix to the new filename.

To use another scenario, change `config_path` in `run.py`, or run one directly
through the Python API:

```python
from smartevsim.engine_api import build_engine_from_yaml

engine = build_engine_from_yaml("centralized_config.yaml")
engine.run()
engine.save_records_to_csv("results/simulation.csv")
engine.plot_records()
```

## Configuration

Simulation scenarios are defined in YAML. The main sections configure:

- `sim_config`: time step, simulation horizon, sampling, and random seed
- `priority_config`: controller responsibility and EV prioritization
- `pcc`: point-of-common-coupling timing and communication delay
- `central_controller`: cluster-level control strategy
- `micro_controllers`: per-EV control strategy
- `chargers`: charger model and electrical constraints
- `evs`: arrival, departure, requested energy, and vehicle model

EV and charger specifications are stored under `src/smartevsim/config/`.
Explicit values in a scenario override values loaded from these specification
files. A scenario can also use an `ev_generation` section to generate a seeded,
reproducible population instead of listing every EV manually.

Two example scenarios are included:

- `centralized_config.yaml` uses centralized optimization and centralized micro-controllers.
- `default_config.yaml` demonstrates a locally controlled scenario.

## Project structure

```text
SmartEVSim/
├── run.py                         # Example simulation entry point
├── centralized_config.yaml        # Centralized example scenario
├── default_config.yaml            # Local-control example scenario
├── pyproject.toml                 # Package and tool configuration
└── src/smartevsim/
    ├── engine.py                  # Simulation loop and result recording
    ├── engine_api.py              # YAML-based engine construction
    ├── scenario_generator.py      # Reproducible EV scenario generation
    ├── analysis/                  # Metrics and visualization helpers
    ├── config/                    # EV and charger specifications
    ├── units/                     # Controllers, chargers, EVs, PCC, and server
    └── utils/                     # Shared data types and utilities
```

## Development

After installing the development dependencies, run the quality checks with:

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

## License

This project is distributed under the terms in [LICENSE](LICENSE).
