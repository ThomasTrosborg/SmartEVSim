import logging
from pathlib import Path

from smartevsim.engine_api import build_engine_from_yaml
from smartevsim.utils.logging import setup_logging


if __name__ == "__main__":
    # config_path = "config/distributed_config.yaml"
    config_path = "config/centralized_config.yaml"

    name = Path(config_path).stem
    records_path = f"results/{name}.csv"

    setup_logging()
    engine = build_engine_from_yaml(config_path)
    engine.run()
    engine.save_records_to_csv(records_path)
    engine.plot_records()
