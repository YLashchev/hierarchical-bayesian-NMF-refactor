from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from bayesian_panel_nmf.config import Config

CONFIG_DIR = Path(__file__).parent.parent / "configs"


@pytest.mark.parametrize("config_path", CONFIG_DIR.glob("*.yaml"))
def test_shipped_configs_pass_validation(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # The config files might be just snippets, but they should ideally pass
    # our validation if they have 'data' and 'schema'.
    if "data" in config and "schema" in config["data"]:
        Config.model_validate(config)
