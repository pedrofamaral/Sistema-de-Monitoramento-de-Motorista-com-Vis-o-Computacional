"""Leitura do arquivo de configuração (config/settings.yaml)."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str | Path) -> Path:
    """Converte um caminho do settings.yaml (relativo à raiz) em caminho absoluto."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path
