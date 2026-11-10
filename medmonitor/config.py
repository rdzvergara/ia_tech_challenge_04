"""Carregamento de configuração (YAML + variáveis de ambiente)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=4)
def load_settings(path: str | None = None) -> dict[str, Any]:
    if load_dotenv:
        load_dotenv(ROOT / ".env")
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(p: str | Path) -> Path:
    """Resolve caminho relativo à raiz do projeto."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def env(name: str, default: str | None = None) -> str | None:
    if load_dotenv:
        load_dotenv(ROOT / ".env")
    val = os.getenv(name, default)
    return val if val not in ("", None) else default


def offline_forced() -> bool:
    return (env("MEDMONITOR_OFFLINE", "false") or "false").lower() in ("1", "true", "yes")
