from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def default_data_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "VistaraVision" / "v1"
    return Path.home() / ".vistara_vision" / "v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISTARA_", env_file=".env", extra="ignore")

    app_name: str = "Vistara Vision"
    api_version: str = "v1"
    data_dir: Path = default_data_dir()
    db_filename: str = "vistara_memory.sqlite3"
    pairing_ttl_seconds: int = 180
    cloud_vision_enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765

    @property
    def db_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / self.db_filename
