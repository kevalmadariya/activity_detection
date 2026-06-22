"""
pipeline.config — Central configuration loader.

Loads camera and pipeline settings from ``.env`` with sensible defaults.
Follows the Single Responsibility Principle — *only* handles config loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from pipeline.models import CameraConfig

# ---------------------------------------------------------------------------
# Bootstrap: load .env from project root (idempotent)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH     = _PROJECT_ROOT / ".env"

load_dotenv(_ENV_PATH)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_camera_config(
    camera_id: str = "CAMERA_URL_1",
    username:  Optional[str] = None,
    password:  Optional[str] = None,
) -> CameraConfig:
    """
    Build a :class:`CameraConfig` from environment variables.

    Looks up ``<camera_id>`` in the environment for the base URL,
    and ``CAMERA_USER`` / ``CAMERA_PASS`` for credentials.
    Explicit ``username`` / ``password`` kwargs override env values.

    Args:
        camera_id: Environment variable name holding ``host:port``.
        username:  Override for ``CAMERA_USER``.
        password:  Override for ``CAMERA_PASS``.

    Raises:
        EnvironmentError: If required variables are missing.
    """
    base_url = os.getenv(camera_id)
    if not base_url:
        raise EnvironmentError(
            f"Environment variable '{camera_id}' is not set.  "
            f"Add it to {_ENV_PATH}"
        )

    user = username or os.getenv("CAMERA_USER", "")
    pwd  = password or os.getenv("CAMERA_PASS", "")

    return CameraConfig(
        camera_id=camera_id,
        base_url=base_url,
        username=user,
        password=pwd,
    )


def get_vision_llm_config() -> tuple[str, str, list[str]]:
    """
    Returns (api_key, model_name, activity_classes) strictly from .env.
    """
    from dotenv import dotenv_values
    env_vars = dotenv_values(_ENV_PATH)
    
    api_key = env_vars.get("NVIDIA_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY is not set in .env")
        
    model = env_vars.get("VISION_LLM_MODEL") or os.getenv("VISION_LLM_MODEL", "google/paligemma-3b-mix-448")
    
    classes_str = env_vars.get("ACTIVITY_CLASSES", "")
    classes = [c.strip().lower() for c in classes_str.split(",") if c.strip()]
    if not classes:
        classes = ["using_computer", "working_with_machine", "using_phone"]
        
    return api_key, model, classes


def get_temp_dir() -> Path:
    """Return the project-level ``temp/`` directory, creating it if needed."""
    temp = _PROJECT_ROOT / "temp"
    temp.mkdir(parents=True, exist_ok=True)
    return temp
