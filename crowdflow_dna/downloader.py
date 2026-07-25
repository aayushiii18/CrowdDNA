"""Model artifact download helper for CrowdFlow DNA.

Provides an automatic download mechanism to fetch the canonical deployment.pt
from the official GitHub v1.0.0 release if no local model is available.
"""

import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CANONICAL_MODEL_URL = "https://github.com/aayushiii18/CrowdDNA/releases/download/v1.0.0/deployment.pt"
_CACHE_DIR = Path(".cache/crowdflow_dna")
_CACHE_PATH = _CACHE_DIR / "deployment.pt"


def resolve_model_path(env_path: Optional[str]) -> Optional[str]:
    """Resolve the path to the deployment.pt model artifact.

    Resolution order:
    1. If `env_path` is provided and the file exists, return `env_path`.
    2. If a locally cached version exists, return the cache path.
    3. Otherwise, attempt to download from the canonical GitHub release URL
       into the local cache. If successful, return the cache path.
    4. If download fails, return None (triggering dummy mode fallback).

    Args:
        env_path: The value of the CROWDDNA_MODEL_PATH environment variable,
            or None if unset.

    Returns:
        Absolute or relative path to a valid deployment.pt file, or None
        if resolution/download fails.
    """
    # 1. Existing local override
    if env_path is not None:
        if os.path.exists(env_path):
            logger.info("Using configured model path: %s", env_path)
            return env_path
        else:
            logger.warning(
                "Configured CROWDDNA_MODEL_PATH does not exist: %s", env_path
            )

    # 2. Existing cached artifact
    if _CACHE_PATH.exists():
        logger.info("Using cached deployment artifact: %s", _CACHE_PATH)
        return str(_CACHE_PATH)

    # 3. Download from canonical source
    logger.info("No local model found. Attempting to download from GitHub Releases...")
    logger.info("URL: %s", _CANONICAL_MODEL_URL)
    
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # We use a temporary download path to avoid corrupting the cache on failure.
        temp_download_path = _CACHE_PATH.with_suffix(".tmp")
        
        # nosec: Hardcoded HTTPS URL to exact release asset, no arbitrary input
        urllib.request.urlretrieve(_CANONICAL_MODEL_URL, temp_download_path)
        
        # Atomic-like rename on POSIX; handles overwrite safely
        temp_download_path.replace(_CACHE_PATH)
        
        logger.info("Successfully downloaded deployment artifact to %s", _CACHE_PATH)
        return str(_CACHE_PATH)
    except urllib.error.URLError as exc:
        logger.warning(
            "Failed to download deployment.pt. The application will fall back "
            "to dummy mode. Reason: %s",
            exc,
        )
        return None
    except Exception as exc:
        logger.error(
            "Unexpected error downloading deployment.pt: %s", exc
        )
        return None
