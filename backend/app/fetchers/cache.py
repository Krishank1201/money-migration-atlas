import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

from app.core.schemas import Chain, FetchResult
from app.config import get_settings

logger = logging.getLogger(__name__)


class FileCache:
    """
    File-based local JSON cache keyed by SHA256(chain:address).
    Prevents redundant external API calls and rate-limit burn.
    """

    def __init__(self, cache_dir: Optional[str] = None, ttl_hours: Optional[int] = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR)
        self.ttl_seconds = (ttl_hours or settings.CACHE_TTL_HOURS) * 3600
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self, address: str, chain: Chain) -> str:
        raw = f"{chain.value.lower()}:{address.lower()}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _get_file_path(self, address: str, chain: Chain) -> Path:
        key = self._get_key(address, chain)
        return self.cache_dir / f"{key}.json"

    def get(self, address: str, chain: Chain) -> Optional[FetchResult]:
        file_path = self._get_file_path(address, chain)
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cached_at = data.get("cached_at", 0)
            now = int(time.time())
            if now - cached_at > self.ttl_seconds:
                logger.debug("Cache expired for %s:%s", chain.value, address)
                return None

            result = FetchResult.model_validate(data)
            result.data_source = "cache"
            return result
        except Exception as e:
            logger.warning("Failed reading cache file %s: %s", file_path, e)
            return None

    def set(self, address: str, chain: Chain, result: FetchResult) -> None:
        file_path = self._get_file_path(address, chain)
        try:
            data = result.model_dump()
            data["cached_at"] = int(time.time())
            data["data_source"] = "cache"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug("Cached %s:%s to %s", chain.value, address, file_path.name)
        except Exception as e:
            logger.warning("Failed writing cache file %s: %s", file_path, e)

    def clear(self) -> None:
        for p in self.cache_dir.glob("*.json"):
            try:
                p.unlink()
            except Exception:
                pass
