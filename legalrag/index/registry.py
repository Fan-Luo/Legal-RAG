from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class IndexRegistry:
    """
    Minimal index registry:
    - data/index/ACTIVE stores active version name
    - data/index/versions/<version>/ contains index files
    - fallback to data/index when ACTIVE is missing or invalid
    """

    index_root: Path

    def versions_dir(self) -> Path:
        return self.index_root / "versions"

    def active_version(self) -> Optional[str]:
        active = self.index_root / "ACTIVE"
        if not active.exists():
            return None
        value = active.read_text(encoding="utf-8").strip()
        return value or None

    def active_index_dir(self) -> Path:
        version = self.active_version()
        if not version:
            return self.index_root
        candidate = self.versions_dir() / version
        return candidate if candidate.exists() else self.index_root

    def list_versions(self) -> List[str]:
        vdir = self.versions_dir()
        if not vdir.exists():
            return []
        return sorted([p.name for p in vdir.iterdir() if p.is_dir()])

    def activate(self, version: str) -> Path:
        vdir = self.versions_dir() / version
        if not vdir.exists():
            raise FileNotFoundError(f"index version not found: {version}")
        active = self.index_root / "ACTIVE"
        active.write_text(version, encoding="utf-8")
        return vdir

    def ensure_version_dir(self, version: str) -> Path:
        vdir = self.versions_dir() / version
        vdir.mkdir(parents=True, exist_ok=True)
        return vdir
