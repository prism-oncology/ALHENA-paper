import os
from pathlib import Path
from typing import Any, Mapping, Optional
import numpy as np

from .common import flatten_dict, mkdir, save_json


def _to_builtin(x: Any) -> Any:
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    return x


class WandbRunHandle:
    """
    Optional wandb wrapper (no-op if wandb is disabled or unavailable).
    Writes wandb_manifest.json in run_dir and converts logged values to builtin types.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        mode: str,
        run_dir: Path,
        project: str,
        run_name: str,
        group: str,
        job_type: str,
        config: Mapping[str, Any],
        tags: Optional[list[str]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = str(mode)
        self.run_dir = mkdir(run_dir)
        self.project = str(project)
        self.run_name = str(run_name)
        self.group = str(group)
        self.job_type = str(job_type)
        self.tags = list(tags or [])
        self._run = None
        self._wandb = None
        self._status = "disabled"
        self._message = "wandb disabled"

        manifest = {
            "enabled_requested": self.enabled,
            "mode": self.mode,
            "project": self.project,
            "run_name": self.run_name,
            "group": self.group,
            "job_type": self.job_type,
            "tags": self.tags,
        }

        if not self.enabled:
            save_json(self.run_dir / "wandb_manifest.json", {**manifest, "status": self._status, "message": self._message})
            return

        try:
            import wandb  # type: ignore

            os.environ.setdefault("WANDB_MODE", self.mode)
            os.environ.setdefault("WANDB_DIR", str(self.run_dir))
            os.environ.setdefault("WANDB_SILENT", "true")
            self._wandb = wandb
            cfg_flat = flatten_dict(_to_builtin(dict(config)))
            self._run = wandb.init(
                project=self.project,
                name=self.run_name,
                group=self.group,
                job_type=self.job_type,
                dir=str(self.run_dir),
                mode=self.mode,
                config=cfg_flat,
                tags=self.tags,
                reinit=True,
            )
            self._status = "active"
            self._message = "wandb initialized"
        except Exception as e:  # pragma: no cover - depends on local env
            self._run = None
            self._wandb = None
            self._status = "unavailable"
            self._message = f"wandb import/init failed: {e}"
            with open(self.run_dir / "wandb_unavailable.txt", "w", encoding="utf-8") as f:
                f.write(self._message + "\n")

        save_json(self.run_dir / "wandb_manifest.json", {**manifest, "status": self._status, "message": self._message})

    @property
    def active(self) -> bool:
        return self._run is not None

    def log(self, payload: Mapping[str, Any], *, step: Optional[int] = None) -> None:
        if self._run is None:
            return
        clean = {str(k): _to_builtin(v) for k, v in payload.items()}
        if not clean:
            return
        if step is None:
            self._run.log(clean)
        else:
            self._run.log(clean, step=int(step))

    def summary_update(self, payload: Mapping[str, Any]) -> None:
        if self._run is None:
            return
        for k, v in payload.items():
            self._run.summary[str(k)] = _to_builtin(v)

    def save_artifact_manifest(self, name: str, payload: Mapping[str, Any]) -> None:
        save_json(self.run_dir / f"{name}.json", _to_builtin(dict(payload)))

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        finally:
            self._run = None
