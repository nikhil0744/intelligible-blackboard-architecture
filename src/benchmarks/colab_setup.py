"""Rerunnable Colab setup: pinned user-space Ollama; no systemd, tunnels or paid APIs."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from llm_broker.backends.ollama import OllamaBackend
from llm_broker.preflight import hardware_report
from llm_broker.types import LLMError

from .preparation import digest, resources

_OWNED: dict[str, subprocess.Popen] = {}


def install_ollama(directory: Path, progress=print):
    spec = resources()["runtime"]
    root = Path(directory) / spec["ollama_version"]
    binary = root / "bin/ollama"
    if root.exists():
        ready = json.loads((root / "ready.json").read_text())
        if ready["archive_sha256"] != spec["ollama_archive_sha256"] or digest(binary) != ready["binary_sha256"]:
            raise ValueError(f"Invalid Ollama install at {root}; use a fresh setup directory")
        return binary
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise ValueError("The pinned Colab installer supports Linux x86_64 only")
    import zstandard  # optional small wheel installed by .[colab]
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ollama-install-", dir=root.parent) as temporary:
        temporary = Path(temporary)
        archive = temporary / "ollama.tar.zst"
        checksum, received = hashlib.sha256(), 0
        progress(f"Downloading pinned Ollama {spec['ollama_version']} ({spec['ollama_archive_bytes'] / 1024**3:.2f} GiB)")
        with urllib.request.urlopen(spec["ollama_archive_url"], timeout=60) as response, archive.open("xb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                checksum.update(chunk)
                received += len(chunk)
                if received > spec["ollama_archive_bytes"]:
                    raise ValueError("Ollama download exceeds pinned size")
                if received // (64 * 1024**2) != (received - len(chunk)) // (64 * 1024**2):
                    progress(f"  {received // 1024**2} MiB downloaded")
        if received != spec["ollama_archive_bytes"] or checksum.hexdigest() != spec["ollama_archive_sha256"]:
            raise ValueError("Ollama archive verification failed; incomplete install was discarded")
        staged = temporary / "runtime"
        staged.mkdir()
        with archive.open("rb") as stream, zstandard.ZstdDecompressor().stream_reader(stream) as decoded:
            with tarfile.open(fileobj=decoded, mode="r|") as bundle:
                bundle.extractall(staged, filter="data")
        candidate = staged / "bin/ollama"
        if not candidate.is_file():
            raise ValueError("Pinned archive contains no Ollama binary")
        (staged / "ready.json").write_text(json.dumps({"archive_sha256": checksum.hexdigest(), "binary_sha256": digest(candidate)}))
        os.rename(staged, root)
    return binary


@dataclass
class LocalService:
    """Only processes created by this setup are eligible for stop()."""
    process: subprocess.Popen | None
    binary: Path
    host: str
    reused: bool

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


def ensure_service(directory=Path("/content/blackboard-runtime"), *, cpu_smoke=False, progress=print):
    """Called explicitly by notebook setup; never from inference or benchmark solving."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    hardware = hardware_report()
    if not cpu_smoke and not hardware["gpus"]:
        raise ValueError("Select a GPU in Runtime > Change runtime type, reconnect, then run setup again")
    spec = resources()["runtime"]
    install_dir = directory / "ollama" / spec["ollama_version"]
    if not install_dir.exists() and hardware["disk_free_mb"] < 18 * 1024:
        raise ValueError("Initial setup needs 18 GiB of local free disk for download and extraction")
    binary = install_ollama(directory / "ollama", progress)
    host = "http://127.0.0.1:11434"
    backend = OllamaBackend(host=host)
    process = None
    try:
        version = backend.version()
        reused = True
    except LLMError:
        environment = {**os.environ, "OLLAMA_HOST": "127.0.0.1:11434",
                       "OLLAMA_MODELS": str(directory / "models"), "OLLAMA_NUM_PARALLEL": "1",
                       "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_NO_CLOUD": "1"}
        with (directory / "ollama.log").open("ab") as log:
            process = subprocess.Popen([str(binary), "serve"], env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        service = LocalService(process, binary, host, False)
        try:
            deadline = time.monotonic() + 90
            while True:
                if process.poll() is not None:
                    raise ValueError(f"Ollama exited; inspect {directory / 'ollama.log'}")
                try:
                    version = backend.version()
                    break
                except LLMError:
                    if time.monotonic() >= deadline:
                        raise ValueError(f"Ollama did not become ready; inspect {directory / 'ollama.log'}")
                    time.sleep(1)
        except BaseException:
            service.stop()
            raise
        reused = False
    service = LocalService(process, binary, host, reused)
    if process is not None:
        _OWNED[host] = process
    elif host in _OWNED and _OWNED[host].poll() is None:
        service.process = _OWNED[host]
    try:
        if version != spec["ollama_version"]:
            raise ValueError(f"Port 11434 is running Ollama {version}; the notebook requires {spec['ollama_version']}. Stop the other service before setup.")
        entries = {entry["name"]: entry for entry in backend.model_entries()}
        if spec["model"] not in entries:
            if hardware["disk_free_mb"] * 1024**2 < spec["model_download_bytes"]:
                raise ValueError("Insufficient local disk space for the pinned model download")
            if hardware["memory"] and hardware["memory"]["available_mb"] < 6 * 1024:
                raise ValueError("Less than 6 GiB of host RAM is available for initial model setup")
            progress(f"Pulling {spec['model']} (~{spec['model_download_bytes'] / 1024**3:.1f} GiB). Model files stay on local runtime disk.")
            subprocess.run([str(binary), "pull", spec["model"]],
                           env={**os.environ, "OLLAMA_HOST": host}, timeout=1800, check=True)
            entries = {entry["name"]: entry for entry in backend.model_entries()}
        actual = entries[spec["model"]].get("digest", "").removeprefix("sha256:")
        if actual != spec["model_digest"]:
            raise ValueError("The requested model tag has a different digest from the pinned preset. Use the original model or start a separately validated preset; no substitution was made.")
        progress(f"Ollama {version} ready; model digest {actual}; reused service={reused}")
        return service
    except BaseException:
        if not reused:
            service.stop()
        raise
