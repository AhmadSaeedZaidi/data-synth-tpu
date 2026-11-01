"""Read-only installation inventory and opt-in BeamNGpy connection probe.

No simulator launch, control, sensors, label extraction, or license bypass.
"""

import json
import os
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path, PureWindowsPath


class BeamNGError(RuntimeError):
    """Safe simulator diagnostic failure."""


def host_path(home: str) -> Path:
    """Resolve a configured Windows path on Windows or its default WSL mount."""
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", home):
        kernel = Path("/proc/sys/kernel/osrelease")
        if not kernel.exists() or "microsoft" not in kernel.read_text().lower():
            raise BeamNGError("Windows installation paths require Windows or WSL")
        path = PureWindowsPath(home)
        return Path("/mnt") / path.drive[0].lower() / Path(*path.parts[1:])
    return Path(home).expanduser()


def inspect_installation(home: str, edition: str) -> dict:
    """Report files/source markers as observations, never live capability proof."""
    if edition not in {"drive", "tech"}:
        raise BeamNGError("edition must be drive or tech")
    root = host_path(home)
    binary = root / "Bin64" / f"BeamNG.{edition}.x64.exe"
    source_path = root / "lua/ge/extensions/tech/techCore.lua"
    source = source_path.read_text(encoding="utf-8") if source_path.is_file() else ""
    camera = re.search(r"M\.handleOpenCamera\s*=\s*function\(request\)(.{0,500})", source, re.S)
    protocol_path = root / "lua/common/tech/techCommunication.lua"
    protocol_source = protocol_path.read_text() if protocol_path.is_file() else ""
    protocol = re.search(r"M\.protocolVersion\s*=\s*['\"]([^'\"]+)", protocol_source)
    sdk_protocol = None
    sdk_version = None
    try:
        sdk = distribution("beamngpy")
        sdk_version = sdk.version
        sdk_source = sdk.locate_file("beamngpy/connection/connection.py").read_text()
        match = re.search(r"PROTOCOL_VERSION\s*=\s*['\"]([^'\"]+)", sdk_source)
        sdk_protocol = match.group(1) if match else None
    except (PackageNotFoundError, OSError):
        pass
    installed_protocol = protocol.group(1) if protocol else None
    return {
        "configured_edition": edition,
        "installation_found": binary.is_file(),
        "resolved_home": str(root.resolve()),
        "tech_core_source_present": bool(source),
        "installed_protocol": installed_protocol,
        "beamngpy_version": sdk_version,
        "sdk_protocol": sdk_protocol,
        "protocol_match": (
            installed_protocol == sdk_protocol if installed_protocol and sdk_protocol else None
        ),
        "connection_handler_present": "M.handleHello" in source,
        "system_info_handler_present": "M.handleGetSystemInfo" in source,
        "camera_license_gate_observed": (
            "ResearchVerifier.isTechLicenseVerified()" in camera.group(1) if camera else None
        ),
        "live_connection_tested": False,
        "tech_enabled": None,
        "camera_capture_tested": False,
    }


def _probe_sdk(host: str, port: int, factory=None) -> dict:
    """Worker body; the parent process enforces an overall deadline.

    An already-running simulator must expose its documented techCore endpoint.
    Source compatibility with Drive is not a guarantee of successful connection.
    """
    client = None
    try:
        if factory is None:
            from beamngpy import BeamNGpy

            factory = BeamNGpy
        client = factory(host, port, quit_on_close=False)
        client.open(launch=False)
        return {
            "connected": True,
            "beamngpy_version": version("beamngpy"),
            "host_os": client.host_os(),
            "tech_enabled": client.tech_enabled(),
            "camera_capture_tested": False,
            "telemetry_capture_tested": False,
        }
    except Exception:
        raise BeamNGError(
            "BeamNG probe failed; check SDK, endpoint, and simulator console"
        ) from None
    finally:
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                pass


def probe_connection(host: str, port: int, *, timeout: float = 5.0) -> dict:
    """Bound the older SDK's blocking socket operations in a separate process."""
    if not host.strip() or not 1 <= port <= 65535 or not 0 < timeout <= 30:
        raise BeamNGError("invalid connection settings")
    try:
        completed = subprocess.run(
            [sys.executable, "-m", __name__, host, str(port)],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        return json.loads(completed.stdout)
    except Exception:
        raise BeamNGError(
            "BeamNG probe failed or timed out; check endpoint and SDK compatibility"
        ) from None


if __name__ == "__main__":
    try:
        print(json.dumps(_probe_sdk(sys.argv[1], int(sys.argv[2]))))
    except Exception:
        raise SystemExit(1) from None
