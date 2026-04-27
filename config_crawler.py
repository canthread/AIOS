"""
config_crawler.py
-----------------
Ingestion layer — config file crawler.

Crawls ~/docker/ for docker-compose files and optional nginx/app configs.
Produces structured documents ready for embedding into ChromaDB alongside
the Docker socket state from docker_crawler.py.

Convention (enforced, not discovered):
  ~/docker/<stack_name>/docker-compose.yml   — required
  ~/docker/<stack_name>/*.conf               — optional, crawled if present
  ~/docker/<stack_name>/*.env                — optional, crawled if present
                                               (secrets stripped)

OS framing: the Docker socket crawler reads runtime state (what IS running).
This crawler reads declared state (what SHOULD be running). Having both in
ChromaDB lets the planner reason about drift between intent and reality.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOCKER_DIR = Path.home() / "docker"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ComposeServiceDocument:
    """
    One document per service defined in a docker-compose file.
    Granular by service — not by file — so retrieval for "what does radarr
    need" returns the radarr block, not the entire compose file.
    """
    stack_name: str           # parent directory name e.g. "mediastack"
    service_name: str         # service key in the compose file e.g. "radarr"
    image: Optional[str]
    ports: list[str]          # ["7878:7878"]
    volumes: list[str]        # ["/data/media:/movies"]
    networks: list[str]       # ["mediastack"]
    environment: dict         # env vars (secrets stripped)
    depends_on: list[str]     # other services this one depends on
    restart_policy: Optional[str]
    source_file: str          # absolute path to the compose file

    def to_text(self) -> str:
        lines = [
            f"Compose service: {self.service_name}",
            f"Stack: {self.stack_name}",
            f"Source: {self.source_file}",
        ]
        if self.image:
            lines.append(f"Image: {self.image}")
        if self.ports:
            lines.append(f"Ports: {', '.join(self.ports)}")
        if self.volumes:
            lines.append(f"Volumes: {', '.join(self.volumes)}")
        if self.networks:
            lines.append(f"Networks: {', '.join(self.networks)}")
        if self.environment:
            env_str = ", ".join(f"{k}={v}" for k, v in self.environment.items())
            lines.append(f"Environment: {env_str}")
        if self.depends_on:
            lines.append(f"Depends on: {', '.join(self.depends_on)}")
        if self.restart_policy:
            lines.append(f"Restart: {self.restart_policy}")
        return "\n".join(lines)


@dataclass
class ComposeNetworkDocument:
    """One document per network declared in a compose file."""
    stack_name: str
    network_name: str
    driver: Optional[str]
    external: bool            # True if it references an existing network
    source_file: str

    def to_text(self) -> str:
        lines = [
            f"Compose network: {self.network_name}",
            f"Stack: {self.stack_name}",
            f"Driver: {self.driver or 'bridge'}",
            f"External: {self.external}",
            f"Source: {self.source_file}",
        ]
        return "\n".join(lines)


@dataclass
class RawConfigDocument:
    """
    One document per non-compose config file (nginx.conf, app.conf, etc.)
    Stored as raw text — the LLM can read it as-is.
    """
    stack_name: str
    filename: str
    content: str              # full file content (secrets stripped)
    source_file: str

    def to_text(self) -> str:
        return "\n".join([
            f"Config file: {self.filename}",
            f"Stack: {self.stack_name}",
            f"Source: {self.source_file}",
            "---",
            self.content,
        ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SENSITIVE_KEYS = {"PASSWORD", "SECRET", "KEY", "TOKEN", "API", "PASS", "CREDENTIAL"}

def _strip_sensitive(env: dict) -> dict:
    """Remove env vars whose names contain sensitive keywords."""
    return {
        k: v for k, v in env.items()
        if not any(s in str(k).upper() for s in SENSITIVE_KEYS)
    }

def _parse_env(raw_env) -> dict:
    """
    Compose env can be a list ["KEY=VALUE", "FLAG"] or a dict {"KEY": "VALUE"}.
    Normalize to dict either way.
    """
    if not raw_env:
        return {}
    if isinstance(raw_env, dict):
        return {str(k): str(v) for k, v in raw_env.items()}
    if isinstance(raw_env, list):
        result = {}
        for entry in raw_env:
            if "=" in str(entry):
                k, _, v = str(entry).partition("=")
                result[k] = v
            else:
                result[str(entry)] = ""
        return result
    return {}

def _parse_ports(raw_ports) -> list[str]:
    """Normalize port entries to strings."""
    if not raw_ports:
        return []
    return [str(p) for p in raw_ports]

def _parse_volumes(raw_volumes) -> list[str]:
    """
    Volumes can be short syntax strings or long-form dicts.
    Normalize to "source:target" strings.
    """
    if not raw_volumes:
        return []
    result = []
    for v in raw_volumes:
        if isinstance(v, str):
            result.append(v)
        elif isinstance(v, dict):
            source = v.get("source", "")
            target = v.get("target", "")
            result.append(f"{source}:{target}")
    return result

def _parse_depends_on(raw) -> list[str]:
    """depends_on can be a list or a dict with condition keys."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return list(raw.keys())
    return []

def _parse_networks(raw) -> list[str]:
    """Networks can be a list or a dict."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return list(raw.keys())
    return []

def _strip_secrets_from_text(content: str) -> str:
    """
    Best-effort secret stripping from raw config files.
    Replaces lines containing sensitive keywords with a placeholder.
    """
    clean_lines = []
    for line in content.splitlines():
        upper = line.upper()
        if any(s in upper for s in SENSITIVE_KEYS):
            # Keep the key name, redact the value
            if "=" in line:
                key, _, _ = line.partition("=")
                clean_lines.append(f"{key}=[REDACTED]")
            else:
                clean_lines.append("[REDACTED LINE]")
        else:
            clean_lines.append(line)
    return "\n".join(clean_lines)


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

class ConfigCrawler:
    """
    Crawls ~/docker/ for compose files and config files.

    Convention:
      ~/docker/<stack>/docker-compose.yml   → parsed into service documents
      ~/docker/<stack>/*.conf               → stored as raw config documents
      ~/docker/<stack>/*.env                → parsed as env, secrets stripped

    One level deep only — no recursion into subdirectories of stacks.
    """

    def __init__(self, docker_dir: Path = DOCKER_DIR):
        self.docker_dir = docker_dir

    def _crawl_compose(
        self,
        stack_name: str,
        compose_path: Path,
    ) -> tuple[list[ComposeServiceDocument], list[ComposeNetworkDocument]]:
        """Parse a single docker-compose.yml into service and network documents."""
        service_docs = []
        network_docs = []

        with open(compose_path) as f:
            data = yaml.safe_load(f)

        if not data:
            return service_docs, network_docs

        services = data.get("services") or {}
        for svc_name, svc_config in services.items():
            if not svc_config:
                svc_config = {}

            env = _parse_env(svc_config.get("environment"))
            env = _strip_sensitive(env)

            # restart policy can be a string or a dict (long form)
            restart = svc_config.get("restart")
            if isinstance(restart, dict):
                restart = restart.get("condition")

            service_docs.append(ComposeServiceDocument(
                stack_name=stack_name,
                service_name=svc_name,
                image=svc_config.get("image"),
                ports=_parse_ports(svc_config.get("ports")),
                volumes=_parse_volumes(svc_config.get("volumes")),
                networks=_parse_networks(svc_config.get("networks")),
                environment=env,
                depends_on=_parse_depends_on(svc_config.get("depends_on")),
                restart_policy=restart,
                source_file=str(compose_path),
            ))

        # Top-level networks block
        networks = data.get("networks") or {}
        for net_name, net_config in networks.items():
            if not net_config:
                net_config = {}
            external = False
            if isinstance(net_config.get("external"), bool):
                external = net_config["external"]
            elif isinstance(net_config.get("external"), dict):
                external = True  # old compose syntax: external: {name: foo}

            network_docs.append(ComposeNetworkDocument(
                stack_name=stack_name,
                network_name=net_name,
                driver=net_config.get("driver"),
                external=external,
                source_file=str(compose_path),
            ))

        return service_docs, network_docs

    def _crawl_conf_files(
        self,
        stack_name: str,
        stack_dir: Path,
    ) -> list[RawConfigDocument]:
        """Read *.conf and *.env files alongside the compose file."""
        docs = []

        for ext in ("*.conf", "*.env"):
            for conf_path in sorted(stack_dir.glob(ext)):
                try:
                    content = conf_path.read_text(encoding="utf-8", errors="replace")
                    content = _strip_secrets_from_text(content)
                    docs.append(RawConfigDocument(
                        stack_name=stack_name,
                        filename=conf_path.name,
                        content=content,
                        source_file=str(conf_path),
                    ))
                except Exception as e:
                    print(f"[config_crawler] Warning: could not read {conf_path}: {e}")

        return docs

    def crawl(self) -> dict[str, list]:
        """
        Walk ~/docker/ one level deep. For each subdirectory that contains
        a docker-compose.yml, crawl it.

        Returns:
            {
                "services":  list[ComposeServiceDocument],
                "networks":  list[ComposeNetworkDocument],
                "configs":   list[RawConfigDocument],
            }
        """
        all_services = []
        all_networks = []
        all_configs = []

        if not self.docker_dir.exists():
            print(f"[config_crawler] {self.docker_dir} does not exist. "
                  f"Create it and add your stack directories.")
            return {"services": [], "networks": [], "configs": []}

        # One level deep only
        for stack_dir in sorted(self.docker_dir.iterdir()):
            if not stack_dir.is_dir():
                continue

            compose_path = stack_dir / "docker-compose.yml"
            if not compose_path.exists():
                compose_path = stack_dir / "docker-compose.yaml"
            if not compose_path.exists():
                print(f"[config_crawler] Skipping {stack_dir.name} — no compose file found.")
                continue

            stack_name = stack_dir.name
            print(f"[config_crawler] Crawling stack: {stack_name}")

            try:
                services, networks = self._crawl_compose(stack_name, compose_path)
                all_services.extend(services)
                all_networks.extend(networks)
            except yaml.YAMLError as e:
                print(f"[config_crawler] Warning: could not parse {compose_path}: {e}")
                continue

            configs = self._crawl_conf_files(stack_name, stack_dir)
            all_configs.extend(configs)

        return {
            "services": all_services,
            "networks": all_networks,
            "configs": all_configs,
        }


# ---------------------------------------------------------------------------
# Entry point — run directly to inspect output before embedding
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    crawler = ConfigCrawler()

    print("=" * 60)
    print(f"CONFIG CRAWLER — {crawler.docker_dir}")
    print("=" * 60)

    results = crawler.crawl()

    print(f"\n[COMPOSE SERVICES] ({len(results['services'])} found)\n")
    for doc in results["services"]:
        print(doc.to_text())
        print("-" * 40)

    print(f"\n[COMPOSE NETWORKS] ({len(results['networks'])} found)\n")
    for doc in results["networks"]:
        print(doc.to_text())
        print("-" * 40)

    print(f"\n[RAW CONFIGS] ({len(results['configs'])} found)\n")
    for doc in results["configs"]:
        # Print just the header, not full content, to keep output readable
        lines = doc.to_text().splitlines()
        print("\n".join(lines[:4]) + "\n[...content truncated for display...]")
        print("-" * 40)

    total = len(results["services"]) + len(results["networks"]) + len(results["configs"])
    print(f"\nTotal documents ready for embedding: {total}")
