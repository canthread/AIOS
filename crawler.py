"""
docker_crawler.py
-----------------
Ingestion layer — Docker socket crawler.

Connects to the local Docker socket, reads running stack state, and serializes
it into structured text documents ready for chunking and embedding into ChromaDB.

OS framing: this is the kernel reading hardware state — it produces the ground
truth that everything else is built on. If this is wrong, everything downstream
is wrong.
"""

import json
import docker
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ContainerDocument:
    """
    One document per container. This is the unit that gets chunked and embedded.
    Kept as a dataclass so it's easy to serialize to dict or plain text.
    """
    container_id: str
    name: str
    image: str
    status: str
    ports: dict
    networks: list[str]
    volumes: list[str]
    environment: dict
    labels: dict
    compose_project: Optional[str]  # set if container was started via compose
    compose_service: Optional[str]

    def to_text(self) -> str:
        """
        Serialize to plain text. This is what gets embedded into ChromaDB.
        Format is deliberately readable — the LLM will see this as retrieved
        context and needs to parse it reliably.
        """
        lines = [
            f"Container: {self.name}",
            f"ID: {self.container_id[:12]}",
            f"Image: {self.image}",
            f"Status: {self.status}",
        ]

        if self.ports:
            port_strs = []
            for container_port, host_bindings in self.ports.items():
                if host_bindings:
                    for binding in host_bindings:
                        port_strs.append(
                            f"{binding['HostIp'] or '0.0.0.0'}:{binding['HostPort']}->{container_port}"
                        )
                else:
                    port_strs.append(f"(unexposed) {container_port}")
            lines.append(f"Ports: {', '.join(port_strs)}")
        else:
            lines.append("Ports: none")

        if self.networks:
            lines.append(f"Networks: {', '.join(self.networks)}")
        else:
            lines.append("Networks: none")

        if self.volumes:
            lines.append(f"Volumes: {', '.join(self.volumes)}")
        else:
            lines.append("Volumes: none")

        # Only include non-sensitive env vars — skip anything with
        # PASSWORD, SECRET, KEY, TOKEN in the name
        safe_env = {
            k: v for k, v in self.environment.items()
            if not any(s in k.upper() for s in ("PASSWORD", "SECRET", "KEY", "TOKEN", "API"))
        }
        if safe_env:
            env_str = ", ".join(f"{k}={v}" for k, v in safe_env.items())
            lines.append(f"Environment: {env_str}")

        if self.compose_project:
            lines.append(f"Compose project: {self.compose_project}")
        if self.compose_service:
            lines.append(f"Compose service: {self.compose_service}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """For JSON serialization / inspection."""
        return {
            "container_id": self.container_id,
            "name": self.name,
            "image": self.image,
            "status": self.status,
            "ports": self.ports,
            "networks": self.networks,
            "volumes": self.volumes,
            "environment": self.environment,
            "labels": self.labels,
            "compose_project": self.compose_project,
            "compose_service": self.compose_service,
        }


@dataclass
class NetworkDocument:
    """One document per Docker network."""
    network_id: str
    name: str
    driver: str
    scope: str
    subnet: Optional[str]
    gateway: Optional[str]
    connected_containers: list[str]

    def to_text(self) -> str:
        lines = [
            f"Network: {self.name}",
            f"ID: {self.network_id[:12]}",
            f"Driver: {self.driver}",
            f"Scope: {self.scope}",
        ]
        if self.subnet:
            lines.append(f"Subnet: {self.subnet}")
        if self.gateway:
            lines.append(f"Gateway: {self.gateway}")
        if self.connected_containers:
            lines.append(f"Connected containers: {', '.join(self.connected_containers)}")
        return "\n".join(lines)


@dataclass
class VolumeDocument:
    """One document per Docker volume."""
    name: str
    driver: str
    mountpoint: str
    labels: dict

    def to_text(self) -> str:
        lines = [
            f"Volume: {self.name}",
            f"Driver: {self.driver}",
            f"Mountpoint: {self.mountpoint}",
        ]
        if self.labels:
            label_str = ", ".join(f"{k}={v}" for k, v in self.labels.items())
            lines.append(f"Labels: {label_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

class DockerCrawler:
    """
    Reads Docker socket state and returns structured documents.

    OS framing: this is the hardware abstraction layer. It translates raw
    Docker API responses into clean, typed documents that the rest of the
    system can reason about.
    """

    def __init__(self, base_url: str = "unix://var/run/docker.sock"):
        self.client = docker.DockerClient(base_url=base_url)

    def crawl_containers(self, all_containers: bool = False) -> list[ContainerDocument]:
        """
        Crawl all containers (running only by default).
        Set all_containers=True to include stopped containers.
        """
        containers = self.client.containers.list(all=all_containers)
        docs = []

        for c in containers:
            c.reload()  # ensure attributes are fresh

            # Parse port mappings from the API response
            ports = c.ports  # dict: {"7878/tcp": [{"HostIp": "0.0.0.0", "HostPort": "7878"}]}

            # Parse network names
            networks = list(c.attrs["NetworkSettings"]["Networks"].keys())

            # Parse volume mounts — only named volumes and bind mounts
            volumes = []
            for mount in c.attrs.get("Mounts", []):
                if mount["Type"] == "bind":
                    volumes.append(f"{mount['Source']}:{mount['Destination']}")
                elif mount["Type"] == "volume":
                    volumes.append(f"{mount['Name']}:{mount['Destination']}")

            # Parse environment variables into a dict
            env_list = c.attrs["Config"].get("Env") or []
            env_dict = {}
            for entry in env_list:
                if "=" in entry:
                    k, _, v = entry.partition("=")
                    env_dict[k] = v

            # Compose metadata lives in labels
            labels = c.labels or {}
            compose_project = labels.get("com.docker.compose.project")
            compose_service = labels.get("com.docker.compose.service")

            docs.append(ContainerDocument(
                container_id=c.id,
                name=c.name,
                image=c.image.tags[0] if c.image.tags else c.image.short_id,
                status=c.status,
                ports=ports,
                networks=networks,
                volumes=volumes,
                environment=env_dict,
                labels=labels,
                compose_project=compose_project,
                compose_service=compose_service,
            ))

        return docs

    def crawl_networks(self) -> list[NetworkDocument]:
        """Crawl all user-defined Docker networks (skips bridge/host/none)."""
        system_networks = {"bridge", "host", "none"}
        networks = self.client.networks.list()
        docs = []

        for net in networks:
            if net.name in system_networks:
                continue

            net.reload()
            ipam = net.attrs.get("IPAM", {})
            config = ipam.get("Config") or [{}]
            subnet = config[0].get("Subnet") if config else None
            gateway = config[0].get("Gateway") if config else None

            connected = list(net.attrs.get("Containers", {}).keys())
            # Resolve container IDs to names where possible
            connected_names = []
            for cid in connected:
                try:
                    c = self.client.containers.get(cid)
                    connected_names.append(c.name)
                except Exception:
                    connected_names.append(cid[:12])

            docs.append(NetworkDocument(
                network_id=net.id,
                name=net.name,
                driver=net.attrs.get("Driver", "unknown"),
                scope=net.attrs.get("Scope", "unknown"),
                subnet=subnet,
                gateway=gateway,
                connected_containers=connected_names,
            ))

        return docs

    def crawl_volumes(self) -> list[VolumeDocument]:
        """Crawl all Docker volumes."""
        volumes = self.client.volumes.list()
        docs = []

        for vol in volumes:
            docs.append(VolumeDocument(
                name=vol.name,
                driver=vol.attrs.get("Driver", "local"),
                mountpoint=vol.attrs.get("Mountpoint", ""),
                labels=vol.attrs.get("Labels") or {},
            ))

        return docs

    def crawl_all(self) -> dict[str, list]:
        """
        Run all crawlers. Returns a dict with containers, networks, volumes.
        This is the single entry point for the ingestion pipeline.
        """
        return {
            "containers": self.crawl_containers(),
            "networks": self.crawl_networks(),
            "volumes": self.crawl_volumes(),
        }

    def close(self):
        self.client.close()


# ---------------------------------------------------------------------------
# CLI entry point — run directly to inspect output before embedding anything
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    crawler = DockerCrawler()

    try:
        print("=" * 60)
        print("DOCKER SOCKET CRAWLER — RAW OUTPUT")
        print("=" * 60)

        results = crawler.crawl_all()

        print(f"\n[CONTAINERS] ({len(results['containers'])} found)\n")
        for doc in results["containers"]:
            print(doc.to_text())
            print("-" * 40)

        print(f"\n[NETWORKS] ({len(results['networks'])} found)\n")
        for doc in results["networks"]:
            print(doc.to_text())
            print("-" * 40)

        print(f"\n[VOLUMES] ({len(results['volumes'])} found)\n")
        for doc in results["volumes"]:
            print(doc.to_text())
            print("-" * 40)

        # Also dump JSON for inspection
        output = {
            "containers": [d.to_dict() for d in results["containers"]],
            "networks": [d.to_text() for d in results["networks"]],
            "volumes": [d.to_text() for d in results["volumes"]],
        }
        with open("docker_state.json", "w") as f:
            json.dump(output, f, indent=2)
        print("\n[INFO] Full state written to docker_state.json")

    except docker.errors.DockerException as e:
        print(f"[ERROR] Could not connect to Docker socket: {e}")
        print("Make sure Docker is running and you have permission to read the socket.")
        print("You may need to add your user to the docker group: sudo usermod -aG docker $USER")
        sys.exit(1)

    finally:
        crawler.close()
