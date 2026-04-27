"""
schema.py
---------
Pydantic models for the task graph produced by the planning layer.

OS framing: these are the system call definitions. Just as a kernel exposes
a finite set of syscalls that user-space can invoke, the orchestrator exposes
a finite set of exec_types. The LLM cannot invent operations outside this set.

Every field has a description — these get injected into the planning prompt
so the LLM knows exactly what each field means and what values are valid.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enums — the finite vocabularies the LLM must choose from
# ---------------------------------------------------------------------------

class ExecType(str, Enum):
    """
    The complete set of operations the orchestrator can perform.
    Adding a new operation here requires a corresponding handler in the
    execution layer — the schema and the executor must stay in sync.
    """
    SHELL          = "shell"           # run a shell command (inspection, certbot, etc.)
    FILE_WRITE     = "file_write"      # write a new file to disk
    DOCKER_CREATE  = "docker_create"   # docker compose up -d for a stack
    DOCKER_STOP    = "docker_stop"     # docker compose down for a stack
    CONFIG_UPDATE  = "config_update"   # modify an existing config file in place
    NGINX_RELOAD   = "nginx_reload"    # nginx -s reload (safe, no args needed)
    DNS_CLOUDFLARE = "dns_cloudflare"  # create or update a Cloudflare DNS record


class TrustLevel(str, Enum):
    """
    Protection ring equivalent. Determines whether human confirmation
    is required before execution.

    read        — no side effects, safe to run without confirmation
    write       — creates or modifies files/containers, confirmation recommended
    destructive — external side effects or hard-to-reverse actions, always confirm
    """
    READ        = "read"
    WRITE       = "write"
    DESTRUCTIVE = "destructive"


class DnsRecordType(str, Enum):
    A     = "A"
    CNAME = "CNAME"
    TXT   = "TXT"
    MX    = "MX"


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """
    A single unit of work in the task graph.
    Maps directly to one operation the execution layer will perform.
    """

    id: str = Field(
        description="Unique identifier for this task within the plan. "
                    "Use short snake_case strings e.g. 'write_compose', 'start_container'."
    )

    description: str = Field(
        description="Human-readable description of what this task does. "
                    "Be specific — include service names, file paths, port numbers."
    )

    exec_type: ExecType = Field(
        description="The type of operation to perform. Must be one of the defined ExecType values."
    )

    # --- shell / nginx_reload ---
    command: Optional[str] = Field(
        default=None,
        description="Shell command to run. Required for exec_type=shell. "
                    "For nginx_reload, leave null — the executor handles the command. "
                    "Use absolute paths. Do not use sudo — the orchestrator handles privilege."
    )

    # --- file_write / config_update ---
    path: Optional[str] = Field(
        default=None,
        description="Absolute path to the file to write or update. "
                    "Required for exec_type=file_write and config_update. "
                    "Use real paths from the retrieved context — never guess."
    )

    content: Optional[str] = Field(
        default=None,
        description="File content to write. Required for exec_type=file_write. "
                    "For docker-compose files, use valid YAML. "
                    "For nginx configs, use valid nginx config syntax."
    )

    # --- docker_create / docker_stop ---
    stack_path: Optional[str] = Field(
        default=None,
        description="Absolute path to the directory containing docker-compose.yml. "
                    "Required for exec_type=docker_create and docker_stop. "
                    "e.g. '/home/user/docker/radarr'"
    )

    # --- dns_cloudflare ---
    dns_record: Optional[str] = Field(
        default=None,
        description="Subdomain to create or update. e.g. 'radarr' will create "
                    "radarr.<CF_DOMAIN>. Required for exec_type=dns_cloudflare."
    )

    dns_type: Optional[DnsRecordType] = Field(
        default=DnsRecordType.A,
        description="DNS record type. Defaults to A (points to SERVER_IP). "
                    "Use CNAME for aliases."
    )

    dns_value: Optional[str] = Field(
        default=None,
        description="DNS record value. For A records, leave null — the executor "
                    "uses SERVER_IP from the environment. For CNAME, provide the target."
    )

    # --- dependency graph ---
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of task IDs that must complete successfully before this "
                    "task runs. Use this to encode ordering — e.g. docker_create "
                    "must depend on file_write for the compose file."
    )

    trust_level: TrustLevel = Field(
        description="Trust level for this task. "
                    "read = no side effects. "
                    "write = creates/modifies files or containers. "
                    "destructive = dns_cloudflare, docker_stop, or irreversible actions."
    )

    @field_validator("command")
    @classmethod
    def command_required_for_shell(cls, v, info):
        if info.data.get("exec_type") == ExecType.SHELL and not v:
            raise ValueError("command is required when exec_type is 'shell'")
        return v

    @field_validator("path")
    @classmethod
    def path_required_for_file_ops(cls, v, info):
        exec_type = info.data.get("exec_type")
        if exec_type in (ExecType.FILE_WRITE, ExecType.CONFIG_UPDATE) and not v:
            raise ValueError(f"path is required when exec_type is '{exec_type}'")
        return v

    @field_validator("stack_path")
    @classmethod
    def stack_path_required_for_docker(cls, v, info):
        exec_type = info.data.get("exec_type")
        if exec_type in (ExecType.DOCKER_CREATE, ExecType.DOCKER_STOP) and not v:
            raise ValueError(f"stack_path is required when exec_type is '{exec_type}'")
        return v

    @field_validator("dns_record")
    @classmethod
    def dns_record_required_for_cloudflare(cls, v, info):
        if info.data.get("exec_type") == ExecType.DNS_CLOUDFLARE and not v:
            raise ValueError("dns_record is required when exec_type is 'dns_cloudflare'")
        return v


# ---------------------------------------------------------------------------
# TaskPlan model
# ---------------------------------------------------------------------------

class TaskPlan(BaseModel):
    """
    The complete output of the planning layer.
    A validated TaskPlan is the only thing the execution layer will accept.
    """

    goal: str = Field(
        description="The original user goal, preserved verbatim."
    )

    reasoning: str = Field(
        description="Brief explanation of why these tasks were chosen and "
                    "in what order. Used for human review before execution."
    )

    tasks: list[Task] = Field(
        description="Ordered list of tasks. Tasks with no depends_on can run "
                    "immediately. Tasks with depends_on wait for their dependencies."
    )

    @field_validator("tasks")
    @classmethod
    def validate_dependency_references(cls, tasks):
        """Ensure all depends_on references point to real task IDs."""
        task_ids = {t.id for t in tasks}
        for task in tasks:
            for dep in task.depends_on:
                if dep not in task_ids:
                    raise ValueError(
                        f"Task '{task.id}' depends_on '{dep}' "
                        f"which does not exist in this plan. "
                        f"Valid task IDs are: {sorted(task_ids)}"
                    )
        return tasks

    @field_validator("tasks")
    @classmethod
    def validate_no_empty_plan(cls, tasks):
        if not tasks:
            raise ValueError("TaskPlan must contain at least one task.")
        return tasks

    def has_destructive_tasks(self) -> bool:
        return any(t.trust_level == TrustLevel.DESTRUCTIVE for t in self.tasks)

    def destructive_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.trust_level == TrustLevel.DESTRUCTIVE]

    def summary(self) -> str:
        """Human-readable plan summary for the confirmation gate."""
        lines = [
            f"Goal: {self.goal}",
            f"Reasoning: {self.reasoning}",
            f"Tasks ({len(self.tasks)}):",
        ]
        for t in self.tasks:
            deps = f" [depends: {', '.join(t.depends_on)}]" if t.depends_on else ""
            lines.append(
                f"  [{t.trust_level.value.upper()}] {t.id}: "
                f"{t.exec_type.value} — {t.description}{deps}"
            )
        return "\n".join(lines)
