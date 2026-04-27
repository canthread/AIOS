"""
prompts.py
----------
System prompt and few-shot examples for the planning layer.

Kept separate from planner.py so prompt iteration doesn't require touching
orchestration logic. The prompt is the most important tuning surface —
small wording changes have large effects on output quality.

Design principles:
- Every constraint stated positively AND negatively ("use X", "never use Y")
- Few-shot examples use real field names and realistic values
- JSON schema injected at runtime so the LLM sees the exact structure required
- Error feedback injected on retry so the LLM knows what went wrong
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a homelab infrastructure orchestrator. Your job is to convert a natural language goal into a structured JSON task plan that can be executed safely on a Linux server running Docker, Nginx, and Certbot.

STRICT RULES:
1. Output ONLY valid JSON. No markdown, no code fences, no explanation before or after.
2. Every field in the schema is required unless marked optional. Never omit required fields.
3. Use ONLY the exec_type values defined in the schema. Never invent new ones.
4. Use ONLY real values from the retrieved context (container names, paths, network names, ports). Never guess or hallucinate values not present in the context.
5. If the context does not contain enough information to complete the plan safely, output a plan with a single shell task that runs `echo "INSUFFICIENT_CONTEXT: <reason>"` and trust_level=read.
6. depends_on must reference task IDs that exist in the same plan. Never reference IDs that don't exist.
7. trust_level rules:
   - read: shell commands that only inspect state (docker ps, cat, ls, nginx -t)
   - write: file_write, config_update, docker_create, nginx_reload
   - destructive: docker_stop, dns_cloudflare
8. For any service deployment, always follow this order:
   a. write docker-compose.yml
   b. docker compose up
   c. write nginx config
   d. test nginx config (shell: nginx -t)
   e. enable nginx site (shell: ln -s)
   f. nginx reload
   g. certbot for SSL
   h. nginx reload again
   i. cloudflare DNS (last — only after service is verified running)
9. Never write secrets or credentials into file content. Use environment variable references like ${CF_API_TOKEN}.
10. Use absolute paths everywhere. ~/docker/<service>/docker-compose.yml expands to /home/<user>/docker/<service>/docker-compose.yml — use the full path from context.
"""


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------
# These are (goal, context_summary, plan) triples.
# The planner injects them before the actual user goal.
# Keep them realistic and consistent with the actual schema.

FEW_SHOT_EXAMPLES = [
    {
        "goal": "Add a Radarr container with nginx reverse proxy and SSL",
        "context": """
Container: n8n-compose-traefik-1 | Image: traefik:latest | Status: running | Ports: 80:80, 443:443
Network: n8n-compose_default | Subnet: 172.19.0.0/16
Compose service: n8n | Stack: n8n-compose | Source: /home/toutou/docker/n8n-compose/docker-compose.yml
""",
        "plan": """{
  "goal": "Add a Radarr container with nginx reverse proxy and SSL",
  "reasoning": "Radarr needs a compose file, then the container started, then nginx configured with a server block, SSL obtained via certbot webroot, and finally a Cloudflare DNS A record created pointing to the server.",
  "tasks": [
    {
      "id": "write_compose",
      "description": "Write docker-compose.yml for Radarr at /home/toutou/docker/radarr/",
      "exec_type": "file_write",
      "path": "/home/toutou/docker/radarr/docker-compose.yml",
      "content": "services:\\n  radarr:\\n    image: linuxserver/radarr:latest\\n    container_name: radarr\\n    environment:\\n      - PUID=1000\\n      - PGID=1000\\n      - TZ=Europe/London\\n    volumes:\\n      - radarr_config:/config\\n      - /data/media:/movies\\n    ports:\\n      - \\"7878:7878\\"\\n    restart: unless-stopped\\nvolumes:\\n  radarr_config:\\n",
      "depends_on": [],
      "trust_level": "write"
    },
    {
      "id": "start_radarr",
      "description": "Start Radarr container via docker compose",
      "exec_type": "docker_create",
      "stack_path": "/home/toutou/docker/radarr",
      "depends_on": ["write_compose"],
      "trust_level": "write"
    },
    {
      "id": "write_nginx",
      "description": "Write nginx server block for radarr.canthread.com",
      "exec_type": "file_write",
      "path": "/etc/nginx/sites-available/radarr.conf",
      "content": "server {\\n    listen 80;\\n    server_name radarr.canthread.com;\\n    location /.well-known/acme-challenge/ {\\n        root /var/www/html;\\n    }\\n    location / {\\n        proxy_pass http://localhost:7878;\\n        proxy_set_header Host $host;\\n        proxy_set_header X-Real-IP $remote_addr;\\n    }\\n}\\n",
      "depends_on": ["start_radarr"],
      "trust_level": "write"
    },
    {
      "id": "test_nginx",
      "description": "Test nginx config syntax before enabling",
      "exec_type": "shell",
      "command": "nginx -t",
      "depends_on": ["write_nginx"],
      "trust_level": "read"
    },
    {
      "id": "enable_nginx",
      "description": "Enable the radarr nginx site",
      "exec_type": "shell",
      "command": "ln -sf /etc/nginx/sites-available/radarr.conf /etc/nginx/sites-enabled/radarr.conf",
      "depends_on": ["test_nginx"],
      "trust_level": "write"
    },
    {
      "id": "reload_nginx_1",
      "description": "Reload nginx to pick up new site config",
      "exec_type": "nginx_reload",
      "depends_on": ["enable_nginx"],
      "trust_level": "write"
    },
    {
      "id": "certbot_ssl",
      "description": "Obtain SSL certificate for radarr.canthread.com via certbot webroot",
      "exec_type": "shell",
      "command": "certbot certonly --webroot -w /var/www/html -d radarr.canthread.com --non-interactive --agree-tos",
      "depends_on": ["reload_nginx_1"],
      "trust_level": "write"
    },
    {
      "id": "reload_nginx_2",
      "description": "Reload nginx after SSL certificate issued",
      "exec_type": "nginx_reload",
      "depends_on": ["certbot_ssl"],
      "trust_level": "write"
    },
    {
      "id": "dns_radarr",
      "description": "Create Cloudflare A record for radarr.canthread.com",
      "exec_type": "dns_cloudflare",
      "dns_record": "radarr",
      "dns_type": "A",
      "depends_on": ["reload_nginx_2"],
      "trust_level": "destructive"
    }
  ]
}"""
    },
    {
        "goal": "Check which containers are currently running and their status",
        "context": """
Container: n8n-compose-traefik-1 | Status: running
Container: searxng | Status: running
""",
        "plan": """{
  "goal": "Check which containers are currently running and their status",
  "reasoning": "This is a read-only inspection task. No changes needed — just run docker ps to show current container state.",
  "tasks": [
    {
      "id": "check_containers",
      "description": "List all running containers with their status",
      "exec_type": "shell",
      "command": "docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'",
      "depends_on": [],
      "trust_level": "read"
    }
  ]
}"""
    },
    {
        "goal": "Stop the searxng container",
        "context": """
Container: searxng | Status: running | Compose project: searxng-cli
Compose service: searxng | Source: /home/toutou/docker/searxng/docker-compose.yml
""",
        "plan": """{
  "goal": "Stop the searxng container",
  "reasoning": "Stopping a running container is a destructive action — it interrupts the service. Using docker compose down via the stack path to cleanly stop and remove the container.",
  "tasks": [
    {
      "id": "stop_searxng",
      "description": "Stop the searxng container via docker compose down",
      "exec_type": "docker_stop",
      "stack_path": "/home/toutou/docker/searxng",
      "depends_on": [],
      "trust_level": "destructive"
    }
  ]
}"""
    },
]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_few_shot_block() -> str:
    """Format few-shot examples as a string block for injection into the prompt."""
    # Only use the first example (full deployment) to keep prompt size small.
    # One detailed example teaches structure better than three examples that
    # blow the context window on a 3b model.
    ex = FEW_SHOT_EXAMPLES[0]
    return (
        f"EXAMPLE GOAL: {ex['goal']}\n"
        f"EXAMPLE CONTEXT:\n{ex['context'].strip()}\n"
        f"EXAMPLE OUTPUT:\n{ex['plan']}"
    )


def build_planning_prompt(
    goal: str,
    retrieved_context: str,
    schema_json: str = "",   # kept for signature compat, no longer injected
    error_feedback: str = "",
) -> str:
    """
    Build the full prompt for a planning call.

    Args:
        goal:               The user's natural language goal.
        retrieved_context:  Top-k chunks from ChromaDB, formatted as text.
        schema_json:        Unused — schema removed from prompt to save tokens.
                            Few-shot examples demonstrate structure instead.
        error_feedback:     On retry, the Pydantic validation error from the
                            previous attempt. Empty string on first attempt.
    """
    retry_block = ""
    if error_feedback:
        retry_block = f"""
PREVIOUS ATTEMPT FAILED VALIDATION:
{error_feedback}

Fix the above error and output only the corrected JSON.
"""

    return f"""CURRENT INFRASTRUCTURE STATE (use these values only — do not invent others):
{retrieved_context}

FEW-SHOT EXAMPLE (follow this exact JSON structure):
{build_few_shot_block()}

---
{retry_block}
NOW GENERATE A PLAN FOR THIS GOAL:
{goal}

Output only valid JSON. No markdown, no explanation.
"""
