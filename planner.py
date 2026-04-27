"""
planner.py
----------
Planning layer — the brain of the orchestrator.

Takes a natural language goal, retrieves relevant context from ChromaDB,
calls Qwen via Ollama with a structured prompt, and returns a validated
Pydantic TaskPlan. Retries up to MAX_RETRIES times on validation failure,
injecting the specific error back into the prompt each time.

OS framing: this is the process scheduler. It takes a high-level intent
(the user goal, analogous to a program the user wants to run) and produces
a concrete execution plan (the task graph, analogous to the sequence of
syscalls that program will make).
"""

import json
import os
import socket
import requests
from dotenv import load_dotenv

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

from schema import TaskPlan, TrustLevel
from prompts import SYSTEM_PROMPT, build_planning_prompt

load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHROMA_HOST       = "localhost"
CHROMA_PORT       = 8000
COLLECTION_NAME   = "homelab_state"
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PLANNING_MODEL    = os.getenv("PLANNING_MODEL", "qwen2.5:3b")
EMBED_MODEL       = "nomic-embed-text"
RETRIEVAL_TOP_K   = 4     # how many chunks to retrieve per query
MAX_RETRIES       = 3     # max Pydantic validation retry attempts


# ---------------------------------------------------------------------------
# IP resolution
# ---------------------------------------------------------------------------

def get_public_ip() -> str:
    """
    Fetch the server's public IP address dynamically.
    Used for Cloudflare DNS A records — points the domain at this machine.

    Tries multiple providers in order so a single provider outage doesn't
    break deployment. Falls back to the local network IP if all fail
    (useful for LAN-only homelabs without a public IP).
    """
    providers = [
        "https://api.ipify.org",
        "https://ifconfig.me/ip",
        "https://icanhazip.com",
    ]
    for url in providers:
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            ip = resp.text.strip()
            if ip:
                return ip
        except requests.RequestException:
            continue

    # All public IP providers failed — fall back to local network IP
    print("[planner] Warning: could not fetch public IP, falling back to local IP.")
    return _get_local_ip()


def _get_local_ip() -> str:
    """Get the local network IP of this machine."""
    try:
        # Connect to an external address to determine the outbound interface IP
        # No actual data is sent — this just resolves the local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Wraps ChromaDB + LlamaIndex retrieval.
    Retrieves the top-k most relevant chunks for a given query.
    """

    def __init__(self):
        Settings.llm = None

        self.embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = chroma_client.get_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=collection)

        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self.embed_model,
        )

        self.retriever = index.as_retriever(
            embed_model=self.embed_model,
            similarity_top_k=RETRIEVAL_TOP_K,
        )

    def retrieve(self, query: str) -> str:
        """
        Retrieve top-k chunks and format them as a single text block
        for injection into the planning prompt.
        """
        nodes = self.retriever.retrieve(query)
        chunks = []
        for node in nodes:
            chunks.append(node.text)
        return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def call_ollama(system_prompt: str, user_prompt: str) -> str:
    """
    Call Ollama's chat endpoint with a system + user message.
    Returns the raw text response from the model.

    Using the /api/chat endpoint (not /api/generate) because it supports
    system/user role separation cleanly.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": PLANNING_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "options": {
            # Lower temperature = more deterministic JSON output
            # Higher would make the model more creative but less reliable
            "temperature": 0.1,
            "top_p": 0.9,
        }
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    return data["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------

def get_schema_json() -> str:
    """
    Generate a compact JSON schema description from the TaskPlan Pydantic model.
    Injected into the planning prompt so the LLM sees the exact structure.
    """
    schema = TaskPlan.model_json_schema()
    return json.dumps(schema, indent=2)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """
    Orchestrates the full planning pipeline:
    1. Retrieve relevant context from ChromaDB
    2. Build the planning prompt
    3. Call Qwen via Ollama
    4. Parse and validate as TaskPlan
    5. Retry with error feedback on validation failure
    6. Present human confirmation gate for destructive tasks
    """

    def __init__(self):
        self.retriever = Retriever()

    def _parse_response(self, raw: str) -> TaskPlan:
        """
        Parse raw LLM output as TaskPlan.
        Strips markdown code fences if the model added them despite instructions.
        """
        # Strip common markdown wrapping
        text = raw.strip()
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        data = json.loads(text)
        return TaskPlan.model_validate(data)

    def _confirmation_gate(self, plan: TaskPlan) -> bool:
        """
        Human confirmation gate. Always shown for plans with destructive tasks.
        Returns True if the user confirms, False if they abort.

        OS framing: this is the privilege escalation prompt — like sudo asking
        for a password before a root operation.
        """
        print("\n" + "=" * 60)
        print("PLAN REVIEW — CONFIRMATION REQUIRED")
        print("=" * 60)
        print(plan.summary())

        if plan.has_destructive_tasks():
            print("\n⚠️  This plan contains DESTRUCTIVE tasks:")
            for t in plan.destructive_tasks():
                print(f"   - {t.id}: {t.description}")

        print("\nProceed with execution? [y/N] ", end="", flush=True)
        answer = input().strip().lower()
        return answer == "y"

    def plan(self, goal: str, auto_confirm: bool = False) -> TaskPlan | None:
        """
        Full planning pipeline for a user goal.

        Args:
            goal:         Natural language goal from the user.
            auto_confirm: Skip the confirmation gate (use only for read-only plans
                          or in testing). Never auto-confirm destructive plans.

        Returns:
            Validated TaskPlan, or None if the user aborted at the confirmation gate.
        """
        print(f"\n[planner] Goal: {goal}")
        print(f"[planner] Retrieving context from ChromaDB...")

        # Retrieve relevant context from ChromaDB
        context = self.retriever.retrieve(goal)

        # Append server environment facts so the LLM has them without
        # needing to retrieve them — IP, domain, paths are always relevant
        public_ip = get_public_ip()
        env_context = "\n".join(filter(None, [
            f"Server public IP: {public_ip}",
            f"Domain: {os.getenv('CF_DOMAIN', '')}",
            f"Nginx sites-available: {os.getenv('NGINX_SITES_AVAILABLE', '/etc/nginx/sites-available')}",
            f"Certbot webroot: {os.getenv('NGINX_WEBROOT', '/var/www/html')}",
            f"Certbot email: {os.getenv('CERTBOT_EMAIL', '')}",
        ]))
        context = f"{env_context}\n\n{context}"

        print(f"[planner] Context ready ({len(context.splitlines())} lines, public IP: {public_ip}).")

        error_feedback = ""
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"[planner] Planning attempt {attempt}/{MAX_RETRIES}...")

            # Build prompt
            prompt = build_planning_prompt(
                goal=goal,
                retrieved_context=context,
                error_feedback=error_feedback,
            )

            # Call Qwen
            try:
                raw_response = call_ollama(SYSTEM_PROMPT, prompt)
            except requests.RequestException as e:
                print(f"[planner] Ollama call failed: {e}")
                raise

            # Parse and validate
            try:
                plan = self._parse_response(raw_response)
                print(f"[planner] Plan validated. {len(plan.tasks)} tasks.")

                # Confirmation gate
                if auto_confirm and not plan.has_destructive_tasks():
                    return plan

                confirmed = self._confirmation_gate(plan)
                if confirmed:
                    return plan
                else:
                    print("[planner] Aborted by user.")
                    return None

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                error_feedback = str(e)
                print(f"[planner] Validation failed (attempt {attempt}): {e}")
                if attempt < MAX_RETRIES:
                    print(f"[planner] Retrying with error feedback...")

        print(f"[planner] All {MAX_RETRIES} attempts failed. Last error: {last_error}")
        raise ValueError(
            f"Planning failed after {MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )


# ---------------------------------------------------------------------------
# Entry point — test the planner directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    planner = Planner()

    test_goals = [
        "Check which containers are currently running",
        "Add a Radarr container with nginx reverse proxy and SSL",
    ]

    for goal in test_goals:
        print("\n" + "=" * 60)
        try:
            plan = planner.plan(goal, auto_confirm=True)
            if plan:
                print("\n[result] Final plan:")
                print(plan.summary())
                print("\n[result] Full JSON:")
                print(plan.model_dump_json(indent=2))
        except Exception as e:
            print(f"[error] {e}")
