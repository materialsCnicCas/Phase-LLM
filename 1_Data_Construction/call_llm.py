# -*- coding: utf-8 -*-
"""
Unified LLM API Client for Phase-LLM.

Supports any OpenAI-compatible API endpoint (Qwen-Plus, DeepSeek, GPT-4, etc.).
API credentials are read from environment variables for security.

Usage:
    from call_llm import TeacherLLM
    llm = TeacherLLM()
    response = llm.retry_call("Your prompt here")
"""

import os
from urllib.parse import urlsplit, urlunsplit
from openai import OpenAI
from retrying import retry

# 设置代理，如果在国外可以注释掉这两行
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


def _first_nonempty_env(*keys: str, default: str = "") -> str:
    """Return the first non-empty environment variable value from keys."""
    for key in keys:
        value = os.environ.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def _compose_base_url(base_url: str, port: str) -> str:
    """Compose final base_url by injecting/overriding port for local endpoints only."""
    if not base_url:
        return base_url
    if not port:
        return base_url

    split = urlsplit(base_url)
    if not split.scheme or not split.hostname:
        return base_url

    # Safety rule: for public API domains, keep original URL/port (usually 443).
    # Only apply custom port to local/private endpoints.
    local_hosts = {"127.0.0.1", "localhost", "0.0.0.0"}
    if split.hostname not in local_hosts and not split.hostname.startswith("192.168.") and not split.hostname.startswith("10."):
        return base_url

    userinfo = ""
    if split.username:
        userinfo = split.username
        if split.password:
            userinfo += f":{split.password}"
        userinfo += "@"

    netloc = f"{userinfo}{split.hostname}:{port}"
    return urlunsplit((split.scheme, netloc, split.path, split.query, split.fragment))


class TeacherLLM:
    """
    Unified LLM client for the teacher model used in CoT data construction.

    Reads configuration from environment variables:
        TEACHER_API_KEY:  API key for the LLM service
        TEACHER_BASE_URL: Base URL of the API endpoint
        TEACHER_MODEL:    Model name (e.g., "qwen-plus", "deepseek-chat", "gpt-4o")

    Example:
        export TEACHER_API_KEY="sk-xxxx"
        export TEACHER_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
        export TEACHER_MODEL="qwen-plus"
    """

    def __init__(self, model: str = None, api_key: str = None, base_url: str = None, port: str = None):
        """
        Initialize the LLM client.

        Args:
            model:    Override TEACHER_MODEL env var
            api_key:  Override TEACHER_API_KEY env var
            base_url: Override TEACHER_BASE_URL env var
            port:     Optional port (e.g., "4780", "8000"). If provided, applied to base_url.
        """
        self.model = model or _first_nonempty_env("TEACHER_MODEL", default="qwen-plus")
        _api_key = api_key or _first_nonempty_env(
            "TEACHER_API_KEY",
            "OPENAI_API_KEY",
            "DASHSCOPE_API_KEY",
            "DEEPSEEK_API_KEY",
            default="",
        )
        _base_url = base_url or _first_nonempty_env(
            "TEACHER_BASE_URL",
            "OPENAI_BASE_URL",
            default="https://api.deepseek.com",
        )
        _port = str(port or _first_nonempty_env("TEACHER_PORT", "OPENAI_PORT", default="")).strip()
        _base_url = _compose_base_url(_base_url, _port)

        if not _api_key:
            raise ValueError(
                "API key not provided. Set one of: TEACHER_API_KEY / OPENAI_API_KEY / "
                "DASHSCOPE_API_KEY / DEEPSEEK_API_KEY, or pass api_key parameter."
            )

        self.client = OpenAI(api_key=_api_key, base_url=_base_url)

    def call(self, content: str, additional_args: dict = None) -> str:
        """
        Send a single prompt to the LLM and return the response.

        Args:
            content:         The user prompt string
            additional_args: Optional dict with 'temperature', 'top_p', 'max_completion_tokens'

        Returns:
            The model's response string
        """
        if additional_args is None:
            additional_args = {}

        messages = [{"role": "user", "content": content}]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=additional_args.get("max_completion_tokens", 8192),
            temperature=additional_args.get("temperature", 0.3),
            top_p=additional_args.get("top_p", 0.8),
        )

        answer = completion.choices[0].message.content
        return answer

    @retry(wait_fixed=3000, stop_max_attempt_number=5)
    def retry_call(self, content: str, additional_args: dict = None) -> str:
        """
        Call the LLM with automatic retry on failure (up to 5 attempts, 3s interval).

        Args:
            content:         The user prompt string
            additional_args: Optional dict with generation parameters

        Returns:
            The model's response string
        """
        if additional_args is None:
            additional_args = {"max_completion_tokens": 8192}
        return self.call(content, additional_args)


# ============================================================================
# Convenience: allow `from call_llm import llm_instance` for quick usage
# ============================================================================

def create_default_client() -> TeacherLLM:
    """Create a default client if env vars are set; otherwise return None."""
    try:
        return TeacherLLM()
    except ValueError:
        return None


llm_instance = create_default_client()
