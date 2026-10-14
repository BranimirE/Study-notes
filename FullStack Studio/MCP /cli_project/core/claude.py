from anthropic import Anthropic
from anthropic.types import Message
import os
import json
from pathlib import Path


class Claude:
    def __init__(self, model: str):
        api_key, extra_headers = self._load_claude_credentials()
        if not api_key:
            raise RuntimeError("No ANTHROPIC_API_KEY env var or Claude Code credentials found")

        if extra_headers:
            self.client = Anthropic(auth_token=api_key, default_headers=extra_headers)
        else:
            self.client = Anthropic(api_key=api_key)

        self.model = model

    @staticmethod
    def _load_claude_credentials():
        """Returns (token, extra_headers_dict). Falls back: env var → Claude Code OAuth."""
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if key:
            return key, {}

        creds_path = Path.home() / ".claude" / ".credentials.json"
        try:
            data = json.loads(creds_path.read_text())
            oauth_token = data["claudeAiOauth"]["accessToken"]
            cc_version = Claude._get_claude_code_version()
            billing_header = f"cc_version={cc_version}; cc_entrypoint=cli; cch=00000;"
            return oauth_token, {"x-anthropic-billing-header": billing_header}
        except Exception:
            return "", {}

    @staticmethod
    def _get_claude_code_version():
        try:
            binary = Path.home() / ".local" / "share" / "claude" / "versions"
            versions = sorted(binary.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            return versions[0].name if versions else "2.1.146"
        except Exception:
            return "2.1.146"

    def add_user_message(self, messages: list, message):
        user_message = {
            "role": "user",
            "content": message.content
            if isinstance(message, Message)
            else message,
        }
        messages.append(user_message)

    def add_assistant_message(self, messages: list, message):
        assistant_message = {
            "role": "assistant",
            "content": message.content
            if isinstance(message, Message)
            else message,
        }
        messages.append(assistant_message)

    def text_from_message(self, message: Message):
        return "\n".join(
            [block.text for block in message.content if block.type == "text"]
        )

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ) -> Message:
        params = {
            "model": self.model,
            "max_tokens": 8000,
            "messages": messages,
            "temperature": temperature,
            "stop_sequences": stop_sequences,
        }

        if thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        if tools:
            params["tools"] = tools

        if system:
            params["system"] = system

        message = self.client.messages.create(**params)
        return message
