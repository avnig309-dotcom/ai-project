"""Provider interfaces. Organizer request mapping intentionally awaits official docs."""
from __future__ import annotations
import os


class LocalProvider:
    mode = "local"
    def audit(self, *args, **kwargs):
        return {"mode": self.mode, "used": False}


class MockProvider:
    mode = "mock"
    def __init__(self, response=None): self.response = response or {"ok": True, "mock": True}
    def audit(self, *args, **kwargs): return {"mode": self.mode, "used": True, "response": self.response}


class OrganizerAPIProvider:
    """Configuration-only adapter until endpoint, auth, and schemas are documented."""
    mode = "organizer"
    def __init__(self):
        self.api_key = os.getenv("HACKATHON_API_KEY")
        self.base_url = os.getenv("HACKATHON_API_BASE_URL")
        self.model = os.getenv("HACKATHON_API_MODEL")
        self.timeout = float(os.getenv("HACKATHON_API_TIMEOUT_SECONDS", "10"))
        self.max_retries = min(3, max(0, int(os.getenv("HACKATHON_API_RETRIES", "1"))))
    def audit(self, *args, **kwargs):
        raise RuntimeError("Organizer API request/response mapping is not implemented: official API documentation is required.")


def configured_provider():
    mode = os.getenv("AI_PROVIDER_MODE", "local").lower()
    return {"local": LocalProvider, "mock": MockProvider, "organizer": OrganizerAPIProvider}.get(mode, LocalProvider)()
