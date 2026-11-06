from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


class AnkiError(RuntimeError):
    pass


class AnkiConnectionError(AnkiError):
    pass


class AnkiAPIError(AnkiError):
    pass


class AnkiBridge:
    def __init__(
        self,
        url: str = "http://localhost:8765",
        *,
        timeout: float = 10,
        session: requests.Session | None = None,
    ):
        self.url = url
        self.timeout = timeout
        self.session = session or requests.Session()

    def _invoke(self, action: str, **params: Any) -> Any:
        logger.info("Anki-Connect action=%s started", action)
        try:
            response = self.session.post(
                self.url,
                json={"action": action, "version": 6, "params": params},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.error("Anki-Connect action=%s connection failed: %s", action, exc)
            raise AnkiConnectionError(
                f"Could not communicate with Anki-Connect at {self.url}. "
                "Check that Anki is running and the add-on is installed."
            ) from exc
        if payload.get("error"):
            logger.error("Anki-Connect action=%s failed: %s", action, payload["error"])
            raise AnkiAPIError(f"Anki-Connect {action} failed: {payload['error']}")
        logger.info("Anki-Connect action=%s succeeded", action)
        return payload.get("result")

    def ping(self) -> int:
        return self._invoke("version")

    def create_deck(self, name: str) -> int:
        return self._invoke("createDeck", deck=name)

    def add_note(
        self,
        *,
        deck: str,
        model: str,
        front: str,
        back: str,
        tags: list[str],
        audio_path: str | Path | None = None,
    ) -> int:
        note: dict[str, Any] = {
            "deckName": deck,
            "modelName": model,
            "fields": {"Front": front, "Back": back},
            "options": {"allowDuplicate": False},
            "tags": tags,
        }
        if audio_path:
            path = Path(audio_path).expanduser().resolve()
            note["audio"] = [
                {"path": str(path), "filename": path.name, "fields": ["Front"]}
            ]
        result = self._invoke("addNote", note=note)
        if result is None:
            raise AnkiAPIError("Anki-Connect addNote returned no note ID")
        return int(result)

    def delete_notes(self, note_ids: list[int]) -> None:
        self._invoke("deleteNotes", notes=note_ids)
