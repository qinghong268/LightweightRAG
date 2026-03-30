import copy
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class ConversationStore:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self._lock = threading.RLock()

    def _new_state(self) -> Dict[str, Any]:
        return {
            "active_session_id": str(uuid.uuid4()),
            "updated_at": None,
            "messages": [],
        }

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        for item in messages or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            normalized.append({"role": role, "content": content})
        return normalized

    def load_state(self) -> Dict[str, Any]:
        with self._lock:
            if not self.file_path.exists():
                state = self._new_state()
                self._save_state_locked(state)
                return state

            try:
                raw_state = json.loads(self.file_path.read_text(encoding="utf-8"))
            except Exception:
                state = self._new_state()
                self._save_state_locked(state)
                return state

            state = self._new_state()
            if isinstance(raw_state, dict):
                active_session_id = str(raw_state.get("active_session_id", "")).strip()
                if active_session_id:
                    state["active_session_id"] = active_session_id
                updated_at = raw_state.get("updated_at")
                if isinstance(updated_at, str):
                    state["updated_at"] = updated_at
                state["messages"] = self._normalize_messages(raw_state.get("messages", []))
            return state

    def _save_state_locked(self, state: Dict[str, Any]) -> Dict[str, Any]:
        normalized_state = {
            "active_session_id": str(state.get("active_session_id") or uuid.uuid4()),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": self._normalize_messages(state.get("messages", [])),
        }
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.file_path.with_suffix(f"{self.file_path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(normalized_state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.file_path)
        return normalized_state

    def save_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            return self._save_state_locked(state)

    def get_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self.load_state())

    def get_messages(self) -> List[Dict[str, str]]:
        return self.get_state()["messages"]

    def set_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            state = self.load_state()
            state["messages"] = messages
            return copy.deepcopy(self._save_state_locked(state))

    def reset_session(self) -> Dict[str, Any]:
        with self._lock:
            state = self._new_state()
            return copy.deepcopy(self._save_state_locked(state))
