from typing import Dict, Any


class ConfigManager:
    """
    Manages system configuration for LLM and other settings.
    """
    _config: Dict[str, Any] = {
        "max_chat_history": 50,
        "system_prompt": "You are an assistant for answering queries regarding a Room/Lecture Management System of an university.",
        "llm": {
            "google": {
                "api_key": "",
                "api_url": "https://generativelanguage.googleapis.com/v1beta/models",
                "model_name": "gemini-1.5-flash"
            }
        }
    }

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        return cls._config