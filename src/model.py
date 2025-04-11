import re
import requests
import json

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from typing import Any, Optional, Dict
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_fixed
from config import ConfigManager


@dataclass
class LLMResponse:
    """
    Standardized response for LLM calls.
    """
    text: Optional[str]
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None


class BaseLLMModel(ABC):
    """
    Abstract base class for large language models.
    Each subclass must provide build_payload() and extract_text().
    """

    def __init__(self, api_key: str, api_url: str, model_name: str, system_prompt: str):
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.system_prompt = system_prompt

    @abstractmethod
    def build_payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        pass

    def build_header(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    @abstractmethod
    def extract_text(self, response_data: Dict[str, Any]) -> str:
        pass

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(2))
    def process(self, messages: List[Dict[str, str]]) -> LLMResponse:
        headers = self.build_header(messages)
        payload = self.build_payload(messages)
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return LLMResponse(
                text=None,
                error=str(e),
                status_code=(response.status_code if response is not None else None),
            )
        response_data = response.json()
        try:
            text = self.extract_text(response_data)
            return LLMResponse(text=text, raw_response=response_data, status_code=response.status_code)
        except Exception as e:
            return LLMResponse(text=None, error="Invalid API response format", raw_response=response_data)

    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        text_clean = re.sub(r'```(json)?|```', '', text, flags=re.IGNORECASE)
        decoder = json.JSONDecoder()
        start_index = 0
        while start_index < len(text_clean):
            if text_clean[start_index] != '{':
                start_index += 1
                continue
            try:
                obj, _ = decoder.raw_decode(text_clean[start_index:])
                return obj
            except json.JSONDecodeError:
                start_index += 1
        raise json.JSONDecodeError("No valid JSON found", text_clean, 0)

    @classmethod
    def from_config(cls, provider: str, system_prompt: str):
        config = ConfigManager.get_config()["llm"].get(provider)
        if not config:
            raise ValueError(f"Unknown provider: {provider}")
        return cls(
            api_key=config["api_key"],
            api_url=config["api_url"],
            model_name=config["model_name"],
            system_prompt=system_prompt
        )

class GoogleLLMModel(BaseLLMModel):
    def __init__(self, system_prompt: str):
        config = ConfigManager.get_config()["llm"]["google"]
        api_key = config["api_key"]
        api_url = f"{config['api_url']}/{config['model_name']}:generateContent?key={api_key}"
        super().__init__(
            api_key=config["api_key"],
            api_url=api_url,
            model_name=config["model_name"],
            system_prompt=system_prompt
        )

    def build_header(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return {"Content-Type": "application/json"}

    def build_payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        messages_gemini = []
        system_prompt = None
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                messages_gemini.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                messages_gemini.append({"role": "model", "parts": [{"text": msg["content"]}]})
        payload = {"contents": messages_gemini}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        return payload

    def extract_text(self, response_data: Dict[str, Any]) -> str:
        return response_data["candidates"][0]["content"]["parts"][0]["text"]