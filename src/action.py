import json

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional, Tuple
from typing import Any, Optional, Dict
from pydantic import BaseModel, ValidationError


class NoParams(BaseModel):
    pass

class BaseAction(ABC):
    name: str
    description: str = ""
    hint: str = ""
    requires_confirmation: bool = False
    is_always_anomalie: bool = False
    params_model: Optional[Any] = None

    def __init__(self):
        self._pre_hooks: List[Callable] = []
        self._post_hooks: List[Callable] = []

    @abstractmethod
    def _do_execute(self, params_instance: BaseModel, metadata: dict, is_anomaly: bool) -> Dict[str, Any]:
        pass

    def simulate_execute(self, params_instance: BaseModel) -> Dict[str, Any]:
        msg_to_return = {
            "status": "SIMULATION",
            "data": {
                "action_name": self.name,
                "message": f"Action will be executed with params: {params_instance.model_dump()}.\nReply with 'Yes' to confirm or 'No' to cancel."
            }
        }
        self.logger.debug("Result of Simulation", msg_to_return)
        return msg_to_return

    def check_for_anomalies(self, params_instance: BaseModel) -> bool:
        return False

    def execute(self, is_simulation: bool, metadata: dict = None, **kwargs) -> Dict[str, Any]:
        if not self.params_model:
            params_instance = NoParams()
        else:
            if isinstance(self, RecommendingAction):
                # Fill optional parameters with default values if missing.
                recommended, explanation = self.recommend_parameters(kwargs)
                merged = {**recommended, **kwargs}
                try:
                    params_instance = self.params_model(**merged)
                except ValidationError as e:
                    msg_to_return = {
                        "status": "ERROR",
                        "hint": self.hint,
                        "error_message": json.loads(e.json())
                    }
                    self.logger.error("Validation error", msg_to_return)
                    return msg_to_return
            else:
                try:
                    params_instance = self.params_model(**kwargs)
                except ValidationError as e:
                    msg_to_return = {
                        "status": "ERROR",
                        "hint": self.hint,
                        "error_message": json.loads(e.json())
                    }
                    self.logger.error("Validation error", msg_to_return)
                    return msg_to_return
        if self.requires_confirmation and is_simulation:
            simulated = self.simulate_execute(params_instance)
            msg_to_return = {
                "status": "PENDING_CONFIRMATION",
                "simulation": simulated
            }
            self.logger.debug("Result of Simulation", msg_to_return)
            return msg_to_return
        if self.is_always_anomalie:
            is_anomaly = True
        else:
            is_anomaly = self.check_for_anomalies(params_instance)
        return self._do_execute(params_instance, metadata, is_anomaly)


class RecommendingAction(BaseAction, ABC):
    @abstractmethod
    def recommend_parameters(self, provided_params: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """
        Returns a tuple (recommended_values, explanation).
        'recommended_values' is a dict with default values for optional parameters that were not provided.
        'explanation' is a string explaining the recommendations (can be empty if all required values are present).
        """
        pass


# ActionRegistry and ActionManager
class ActionRegistry:
    _instance = None

    @classmethod
    def get_instance(cls) -> "ActionRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def register(cls):
        def decorator(action_class: type) -> type:
            ActionRegistry.get_instance().register_action(action_class)
            return action_class

        return decorator

    def __init__(self):
        self._actions: Dict[str, BaseAction] = {}

    def register_action(self, action_class: type) -> None:
        instance = action_class()
        self._actions[instance.name] = instance

    def get_action(self, action_name: str) -> Optional[BaseAction]:
        return self._actions.get(action_name)

    def get_all_actions(self) -> Dict[str, BaseAction]:
        return self._actions


class ActionManager:
    def __init__(self):
        self.actions_registry = ActionRegistry.get_instance()

    def get_all_actions_schema(self) -> List[Dict[str, Any]]:
        actions = []
        for action in self.actions_registry.get_all_actions().values():
            schema = {
                "action_name": action.name,
                "description": action.description,
                "hint": action.hint,
            }
            required_params = []
            optional_params = []
            if action.params_model:
                for field_name, field in action.params_model.model_fields.items():
                    field_type = str(field.annotation.__name__) if field.annotation else "Unknown"
                    field_desc = f" - {field.description}" if field.description else ""
                    param_descr = f"{field_name} ({field_type}){field_desc}"
                    if field.is_required():
                        required_params.append(param_descr)
                    else:
                        optional_params.append(param_descr)
            schema["required_params"] = required_params
            schema["optional_params"] = optional_params
            actions.append(schema)
        return actions

    def get_action(self, action_name: str) -> Optional[BaseAction]:
        return self.actions_registry.get_action(action_name)

    def get_all_actions(self) -> Dict[str, BaseAction]:
        return self.actions_registry.get_all_actions()


# ------------------------------------------------------------
# CONCRETE ACTIONS ARE NOT PUBLICLY AVAILABLE
# ------------------------------------------------------------