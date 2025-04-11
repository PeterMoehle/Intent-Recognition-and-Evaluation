import asyncio
import json

from datetime import datetime


class QueryInterpreterModule:
    """
    Converts conversation history + user query into actionable JSON.
    """

    def __init__(self, llm_model, action_manager):
        self.llm = llm_model
        self.action_manager = action_manager
        self.combined_system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        actions_schema = json.dumps(self.action_manager.get_all_actions_schema(), indent=2)
        return (
            f"{self.llm.system_prompt}\n\n"
            "**Steps to follow:**\n"
            "1. Analyze the user’s request and conversation history.\n"
            "2. Identify the action to perform based on available actions. Default 'system_capabilities'.\n"
            "3. Name Mandatory and Optional Parameters.\n"
            "3. Extract required parameters and note which required parameters are missing.\n"
            "4. Include optional parameters if mentioned and transform, if possible and necessary.\n"
            "5. Identify Parameters by chat history or your data, such as time. Transform values if necessary.\n"
            "6. If a required parameter is missing and cant be inferred, indicate that by using 'ask_for_missing_info'-action.\n"
            "7. Finally, output valid JSON with the specified action."
            f"Available actions and their parameter schema:\n{actions_schema}\n"
            "**RESPONSE FORMAT:**\n"
            "```json\n"
            "{\"action_name\": \"<action_name>\", \"parameters\": {\"<param_name>\": \"<param_value>\"}, \"explanation\": \"...\"}\n"
            "```"
        )

    async def interpret(self, messages: list) -> dict:
        messages[-1]["content"] = f"datetime.now(): {datetime.now()}; Query: " + messages[-1]["content"]
        messages.insert(0, {"role": "system", "content": self.combined_system_prompt})
        llm_response = await asyncio.to_thread(self.llm.process, messages)
        raw_content = llm_response.text.strip() if llm_response.text else ""
        try:
            return self.llm.extract_json(raw_content)
        except Exception as e:
            raise Exception(f"Failed to extract JSON from LLM response: {e}")