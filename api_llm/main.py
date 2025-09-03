from ollama import chat
from ollama import ChatResponse


YARD_TO_METER = 0.9144

def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert between yards and meters.

    Args:
        value (float): The numeric value to convert.
        from_unit (str): "meter" or "yard".
        to_unit (str): "meter" or "yard".

    Returns:
        float: Converted value.
    """
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    if from_unit not in {"meter", "yard"} or to_unit not in {"meter", "yard"}:
        raise ValueError("Units must be 'meter' or 'yard'")

    if from_unit == to_unit:
        return float(value)

    if from_unit == "yard" and to_unit == "meter":
        return float(value) * YARD_TO_METER

    return float(value) / YARD_TO_METER


tool = {
    "type": "function",
    "function": {
        "name": "convert_length",
        "description": "Convert a numeric length value between units (supports meter and yard).",
        "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The numeric value to convert."
                    },
                    "from_unit": {
                        "type": "string",
                        "enum": ["meter", "yard"],
                        "description": "The unit of the input value."
                    },
                    "to_unit": {
                        "type": "string",
                        "enum": ["meter", "yard"],
                        "description": "The target unit."
                    }
                },
                "required": ["value", "from_unit", "to_unit"],
                "additionalProperties": False
        }
    }
}

user_msg = {"role": "user", "content": "How many meters is 15 yards?"}
resp1 = chat(model="mistral:7b", messages=[user_msg], tools=[tool])


tc = (resp1.message.tool_calls or [None])[0]
messages = [user_msg, resp1.message]

if tc:
    args = tc.function.arguments            
    result = convert_length(**args)         

    tool_msg = {
        "role": "tool",
        "content": str(result),             
    }
    if getattr(tc, "id", None):
        tool_msg["tool_call_id"] = tc.id    
    messages.append(tool_msg)


resp2 = chat(model="mistral:7b", messages=messages)
print(resp2.message.content)