from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
chatbot = pipeline("text-generation", model="mistralai/Mistral-Large-Instruct-2407")
chatbot(messages)


model_id = "mistralai/Mistral-Large-Instruct-2407"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def get_current_weather(location: str, format: str):
    """
    Get the current weather

    Args:
        location: The city and state, e.g. San Francisco, CA
        format: The temperature unit to use. Infer this from the users location. (choices: ["celsius", "fahrenheit"])
    """
    pass

conversation = [{"role": "user", "content": "What's the weather like in Paris?"}]
tools = [get_current_weather]

# format and tokenize the tool use prompt 
inputs = tokenizer.apply_chat_template(
            conversation,
            tools=tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
)

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

inputs.to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1000)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
