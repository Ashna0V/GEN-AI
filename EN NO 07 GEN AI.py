import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

model = model.to("cpu")
model.eval()

print("Model loaded successfully!")
print("Device: CPU")

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. Answer questions clearly and accurately."
    }
]

def chatbot(user_input):
    global messages

    messages.append({
        "role": "user",
        "content": user_input
    })

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    messages.append({
        "role": "assistant",
        "content": response
    })

    return response

print("\n" + "=" * 50)
print("QWEN AI CHATBOT")
print("=" * 50)
print("Running on CPU")
print("Type 'clear' to clear the conversation.")
print("Type 'exit' to stop.")
print("=" * 50)

while True:
    user_input = input("\nYou: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye!")
        break

    if user_input.lower() == "clear":
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Answer questions clearly and accurately."
            }
        ]
        print("Bot: Conversation cleared.")
        continue

    print("Bot:", chatbot(user_input))