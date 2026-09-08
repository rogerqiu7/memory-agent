from langchain_core.messages import AIMessage, HumanMessage

from model import model


messages = []

while True:
    message = input("You: ")

    if message.lower() in {"exit", "quit"}:
        break

    messages.append({"role": "user", "content": message})

    response = model.invoke(
        [
            HumanMessage(content=entry["content"])
            if entry["role"] == "user"
            else AIMessage(content=entry["content"])
            for entry in messages
        ]
    )

    messages.append({"role": "assistant", "content": response.content})
    print(f"Model: {response.content}")
