from model import model


while True:
    message = input("You: ")

    if message.lower() in {"exit", "quit"}:
        break

    response = model.invoke(message)
    print(f"Model: {response.content}")
