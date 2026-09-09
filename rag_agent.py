from langchain_core.messages import AIMessage, HumanMessage

from model import model
from vector_store import retriever


messages = []

while True:
    message = input("You: ")

    if message.lower() in {"exit", "quit"}:
        break

    documents = retriever.invoke(message)
    context = "\n\n".join(document.page_content for document in documents)
    messages.append({
        "role": "user",
        "content": (
        "Answer the question using the context below.\n\n"
        f"Context:\n{context}\n\nQuestion: {message}"
        ),
    })

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

    print("Retrieved context:")
    print(context)
