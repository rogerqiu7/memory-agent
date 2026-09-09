import os
import uuid
from datetime import datetime, timezone

import boto3
from boto3.dynamodb.conditions import Attr
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from model import model

load_dotenv()
session = boto3.Session(
    profile_name=os.getenv("AWS_PROFILE") or None,
    region_name=os.getenv("AWS_REGION", "us-east-1"),
)
table = session.resource("dynamodb").Table("agent-memory")
user_id = input("User ID: ").strip()
messages = []

while True:
    message = input("You: ")

    if message.lower() in {"exit", "quit"}:
        break

    memories = []
    response = table.scan(FilterExpression=Attr("user_id").eq(user_id))
    memories.extend(response.get("Items", []))

    while response.get("LastEvaluatedKey"):
        response = table.scan(
            FilterExpression=Attr("user_id").eq(user_id),
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        memories.extend(response.get("Items", []))

    memory_text = "\n".join(
        item["content"] for item in memories
    ) or "No previous memories."

    prompt = (
        "Use the relevant memories below to answer the user's message. "
        "Do not claim memories that are not present.\n\n"
        f"Memories:\n{memory_text}\n\nUser: {message}"
    )
    messages.append({"role": "user", "content": prompt})

    response = model.invoke(
        [
            HumanMessage(content=entry["content"])
            if entry["role"] == "user"
            else AIMessage(content=entry["content"])
            for entry in messages
        ]
    )
    answer = response.content
    messages.append({"role": "assistant", "content": answer})

    table.put_item(
        Item={
            "id": f"{user_id}#{uuid.uuid4()}",
            "user_id": user_id,
            "content": f"User: {message}\nAssistant: {answer}",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    print(f"Model: {answer}")
