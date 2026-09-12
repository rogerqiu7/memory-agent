"""
Evaluation harness for the four memory-agent architectures used in the
CS 6795 Cognitive Memory in AI Agents project.

What it tests:
1. Within-session recall
2. Cross-session recall
3. Cross-user persistent recall / isolation
4. External knowledge retrieval

It recreates all four agents in this one file:
- Stateless agent
- Conversation-memory agent
- DynamoDB persistent-memory agent
- RAG agent

Outputs:
- memory_agent_evaluation_results.json
- memory_agent_evaluation_summary.csv

Requirements:
    pip install python-dotenv boto3 langchain-core langchain-openai \
        langchain-aws langchain-chroma

Environment:
    OPENAI_API_KEY=...
    OPENAI_MODEL=gpt-5.4-nano   # optional
    AWS_REGION=us-east-1        # optional
    AWS_PROFILE=...             # optional

AWS:
    This script expects the existing DynamoDB table:
        agent-memory

The DynamoDB tests use unique temporary user IDs and delete the test records
at the end so they do not mix with your normal project data.
"""

import csv
import json
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from boto3.dynamodb.conditions import Attr
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE") or None
DYNAMODB_TABLE = "agent-memory"

RESULTS_JSON = Path("memory_agent_evaluation_results.json")
SUMMARY_CSV = Path("memory_agent_evaluation_summary.csv")

RUN_ID = uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------
# Shared LLM
# ---------------------------------------------------------------------

model = ChatOpenAI(model=OPENAI_MODEL)


def to_messages(history: list[dict[str, str]]):
    """Convert our simple history format into LangChain message objects."""
    converted = []
    for entry in history:
        if entry["role"] == "user":
            converted.append(HumanMessage(content=entry["content"]))
        else:
            converted.append(AIMessage(content=entry["content"]))
    return converted


# ---------------------------------------------------------------------
# Agent 1: Stateless
# ---------------------------------------------------------------------

class StatelessAgent:
    name = "Stateless"

    def send(self, user_id: str, message: str) -> dict[str, Any]:
        response = model.invoke(message)
        return {
            "answer": response.content,
            "retrieved_context": "",
        }

    def reset_session(self, user_id: str) -> None:
        pass

    def cleanup(self) -> None:
        pass


# ---------------------------------------------------------------------
# Agent 2: Conversation memory only
# ---------------------------------------------------------------------

class ConversationAgent:
    name = "Conversation"

    def __init__(self):
        self.histories: dict[str, list[dict[str, str]]] = defaultdict(list)

    def send(self, user_id: str, message: str) -> dict[str, Any]:
        history = self.histories[user_id]
        history.append({"role": "user", "content": message})

        response = model.invoke(to_messages(history))

        history.append({"role": "assistant", "content": response.content})

        return {
            "answer": response.content,
            "retrieved_context": "",
        }

    def reset_session(self, user_id: str) -> None:
        self.histories[user_id] = []

    def cleanup(self) -> None:
        self.histories.clear()


# ---------------------------------------------------------------------
# Agent 3: DynamoDB persistent memory
# ---------------------------------------------------------------------

class DynamoDBMemoryAgent:
    name = "DynamoDB Persistent Memory"

    def __init__(self):
        session = boto3.Session(
            profile_name=AWS_PROFILE,
            region_name=AWS_REGION,
        )
        self.table = session.resource("dynamodb").Table(DYNAMODB_TABLE)
        self.histories: dict[str, list[dict[str, str]]] = defaultdict(list)
        self.written_ids: list[str] = []

    def _load_memories(self, user_id: str) -> list[dict[str, Any]]:
        memories = []

        response = self.table.scan(
            FilterExpression=Attr("user_id").eq(user_id)
        )
        memories.extend(response.get("Items", []))

        while response.get("LastEvaluatedKey"):
            response = self.table.scan(
                FilterExpression=Attr("user_id").eq(user_id),
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            memories.extend(response.get("Items", []))

        memories.sort(key=lambda item: item.get("created_at", ""))
        return memories

    def send(self, user_id: str, message: str) -> dict[str, Any]:
        memories = self._load_memories(user_id)

        memory_text = "\n\n".join(
            item["content"] for item in memories
        ) or "No previous memories."

        prompt = (
            "Use the relevant memories below to answer the user's message. "
            "Do not claim memories that are not present.\n\n"
            f"Memories:\n{memory_text}\n\n"
            f"User: {message}"
        )

        history = self.histories[user_id]
        history.append({"role": "user", "content": prompt})

        response = model.invoke(to_messages(history))
        answer = response.content

        history.append({"role": "assistant", "content": answer})

        item_id = f"eval-{RUN_ID}-{user_id}-{uuid.uuid4().hex}"

        self.table.put_item(
            Item={
                "id": item_id,
                "user_id": user_id,
                "content": f"User: {message}\nAssistant: {answer}",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.written_ids.append(item_id)

        return {
            "answer": answer,
            "retrieved_context": memory_text,
        }

    def reset_session(self, user_id: str) -> None:
        # This clears only the current conversation.
        # DynamoDB memory remains, which is exactly what we want to test.
        self.histories[user_id] = []

    def cleanup(self) -> None:
        # Delete only temporary rows created by this evaluation run.
        for item_id in self.written_ids:
            self.table.delete_item(Key={"id": item_id})
        self.written_ids.clear()
        self.histories.clear()


# ---------------------------------------------------------------------
# Agent 4: RAG
# ---------------------------------------------------------------------

RAG_DOCUMENTS = [
    Document(
        page_content=(
            "Innovative toppings. Their 'Experimental' menu section is amazing. "
            "Just tried one with maple-glazed brussels sprouts, pancetta, and "
            "smoked gouda - sounds weird but it was incredible. They're not "
            "afraid to take risks."
        )
    ),
    Document(
        page_content=(
            "Perfect crust-to-topping ratio. Some places overload toppings so "
            "the crust gets soggy, others are too sparse. This place gets it "
            "perfect - enough toppings to get some in every bite, but not so "
            "much that the structural integrity is compromised."
        )
    ),
    Document(
        page_content=(
            "Mediocre at best. Nothing terrible but nothing special either. "
            "The crust was okay, toppings were standard, and service was fine. "
            "It's the kind of place you go when you're in the area, but "
            "wouldn't make a special trip for."
        )
    ),
    Document(
        page_content=(
            "Toppings slide right off. The ratio of sauce to cheese was so off "
            "that all the toppings slid off with the first bite. Ended up "
            "eating what was essentially dough with sauce while the cheese and "
            "toppings formed a pile on my plate."
        )
    ),
    Document(
        page_content=(
            "Suspiciously uniform toppings. All the vegetable toppings were "
            "cut in precisely identical shapes and sizes, suggesting they came "
            "pre-cut from a food service provider rather than being prepared "
            "fresh in-house as claimed. Tasted fine but felt deceived by the "
            "marketing."
        )
    ),
]


class RAGAgent:
    name = "RAG"

    def __init__(self):
        embedding_kwargs = {
            "model_id": "amazon.titan-embed-text-v2:0",
            "region_name": AWS_REGION,
        }
        if AWS_PROFILE:
            embedding_kwargs["credentials_profile_name"] = AWS_PROFILE

        embeddings = BedrockEmbeddings(**embedding_kwargs)

        self.vector_store = Chroma(
            collection_name=f"memory_eval_{RUN_ID}",
            embedding_function=embeddings,
        )
        self.vector_store.add_documents(RAG_DOCUMENTS)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        self.histories: dict[str, list[dict[str, str]]] = defaultdict(list)

    def send(self, user_id: str, message: str) -> dict[str, Any]:
        documents = self.retriever.invoke(message)
        context = "\n\n".join(doc.page_content for doc in documents)

        prompt = (
            "Answer the user's question. Use the external context when it is "
            "relevant. If the question is personal or conversational, use the "
            "conversation history instead of inventing facts.\n\n"
            f"External context:\n{context}\n\n"
            f"User: {message}"
        )

        history = self.histories[user_id]
        history.append({"role": "user", "content": prompt})

        response = model.invoke(to_messages(history))
        history.append({"role": "assistant", "content": response.content})

        return {
            "answer": response.content,
            "retrieved_context": context,
        }

    def reset_session(self, user_id: str) -> None:
        self.histories[user_id] = []

    def cleanup(self) -> None:
        self.histories.clear()
        try:
            self.vector_store.delete_collection()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------

MEMORY_TESTS = [
    {
        "id": "name",
        "statement": "My name is Roger.",
        "question": "What is my name?",
        "required": [["roger"]],
    },
    {
        "id": "color",
        "statement": "My favorite color is teal.",
        "question": "What is my favorite color?",
        "required": [["teal"]],
    },
    {
        "id": "cat",
        "statement": "My cat is named Tucker.",
        "question": "What is my cat's name?",
        "required": [["tucker"]],
    },
    {
        "id": "project",
        "statement": "The project I am working on is called Atlas.",
        "question": "What is the name of the project I am working on?",
        "required": [["atlas"]],
    },
    {
        "id": "food",
        "statement": "My favorite food is pizza.",
        "question": "What is my favorite food?",
        "required": [["pizza"]],
    },
]


EXTERNAL_KNOWLEDGE_TESTS = [
    {
        "id": "experimental_toppings",
        "question": (
            "According to the restaurant reviews, what toppings were on the "
            "experimental pizza that was described as incredible?"
        ),
        "required": [
            ["brussels", "brussels sprouts"],
            ["pancetta"],
            ["gouda", "smoked gouda"],
        ],
    },
    {
        "id": "sliding_toppings",
        "question": (
            "According to the restaurant reviews, what happened because the "
            "sauce-to-cheese ratio was off?"
        ),
        "required": [
            ["slid off", "slide right off", "toppings slid"],
        ],
    },
    {
        "id": "uniform_toppings",
        "question": (
            "According to the restaurant reviews, why were the uniformly cut "
            "vegetable toppings suspicious?"
        ),
        "required": [
            ["pre-cut", "precut", "food service provider"],
        ],
    },
    {
        "id": "ratio",
        "question": (
            "According to the restaurant reviews, how was the crust-to-topping "
            "ratio described?"
        ),
        "required": [
            ["perfect", "every bite"],
        ],
    },
    {
        "id": "special_trip",
        "question": (
            "According to the restaurant reviews, was the mediocre restaurant "
            "worth making a special trip for?"
        ),
        "required": [
            ["wouldn't", "would not", "not worth", "no"],
            ["special trip"],
        ],
    },
]


# ---------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------

def matches_required_terms(text: str, required: list[list[str]]) -> bool:
    """
    Each inner list contains acceptable alternatives.
    At least one alternative from every inner list must appear.
    """
    lowered = text.lower()

    for alternatives in required:
        if not any(term.lower() in lowered for term in alternatives):
            return False

    return True


def add_result(
    results: list[dict[str, Any]],
    *,
    agent: str,
    category: str,
    test_id: str,
    question: str,
    required: list[list[str]],
    answer: str,
    passed: bool,
    retrieved_context: str = "",
    retrieval_passed: bool | None = None,
):
    results.append(
        {
            "agent": agent,
            "category": category,
            "test_id": test_id,
            "question": question,
            "required_terms": required,
            "answer": answer,
            "passed": passed,
            "retrieved_context": retrieved_context,
            "retrieval_passed": retrieval_passed,
        }
    )


# ---------------------------------------------------------------------
# Evaluation suites
# ---------------------------------------------------------------------

def run_within_session_tests(agent, results):
    for index, test in enumerate(MEMORY_TESTS):
        user_id = f"{RUN_ID}-{agent.name}-within-{index}"
        agent.reset_session(user_id)

        # Tell the agent a fact.
        agent.send(user_id, test["statement"])

        # Ask about it without resetting the session.
        output = agent.send(user_id, test["question"])
        passed = matches_required_terms(output["answer"], test["required"])

        add_result(
            results,
            agent=agent.name,
            category="within_session_recall",
            test_id=test["id"],
            question=test["question"],
            required=test["required"],
            answer=output["answer"],
            passed=passed,
            retrieved_context=output["retrieved_context"],
        )


def run_cross_session_tests(agent, results):
    for index, test in enumerate(MEMORY_TESTS):
        user_id = f"{RUN_ID}-{agent.name}-cross-{index}"
        agent.reset_session(user_id)

        # First session: tell it the fact.
        agent.send(user_id, test["statement"])

        # Simulate closing and reopening the conversation.
        agent.reset_session(user_id)

        # Second session: ask for the fact.
        output = agent.send(user_id, test["question"])
        passed = matches_required_terms(output["answer"], test["required"])

        add_result(
            results,
            agent=agent.name,
            category="cross_session_recall",
            test_id=test["id"],
            question=test["question"],
            required=test["required"],
            answer=output["answer"],
            passed=passed,
            retrieved_context=output["retrieved_context"],
        )


def run_user_isolation_test(agent, results):
    """
    Two separate users provide different names.
    After a session reset, the agent is asked to recall each user's name.

    A persistent memory architecture should remember the correct name for each
    user without mixing them together.
    """
    user_a = f"{RUN_ID}-{agent.name}-user-a"
    user_b = f"{RUN_ID}-{agent.name}-user-b"

    agent.reset_session(user_a)
    agent.reset_session(user_b)

    agent.send(user_a, "My name is Roger.")
    agent.send(user_b, "My name is Mia.")

    agent.reset_session(user_a)
    agent.reset_session(user_b)

    output_a = agent.send(user_a, "What is my name?")
    output_b = agent.send(user_b, "What is my name?")

    passed_a = matches_required_terms(output_a["answer"], [["roger"]])
    passed_b = matches_required_terms(output_b["answer"], [["mia"]])

    add_result(
        results,
        agent=agent.name,
        category="cross_user_persistent_recall",
        test_id="user_a",
        question="What is my name?",
        required=[["roger"]],
        answer=output_a["answer"],
        passed=passed_a,
        retrieved_context=output_a["retrieved_context"],
    )

    add_result(
        results,
        agent=agent.name,
        category="cross_user_persistent_recall",
        test_id="user_b",
        question="What is my name?",
        required=[["mia"]],
        answer=output_b["answer"],
        passed=passed_b,
        retrieved_context=output_b["retrieved_context"],
    )


def run_external_knowledge_tests(agent, results):
    for index, test in enumerate(EXTERNAL_KNOWLEDGE_TESTS):
        # Use a fresh user/session for every external-knowledge question.
        user_id = f"{RUN_ID}-{agent.name}-external-{index}"
        agent.reset_session(user_id)

        output = agent.send(user_id, test["question"])

        answer_passed = matches_required_terms(
            output["answer"],
            test["required"],
        )

        retrieval_passed = None
        if agent.name == "RAG":
            retrieval_passed = matches_required_terms(
                output["retrieved_context"],
                test["required"],
            )

        add_result(
            results,
            agent=agent.name,
            category="external_knowledge",
            test_id=test["id"],
            question=test["question"],
            required=test["required"],
            answer=output["answer"],
            passed=answer_passed,
            retrieved_context=output["retrieved_context"],
            retrieval_passed=retrieval_passed,
        )


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in results:
        grouped[(row["agent"], row["category"])].append(row)

    summary = []

    for (agent, category), rows in grouped.items():
        passed = sum(1 for row in rows if row["passed"])
        total = len(rows)

        retrieval_rows = [
            row for row in rows
            if row["retrieval_passed"] is not None
        ]

        retrieval_accuracy = ""
        if retrieval_rows:
            retrieval_passed = sum(
                1 for row in retrieval_rows if row["retrieval_passed"]
            )
            retrieval_accuracy = round(
                100 * retrieval_passed / len(retrieval_rows), 1
            )

        summary.append(
            {
                "agent": agent,
                "category": category,
                "passed": passed,
                "total": total,
                "accuracy_percent": round(100 * passed / total, 1),
                "retrieval_accuracy_percent": retrieval_accuracy,
            }
        )

    return summary


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print("EVALUATION SUMMARY")
    print("=" * 78)

    current_agent = None

    for row in summary_rows:
        if row["agent"] != current_agent:
            current_agent = row["agent"]
            print(f"\n{current_agent}")

        line = (
            f"  {row['category']}: "
            f"{row['passed']}/{row['total']} "
            f"({row['accuracy_percent']}%)"
        )

        if row["retrieval_accuracy_percent"] != "":
            line += (
                f" | retrieval accuracy: "
                f"{row['retrieval_accuracy_percent']}%"
            )

        print(line)


def save_results(
    results: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "model": OPENAI_MODEL,
                "results": results,
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "agent",
                "category",
                "passed",
                "total",
                "accuracy_percent",
                "retrieval_accuracy_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved detailed results to: {RESULTS_JSON}")
    print(f"Saved summary table to:     {SUMMARY_CSV}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    agents = [
        StatelessAgent(),
        ConversationAgent(),
        DynamoDBMemoryAgent(),
        RAGAgent(),
    ]

    results: list[dict[str, Any]] = []

    try:
        for agent in agents:
            print("\n" + "=" * 78)
            print(f"TESTING: {agent.name}")
            print("=" * 78)

            print("Running within-session recall tests...")
            run_within_session_tests(agent, results)

            print("Running cross-session recall tests...")
            run_cross_session_tests(agent, results)

            print("Running cross-user persistent recall test...")
            run_user_isolation_test(agent, results)

            print("Running external knowledge tests...")
            run_external_knowledge_tests(agent, results)

        summary_rows = summarize(results)
        print_summary(summary_rows)
        save_results(results, summary_rows)

    finally:
        print("\nCleaning up temporary evaluation data...")
        for agent in agents:
            try:
                agent.cleanup()
            except Exception as error:
                print(f"Cleanup warning for {agent.name}: {error}")


if __name__ == "__main__":
    main()
