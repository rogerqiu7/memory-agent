# Cognitive Memory in AI Agents

**CS 6795: Cognitive Science, Fall 2026**
Roger Qiu | [yqiu353@gatech.edu](mailto:yqiu353@gatech.edu)

## Research Question

How do ideas from human memory and cognition influence the design of AI-agent memory systems and their ability to remember, retrieve, and use information across tasks and conversations?

## Project

This project compares four versions of the same LLM agent:

1. **Stateless agent**: receives only the current user message.
2. **Conversation agent**: remembers messages within the current conversation.
3. **Memory agent**: stores episodic conversations and semantic facts per user.
4. **RAG agent**: retrieves relevant information from an external knowledge source.

The comparison connects cognitive concepts such as working memory, episodic memory, semantic memory, learning, and information retrieval to practical LLM-agent architectures.

## Run and Test Locally

### 1. Create the environment

From the project directory, create and activate a virtual environment, then install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead.

### 2. Configure credentials

Create a `.env` file in the project root. The OpenAI key is required by every agent:

```dotenv
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-5.4-nano
AWS_REGION=us-east-1
AWS_PROFILE=your-aws-profile
```

`OPENAI_MODEL`, `AWS_REGION`, and `AWS_PROFILE` are optional. If `AWS_PROFILE` is omitted, boto3 uses the normal AWS credential-chain configuration. Do not commit `.env` or real keys.

The DynamoDB and RAG agents also require AWS access. Before running them:

- Create a DynamoDB table named `agent-memory` with `id` as its partition key (string). The AWS identity needs permission to scan, put, update, get, and delete items.
- Enable access to the Bedrock embedding model `amazon.titan-embed-text-v2:0` in `us-east-1` (or update `vector_store.py` if using another region). The AWS identity needs Bedrock model-invocation permission.
- The RAG data source is `data/realistic_restaurant_reviews.csv`. The first run of `vector_store.py` or `rag_agent.py` creates/populates the local `chroma_langchain_db/` store.

### 3. Run the agents interactively

Each command starts a terminal chat. Type `quit` or `exit` to stop it:

```bash
python stateless_agent.py       # current message only
python conversation_agent.py   # remembers during this process
python memory_agent.py         # persistent per-user memory in DynamoDB
python rag_agent.py            # restaurant-review retrieval with Chroma + Bedrock
```

For the memory agent, enter a user ID when prompted. Run it again with the same ID to test cross-session recall, then use a different ID to test user isolation. The conversation agent loses its history when the process ends; the stateless agent never sends earlier messages to the model.

### 4. Run the smoke tests and evaluation

These are standalone scripts, not pytest tests, so run them with Python:

```bash
python openai_test.py       # verifies the OpenAI model/key
python dynamodb_test.py    # writes, reads, updates, and deletes one temporary item
python evaluate_memory_agents.py
```

`openai_test.py` needs only OpenAI configuration. `dynamodb_test.py` needs the `agent-memory` table and AWS credentials. `evaluate_memory_agents.py` needs both OpenAI and AWS access because it runs all four architectures. It uses temporary, run-specific DynamoDB IDs and deletes the records during cleanup.

The evaluation checks within-session recall, cross-session recall, cross-user isolation, and external restaurant-review knowledge. It prints a summary and creates:

- `memory_agent_evaluation_results.json`: detailed answers and pass/fail results.
- `memory_agent_evaluation_summary.csv`: per-agent and per-category accuracy.

To save the live evaluation output while also viewing it:

```bash
python evaluate_memory_agents.py | tee evaluation_run.txt
```

`memory_test_results.txt` is a hand-written example transcript of the interactive agents; no script currently writes that exact file. Reproduce those scenarios by running the four commands above and capture a session with `tee`, for example `python conversation_agent.py | tee conversation_run.txt`. `evaluation_results.txt` is also a saved example of an earlier evaluation run.

`eval.py` is an older RAG evaluation draft that imports `chain` and `retriever` from an `app` module that is not present in this repository, so it will not run without additional refactoring. Use `evaluate_memory_agents.py` for the supported RAG evaluation.

### What each agent demonstrates

| Agent | Memory or retrieval behavior |
|---|---|
| Stateless | No conversation memory; only the current message is sent. |
| Conversation | In-process conversation history; lost when the process exits. |
| DynamoDB memory | Persistent conversation records separated by `user_id`. |
| RAG | Retrieves restaurant reviews from Chroma and supplies them as external context. |

## Evaluation

Each agent will receive the same tasks. I will compare recall accuracy, retrieval accuracy, and how well each memory design supports information use across interactions. The goal is not to reproduce human memory exactly, but to study where these systems resemble human memory and retrieval processes.

## DynamoDB Memory

This is how the stored conversations look in DynamoDB:

| id | content | created_at | user_id |
|---|---|---|---|
| `1#29171e60-a137-429a-8150-61dab365e5f3` | User: I like pizza<br>Assistant: Nice! What kind of pizza do you like? Pepperoni, ... | 2026-09-09T20:11:25.254924+00:00 | `1` |
| `1#fca6c9baf-9ac-44ce-a070-66a0cc038335` | User: i also like cats<br>Assistant: That's awesome, you've got good taste | 2026-09-09T20:11:32.582562+00:00 | `1` |
| `1#58e07ef8-7201-4708-812a-5f0c978582ff` | User: what do you remember about me<br>Assistant: From our chat, I remember that you like... | 2026-09-09T20:12:07.678097+00:00 | `1` |
| `1#4d075f41-d405-4864-bdd8-ca1de66c3190` | User: what do you know about me<br>Assistant: From our chat, I know that you like... | 2026-09-09T20:12:55.995473+00:00 | `1` |
| `1#d9219a5c-1125-4104-b6cd-4281ef6c33da` | User: also my name is roger<br>Assistant: From our chat, I know you like pizza... | 2026-09-09T20:13:01.557903+00:00 | `1` |
| `2#39f17c5d-ed5b-4405-a5d1-a6de3803b6bb` | User: what do you know about me<br>Assistant: I don't know anything about you... | 2026-09-09T20:13:50.862464+00:00 | `2` |
| `2#a0b7c063-96ad-4a4b-95df-c93a5f2299fe` | User: my name is mia, i like video games and working out and reading<br>Assistant: ... | 2026-09-09T20:14:15.081575+00:00 | `2` |

Each conversation is stored as one item. The agent retrieves items using `user_id`, so User 1 and User 2 have separate memories.

## Current Files

- `model.py`: Shared `ChatOpenAI` model configuration.
- `stateless_agent.py`: Interactive agent with no conversation memory.
- `conversation_agent.py`: Interactive agent with in-process conversation history.
- `memory_agent.py`: Interactive agent that stores and retrieves conversations in DynamoDB.
- `rag_agent.py`: Interactive retrieval-augmented agent using the local Chroma store.
- `vector_store.py`: Loads restaurant reviews from CSV and creates the Chroma retriever.
- `evaluate_memory_agents.py`: Full comparison harness for all four agent architectures.
- `openai_test.py`: Minimal OpenAI connectivity/model test.
- `dynamodb_test.py`: DynamoDB CRUD smoke test using a temporary item.
- `eval.py`: Older evaluation draft that currently references the missing `app` module.
- `memory_test_results.txt`, `evaluation_results.txt`: Example transcripts and expected-result notes.
- `memory_agent_evaluation_results.json`, `memory_agent_evaluation_summary.csv`: Saved evaluation outputs.
- `data/realistic_restaurant_reviews.csv`: Restaurant-review source data for RAG.
- `requirements.txt`: Pinned Python dependencies.

## References

1. Langley, P., Laird, J. E., & Rogers, S. (2009). *Cognitive architectures: Research issues and challenges*. Cognitive Systems Research, 10(2), 141–160.
2. Park, J. S., et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. UIST.
3. Shinn, N., et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 36.
