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
- `stateless_agent.py`: Agent with no conversation memory.
- `conversation_agent.py`: Agent with JSON-style conversation history.
- `memory_agent.py`: Stores and retrieves episodic and semantic memory in DynamoDB.
- `rag_agent.py`: Placeholder for retrieval-augmented generation.

## References

1. Langley, P., Laird, J. E., & Rogers, S. (2009). *Cognitive architectures: Research issues and challenges*. Cognitive Systems Research, 10(2), 141–160.
2. Park, J. S., et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. UIST.
3. Shinn, N., et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 36.
