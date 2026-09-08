# Cognitive Memory in AI Agents

**CS 6795: Cognitive Science, Fall 2026**
Roger Qiu | [yqiu353@gatech.edu](mailto:yqiu353@gatech.edu)

## Research Question

How do ideas from human memory and cognition influence the design of AI-agent memory systems and their ability to remember, retrieve, and use information across tasks and conversations?

## Project

This project compares four versions of the same LLM agent:

1. **Stateless agent**: receives only the current user message.
2. **Conversation agent**: remembers messages within the current conversation.
3. **Semantic agent**: stores and retrieves facts across conversations.
4. **RAG agent**: retrieves relevant information from an external knowledge source.

The comparison connects cognitive concepts such as working memory, episodic memory, semantic memory, learning, and information retrieval to practical LLM-agent architectures.

## Evaluation

Each agent will receive the same tasks. I will compare recall accuracy, retrieval accuracy, and how well each memory design supports information use across interactions. The goal is not to reproduce human memory exactly, but to study where these systems resemble human memory and retrieval processes.

## Current Files

- `model.py`: Shared `ChatOpenAI` model configuration.
- `stateless_agent.py`: Agent with no conversation memory.
- `conversation_agent.py`: Agent with JSON-style conversation history.
- `semantic_agent.py`: Placeholder for semantic memory.
- `rag_agent.py`: Placeholder for retrieval-augmented generation.

## References

1. Langley, P., Laird, J. E., & Rogers, S. (2009). *Cognitive architectures: Research issues and challenges*. Cognitive Systems Research, 10(2), 141–160.
2. Park, J. S., et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. UIST.
3. Shinn, N., et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 36.
