"""
Conversational intelligence layer.

This package will contain:
- types: shared data structures (Intent, ResolvedQuery, ChatAnswer, etc.)
- intent_parser: parse natural-language questions into intents
- parameter_inference: map intents + metadata to structured PSES queries
- retrieval_pipeline: call core queries and assemble results
- narrative_engine: AI prompts and narrative generation with validation
- orchestrator: main "brain" used by the UI to handle each message
- state: conversation state for multi-turn flows (optional)
"""
