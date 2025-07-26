# RAG LangChain Chroma with Memory Management

A comprehensive Retrieval-Augmented Generation (RAG) system built with LangChain and ChromaDB, now enhanced with conversational memory management for maintaining context across interactions.

## Features

### Core RAG Functionality
- Document embedding and retrieval using ChromaDB
- Integration with Google Generative AI (Gemini)
- Custom prompt templates with persona-based responses
- Source document tracking

### Memory Management (New!)
- **Multiple Memory Types**: Choose from different memory strategies based on your use case
- **Conversational Context**: Maintain context across multiple interactions
- **Memory Persistence**: Save and load conversation histories
- **Flexible Configuration**: Customize memory behavior for different scenarios

## Memory Types

### 1. Buffer Window Memory (`create_simple_conversational_rag`)
- **Best for**: Basic chatbots with limited memory requirements
- **How it works**: Keeps only the last N conversation exchanges
- **Use case**: Simple Q&A sessions, resource-constrained environments
- **Configuration**: Set `max_conversations` parameter

### 2. Summary Buffer Memory (`create_smart_conversational_rag`)
- **Best for**: Long conversations requiring intelligent context management
- **How it works**: Summarizes older conversations when token limit is reached
- **Use case**: Extended technical discussions, educational sessions
- **Configuration**: Set `max_tokens` parameter

### 3. Full Buffer Memory (`create_full_memory_rag`)
- **Best for**: Detailed analysis sessions requiring complete context
- **How it works**: Stores entire conversation history
- **Use case**: Research sessions, comprehensive analysis, debugging

## Quick Start

### Basic Usage

```python
from promptest import create_smart_conversational_rag

# Create a conversational RAG system with smart memory management
rag_system = create_smart_conversational_rag(max_tokens=1500)

# Start a conversation
response = rag_system.ask_question("What is machine learning?")
print(response['answer'])

# Continue the conversation - the system remembers previous context
followup = rag_system.ask_question("Can you give me a practical example of what you just explained?")
print(followup['answer'])

# Check conversation history
history = rag_system.get_conversation_history()
print(f"Total exchanges: {len(history)}")
```

### Advanced Usage

```python
from promptest import ConversationalRAGManager

# Create a custom conversational RAG manager
rag_system = ConversationalRAGManager(
    memory_type="summary",
    max_token_limit=2000,
    k_window=5
)

# Have a conversation
response = rag_system.ask_question("Explain microservices architecture")

# Save conversation for later
rag_system.save_conversation("my_conversation.json")

# Load conversation in a new session
new_rag_system = ConversationalRAGManager(memory_type="buffer")
new_rag_system.load_conversation("my_conversation.json")

# Continue where you left off
response = new_rag_system.ask_question("How does this relate to what we discussed about scalability?")
```

## Memory Management Methods

### Core Methods
- `ask_question(question)`: Ask a question with memory context
- `get_conversation_history()`: Retrieve formatted conversation history
- `clear_memory()`: Clear all conversation memory
- `get_memory_stats()`: Get statistics about memory usage

### Persistence Methods
- `save_conversation(filepath)`: Save conversation to JSON file
- `load_conversation(filepath)`: Load conversation from JSON file

## Example Scenarios

### 1. Customer Support Chatbot
```python
# Use window memory for basic support interactions
support_bot = create_simple_conversational_rag(max_conversations=5)
```

### 2. Educational Assistant
```python
# Use summary memory for extended learning sessions
tutor_bot = create_smart_conversational_rag(max_tokens=2000)
```

### 3. Research Assistant
```python
# Use full memory for comprehensive research sessions
research_assistant = create_full_memory_rag()
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Demo

Run the memory management demo to see all features in action:

```bash
python memory_demo.py
```

## Configuration

The system uses environment variables for API keys:
- `GOOGLE_API_KEY`: For Google Generative AI access

## Memory Performance Considerations

- **Buffer Memory**: Fastest but can become expensive with long conversations
- **Window Memory**: Good balance of performance and context
- **Summary Memory**: Most token-efficient for long conversations but requires additional LLM calls for summarization

## Files Structure

- `promptest.py`: Main conversational RAG implementation with memory management
- `memory_demo.py`: Comprehensive demonstration of memory features
- `rag_pipeline.py`: Original RAG pipeline (legacy)
- `requirements.txt`: Required Python packages