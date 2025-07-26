"""
Demonstration of Memory Management in RAG Pipeline using LangChain

This script shows how to use the different memory types available in the
conversational RAG system for maintaining context across interactions.
"""

from promptest import (
    ConversationalRAGManager,
    create_simple_conversational_rag,
    create_smart_conversational_rag,
    create_full_memory_rag,
    demo_conversational_rag
)

def basic_memory_demo():
    """Basic demonstration of conversational memory."""
    print("=== Basic Memory Demo ===\n")
    
    # Create a simple conversational RAG with window memory (last 5 exchanges)
    rag_system = create_simple_conversational_rag(max_conversations=5)
    
    # Simulate a conversation
    conversation = [
        "What is machine learning?",
        "Can you explain supervised learning in more detail?",
        "How is this different from unsupervised learning?",
        "Can you give me examples of each type we discussed?",
        "Which approach would you recommend for predicting house prices?",
        "Based on our conversation, what should I learn next?"
    ]
    
    for i, question in enumerate(conversation, 1):
        print(f"Question {i}: {question}")
        response = rag_system.ask_question(question)
        print(f"Answer: {response['answer'][:150]}...\n")
        
        # Show memory stats
        stats = rag_system.get_memory_stats()
        print(f"Memory: {stats['total_exchanges']} exchanges, Type: {stats['memory_type']}")
        print("-" * 50)

def advanced_memory_demo():
    """Advanced demonstration with summary memory."""
    print("\n=== Advanced Memory Demo (Summary Memory) ===\n")
    
    # Create a smart conversational RAG with summary memory
    rag_system = create_smart_conversational_rag(max_tokens=1000)
    
    # Simulate a longer technical conversation
    tech_conversation = [
        "What are microservices?",
        "How do microservices compare to monolithic architecture?",
        "What are the main challenges when implementing microservices?",
        "How do you handle data consistency across microservices?",
        "What patterns would you recommend for inter-service communication?",
        "Considering everything we've discussed, when would you NOT use microservices?"
    ]
    
    for i, question in enumerate(tech_conversation, 1):
        print(f"Question {i}: {question}")
        response = rag_system.ask_question(question)
        print(f"Answer: {response['answer'][:200]}...\n")
        
        # Show memory stats
        stats = rag_system.get_memory_stats()
        print(f"Memory Stats: {stats}")
        print("-" * 50)

def memory_persistence_demo():
    """Demonstrate saving and loading conversation history."""
    print("\n=== Memory Persistence Demo ===\n")
    
    # Create a RAG system
    rag_system = create_full_memory_rag()
    
    # Have a short conversation
    questions = [
        "What is the difference between REST and GraphQL?",
        "Which one is better for mobile applications?"
    ]
    
    print("Initial conversation:")
    for question in questions:
        response = rag_system.ask_question(question)
        print(f"Q: {question}")
        print(f"A: {response['answer'][:100]}...\n")
    
    # Save the conversation
    save_path = "sample_conversation.json"
    rag_system.save_conversation(save_path)
    print(f"Conversation saved to {save_path}")
    
    # Create a new RAG system and load the conversation
    new_rag_system = create_full_memory_rag()
    if new_rag_system.load_conversation(save_path):
        print("Conversation loaded successfully!")
        
        # Ask a follow-up question that references the previous conversation
        followup = "Based on what we discussed, which would you choose for a real-time chat app?"
        response = new_rag_system.ask_question(followup)
        print(f"\nFollow-up Q: {followup}")
        print(f"Follow-up A: {response['answer'][:150]}...")
        
        # Show that the history is preserved
        history = new_rag_system.get_conversation_history()
        print(f"\nTotal conversation history: {len(history)} exchanges")

def compare_memory_types():
    """Compare different memory types side by side."""
    print("\n=== Memory Types Comparison ===\n")
    
    # Create different memory types
    systems = {
        "Window Memory (3 exchanges)": create_simple_conversational_rag(max_conversations=3),
        "Summary Memory (500 tokens)": create_smart_conversational_rag(max_tokens=500),
        "Full Buffer Memory": create_full_memory_rag()
    }
    
    # Same set of questions for all systems
    questions = [
        "What is artificial intelligence?",
        "How does machine learning relate to what you just explained?",
        "Can you explain deep learning in the context of our discussion?",
        "What are the practical applications of what we've covered?",
        "How would you summarize everything we've discussed so far?"
    ]
    
    for system_name, rag_system in systems.items():
        print(f"\n--- {system_name} ---")
        
        for i, question in enumerate(questions, 1):
            response = rag_system.ask_question(question)
            print(f"Q{i}: {question}")
            print(f"A{i}: {response['answer'][:100]}...")
            
            stats = rag_system.get_memory_stats()
            print(f"Stats: {stats}\n")
        
        print("=" * 60)

if __name__ == "__main__":
    print("LangChain Conversational RAG Memory Management Demo")
    print("=" * 60)
    
    # Run different demos
    try:
        basic_memory_demo()
        advanced_memory_demo()
        memory_persistence_demo()
        compare_memory_types()
        
        print("\n=== Running Full Demo ===")
        # Uncomment the line below to run the comprehensive demo
        # demo_conversational_rag()
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("Make sure you have all required dependencies installed and your vector database is set up.")
    
    print("\nDemo completed! You can now use the ConversationalRAGManager class")
    print("in your own applications for memory-aware conversational AI.")
