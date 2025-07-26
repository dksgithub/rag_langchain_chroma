from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from typing import List, Optional, Dict, Any
import json

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from typing import List, Optional, Dict, Any
import json

class ConversationalRAGManager:
    """
    A comprehensive conversational RAG system with multiple memory management options.
    Supports different types of memory for various use cases.
    """
    
    def __init__(self, memory_type: str = "buffer", max_token_limit: int = 2000, k_window: int = 5):
        """
        Initialize the conversational RAG manager.
        
        Args:
            memory_type: Type of memory to use ('buffer', 'summary', 'window')
            max_token_limit: Maximum tokens for summary memory
            k_window: Window size for window memory
        """
        self.memory_type = memory_type
        self.max_token_limit = max_token_limit
        self.k_window = k_window
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.75)
        self.retriever = self._build_retriever()
        self.memory = self._initialize_memory()
        self.chain = self._build_conversational_chain()
        
    def _build_retriever(self):
        """Build and return the retriever for document search."""
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(collection_name="rag_collection", embedding_function=embedding_function)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        return retriever
    
    def _initialize_memory(self):
        """Initialize memory based on the specified type."""
        if self.memory_type == "buffer":
            # Simple buffer memory - stores all conversation history
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        elif self.memory_type == "summary":
            # Summary buffer memory - summarizes old conversations when they exceed token limit
            return ConversationSummaryBufferMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                max_token_limit=self.max_token_limit
            )
        elif self.memory_type == "window":
            # Window memory - keeps only the last k interactions
            return ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                k=self.k_window
            )
        else:
            raise ValueError(f"Unsupported memory type: {self.memory_type}")
    
    def _build_conversational_chain(self):
        """Build the conversational retrieval chain with custom prompt."""
        # Custom prompt template that includes conversation history
        custom_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""
You are Alex, a seasoned Software Architect with 15+ years of experience in designing and building scalable systems. You have a warm, patient personality and genuinely enjoy helping others understand complex technical concepts.

## Your Personality:
- *Gentle & Patient*: You never rush or dismiss questions, no matter how basic they seem
- *Explanatory*: You break down complex concepts into digestible pieces  
- *Encouraging*: You celebrate learning and progress, making people feel comfortable asking questions
- *Practical*: You provide real-world examples and actionable advice
- *Humble*: You acknowledge when you don't know something and suggest ways to find answers
- *Memory-Aware*: You remember previous conversations and can reference them naturally

## Your Communication Style:
- Use clear, jargon-free language when possible
- When technical terms are necessary, explain them simply
- Provide context and background before diving into details
- Use analogies and metaphors to make complex concepts relatable
- Ask clarifying questions to ensure you understand the user's needs
- Offer multiple perspectives or approaches when appropriate
- Reference previous conversations when relevant to provide continuity

## Your Approach to Answers:
1. *Listen First*: Understand what the user is really asking, considering conversation history
2. *Context Setting*: Provide necessary background information, building on previous discussions
3. *Step-by-Step*: Break down complex solutions into manageable steps
4. *Examples*: Use concrete examples to illustrate concepts
5. *Follow-up*: Suggest next steps or related topics they might find helpful
6. *Continuity*: Connect current questions to previous conversations when appropriate

## Sample Response Style:
"That's a great question about [topic]! Let me walk you through this step by step..."
"I can see why this might be confusing. Let me explain it in a simpler way..."
"Think of it like [analogy] - it helps make the concept clearer..."
"Based on our earlier discussion about [previous topic], this relates to..."

Remember: Your goal is to help people truly understand and feel confident about the concepts they're learning, while maintaining conversational continuity.

Previous Conversation History:
{chat_history}

Retrieved Documents:
{context}

Current Question: {question}

Answer:
"""
        )
        
        # Build conversational retrieval chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            verbose=True
        )
        
        return chain
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get a response with memory context.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary containing answer, source documents, and conversation history
        """
        try:
            result = self.chain({"question": question})
            
            return {
                "answer": result["answer"],
                "source_documents": result.get("source_documents", []),
                "chat_history": self.get_conversation_history(),
                "memory_type": self.memory_type
            }
        except Exception as e:
            return {
                "error": f"An error occurred: {str(e)}",
                "answer": "I apologize, but I encountered an error processing your question. Please try again.",
                "source_documents": [],
                "chat_history": self.get_conversation_history(),
                "memory_type": self.memory_type
            }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get formatted conversation history."""
        try:
            if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                history = []
                messages = self.memory.chat_memory.messages
                
                for i in range(0, len(messages), 2):
                    if i + 1 < len(messages):
                        human_msg = messages[i]
                        ai_msg = messages[i + 1]
                        
                        history.append({
                            "human": human_msg.content if hasattr(human_msg, 'content') else str(human_msg),
                            "ai": ai_msg.content if hasattr(ai_msg, 'content') else str(ai_msg)
                        })
                
                return history
            return []
        except Exception:
            return []
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()
    
    def save_conversation(self, filepath: str):
        """Save conversation history to a file."""
        history = self.get_conversation_history()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "memory_type": self.memory_type,
                "conversation_history": history,
                "timestamp": str(json.datetime.now() if hasattr(json, 'datetime') else "N/A")
            }, f, indent=2, ensure_ascii=False)
    
    def load_conversation(self, filepath: str):
        """Load conversation history from a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            history = data.get("conversation_history", [])
            
            # Rebuild memory with loaded history
            self.memory.clear()
            for exchange in history:
                self.memory.chat_memory.add_user_message(exchange["human"])
                self.memory.chat_memory.add_ai_message(exchange["ai"])
                
            return True
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the current memory usage."""
        stats = {
            "memory_type": self.memory_type,
            "total_exchanges": len(self.get_conversation_history())
        }
        
        if self.memory_type == "summary":
            stats["max_token_limit"] = self.max_token_limit
            if hasattr(self.memory, 'moving_summary_buffer'):
                stats["current_summary"] = getattr(self.memory, 'moving_summary_buffer', '')
        elif self.memory_type == "window":
            stats["window_size"] = self.k_window
            
        return stats

def build_retriever():
    """Legacy function for backward compatibility."""
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(collection_name="rag_collection", embedding_function=embedding_function)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever

def build_qa_chain_with_custom_prompt():
    # Get retriever
    retriever = build_retriever()
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.75)
    
    # Define custom prompt template
    #You are a helpful assistant. Use the following retrieved documents to answer the query 
    #concisely and accurately. If the documents don't provide enough information, 
    #say so explicitly.
    prompt_template = """
    You are Alex, a seasoned Software Architect with 15+ years of experience in designing and building scalable systems. You have a warm, patient personality and genuinely enjoy helping others understand complex technical concepts.
    ## Your Personality:
    - *Gentle & Patient*: You never rush or dismiss questions, no matter how basic they seem
    - *Explanatory*: You break down complex concepts into digestible pieces  
    - *Encouraging*: You celebrate learning and progress, making people feel comfortable asking questions
    - *Practical*: You provide real-world examples and actionable advice
    - *Humble*: You acknowledge when you don't know something and suggest ways to find answers

    ## Your Communication Style:
    - Use clear, jargon-free language when possible
    - When technical terms are necessary, explain them simply
    - Provide context and background before diving into details
    - Use analogies and metaphors to make complex concepts relatable
    - Ask clarifying questions to ensure you understand the user's needs
    - Offer multiple perspectives or approaches when appropriate

    ## Your Approach to Answers:
    1. *Listen First*: Understand what the user is really asking
    2. *Context Setting*: Provide necessary background information
    3. *Step-by-Step*: Break down complex solutions into manageable steps
    4. *Examples*: Use concrete examples to illustrate concepts
    5. *Follow-up*: Suggest next steps or related topics they might find helpful

    ## Sample Response Style:
    "That's a great question about [topic]! Let me walk you through this step by step..."

    "I can see why this might be confusing. Let me explain it in a simpler way..."

    "Think of it like [analogy] - it helps make the concept clearer..."

    Remember: Your goal is to help people truly understand and feel confident about the concepts they're learning.
    

    Retrieved Documents:
    {context}

    Query: {question}

    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    
    # Build RetrievalQA chain with custom prompt
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

# Utility functions for easy access to different memory types

def create_simple_conversational_rag(max_conversations: int = 10) -> ConversationalRAGManager:
    """
    Create a simple conversational RAG with buffer window memory.
    Good for basic chatbots with limited conversation history.
    
    Args:
        max_conversations: Maximum number of conversation exchanges to remember
    """
    return ConversationalRAGManager(memory_type="window", k_window=max_conversations)

def create_smart_conversational_rag(max_tokens: int = 2000) -> ConversationalRAGManager:
    """
    Create a smart conversational RAG with summary memory.
    Good for long conversations that need intelligent summarization.
    
    Args:
        max_tokens: Maximum tokens before summarization kicks in
    """
    return ConversationalRAGManager(memory_type="summary", max_token_limit=max_tokens)

def create_full_memory_rag() -> ConversationalRAGManager:
    """
    Create a conversational RAG that remembers everything.
    Good for detailed analysis sessions where full context is needed.
    """
    return ConversationalRAGManager(memory_type="buffer")

# Example usage and testing function
def demo_conversational_rag():
    """
    Demonstration of the conversational RAG system with different memory types.
    """
    print("=== Conversational RAG with Memory Management Demo ===\n")
    
    # Create different types of conversational RAG systems
    systems = {
        "Simple (Window Memory)": create_simple_conversational_rag(max_conversations=3),
        "Smart (Summary Memory)": create_smart_conversational_rag(max_tokens=1000),
        "Full (Buffer Memory)": create_full_memory_rag()
    }
    
    sample_questions = [
        "What is machine learning?",
        "Can you explain the concept you just mentioned in more detail?",
        "How does this relate to what we discussed earlier?",
        "What would you recommend as next steps based on our conversation?"
    ]
    
    for system_name, rag_system in systems.items():
        print(f"\n--- Testing {system_name} ---")
        
        for i, question in enumerate(sample_questions, 1):
            print(f"\nQ{i}: {question}")
            
            try:
                response = rag_system.ask_question(question)
                print(f"A{i}: {response['answer'][:200]}...")  # Truncate for demo
                
                # Show memory stats
                stats = rag_system.get_memory_stats()
                print(f"Memory Stats: {stats}")
                
            except Exception as e:
                print(f"Error: {e}")
        
        # Clear memory for next system test
        rag_system.clear_memory()
        print("\n" + "="*50)

if __name__ == "__main__":
    # Example of how to use the new conversational RAG system
    
    # Create a smart conversational RAG system
    rag_system = create_smart_conversational_rag(max_tokens=1500)
    
    # Example conversation
    questions = [
        "What is the difference between REST and GraphQL APIs?",
        "Can you give me an example of when to use each approach?",
        "Based on what we discussed, which would you recommend for a mobile app?"
    ]
    
    print("Starting conversation with Alex...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"User: {question}")
        response = rag_system.ask_question(question)
        print(f"Alex: {response['answer']}\n")
        
        # Show conversation history length
        history = rag_system.get_conversation_history()
        print(f"[Conversation exchanges so far: {len(history)}]\n")
    
    # Save the conversation
    # rag_system.save_conversation("conversation_log.json")
    
    # Get memory statistics
    stats = rag_system.get_memory_stats()
    print(f"Final Memory Stats: {stats}")