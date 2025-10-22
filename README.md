# Project Flow: AI Hybrid Travel Assistant

This document explains the step-by-step execution flow of the project in a simple, easy-to-follow manner.

---

## Step-by-Step Process

1. **User runs `hybrid_chat.py`**  
   The program starts, initializes configurations, and prompts the user for a travel-related question.

2. **Load configurations from `config.py`**  
   API keys, model names, and index details are loaded into memory for further processing.

3. **Generate embedding using Hugging Face model**  
   The user query is converted into a numerical vector (768 dimensions) for semantic search.

4. **Search Pinecone vector database**  
   The query embedding is compared with stored travel data embeddings to find similar matches.

5. **Fetch related context from Neo4j graph**  
   For each relevant result, related destinations, activities, and relationships are retrieved.

6. **Combine data for hybrid reasoning**  
   The system merges:
   - User’s query  
   - Pinecone semantic matches  
   - Neo4j relationship insights  
   - LangChain conversation memory (if any)

7. **Send final prompt to the LLM (Hugging Face / OpenAI)**  
   A refined prompt is constructed and sent to the large language model for intelligent response generation.

8. **Display final response to the user**  
   The output (itinerary, travel tips, etc.) is printed on the terminal.

9. **Store conversation history using LangChain Memory**  
   Previous queries and responses are retained for contextual, multi-turn conversations.

10. **Handle next user query**  
    When the user asks a follow-up question, the system reuses memory to deliver more relevant answers.

---

## High-Level Data Flow
User → Hugging Face Embedding → Pinecone Search → Neo4j Query → Hybrid Data Fusion → LLM → Response → LangChain Memory

---

## Summary

This hybrid system integrates:
- **Vector search (Pinecone)** for semantic relevance  
- **Graph search (Neo4j)** for relational reasoning  
- **LLM (Hugging Face)** for natural language generation  
- **LangChain Memory** for multi-turn contextual awareness  

Together, they create an intelligent and memory-aware AI travel assistant.
