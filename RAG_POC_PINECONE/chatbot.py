import os
import pinecone
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone  #  Ensure compatibility
from langchain_community.retrievers import BM25Retriever  #  BM25 Added

# Load Environment Variables
load_dotenv()

#  Pinecone API Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")

#  Initialize Pinecone Client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)  #  Get the actual index

#  Connect to Pinecone Vector Store using LangChain's Pinecone Wrapper
vector_store = LangchainPinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large")
)

#  BM25 Retriever (Lexical Search)
bm25_retriever = BM25Retriever.from_texts(["This is a placeholder text"])  # Ensure BM25 is populated

#  LLM Configuration
llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')

#  Create Retriever (Vector Search)
num_results = 5
vector_retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

#  Function to Process User Query with Hybrid Search
def stream_response(message, history):
    # Retrieve relevant chunks from BM25 & Pinecone
    bm25_docs = bm25_retriever.get_relevant_documents(message)  # BM25 Search
    vector_docs = vector_retriever.invoke(message)  # Vector Search

    #  Merge BM25 and Vector Results
    combined_docs = bm25_docs + vector_docs

    #  Remove duplicates based on content
    unique_docs = {}
    for doc in combined_docs:
        unique_docs[doc.page_content] = doc  # Overwrites duplicate texts

    final_docs = list(unique_docs.values())  # Get unique results

    #  Format knowledge base response
    knowledge = "\n\n".join([doc.page_content for doc in final_docs])

    #  Construct Prompt for LLM


    summary = "The retrieved knowledge provides relevant insights on the topic." if knowledge else "No relevant knowledge was found."

    rag_prompt = f"""
### **🤖 Intelligent AI Assistant**
Hello! I am an AI assistant designed to provide accurate answers based on retrieved knowledge.  
I prioritize factual correctness and structured responses to ensure clarity and depth.  

---

### **📝 Instructions:**
- **Prioritize retrieved knowledge** for responses.
- **Synthesize multiple sources** into a coherent answer when applicable.
- If the retrieved data lacks specific details, **provide general insights to maintain clarity**.
- Avoid speculation—indicate missing information politely.

---

### **🔍 User Query:**
🔹 **Question:** {message}  
🔹 **Conversation History:** {history}  

### **📚 Retrieved Knowledge (Hybrid: BM25 + Pinecone):**
{knowledge}

---

### **📌 Guidelines for Response:**
✅ **Provide structured, well-organized answers** (bullet points, examples, or step-by-step explanations).  
✅ **When applicable, include definitions, explanations, and real-world applications.**  
✅ **If relevant knowledge is limited, offer an informative yet concise response.**  
✅ **If no information is found, suggest how the user can refine their question.**  

---

### **💡 Example Response Structure:**
1️⃣ **Direct Answer**  
*"Based on retrieved knowledge, here’s the most relevant insight:..."*  

2️⃣ **In-Depth Explanation (if needed)**  
- **Concept Overview**: A simple, clear explanation.  
- **Key Details**: Essential points for deeper understanding.  
- **Examples (if available)**: Real-world scenarios for better comprehension.  

3️⃣ **Summary**  
_"To summarize, {summary}."_  

---

🚀 **Now, generate the most clear, complete, and well-structured response using the retrieved knowledge.**  
"""

    #  Stream LLM Response to Gradio UI
    partial_message = ""
    for response in llm.stream(rag_prompt):
        partial_message += response.content
        yield partial_message

#  Launch Gradio Chatbot UI
chatbot = gr.ChatInterface(
    stream_response,
    textbox=gr.Textbox(placeholder="Ask a question...", container=False, autoscroll=True, scale=7),
)

chatbot.launch()