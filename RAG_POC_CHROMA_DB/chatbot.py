from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever  # BM25 Added
import gradio as gr

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the model
llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

#  Set up BM25 Retriever
bm25_retriever = BM25Retriever.from_texts(vector_store.get()["documents"])  #  BM25 Added

# Set up the vectorstore to be the retriever
num_results = 5
vector_retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# call this function for every message added to the chatbot
def stream_response(message, history):
    #print(f"Input: {message}. History: {history}\n")

    # retrieve the relevant chunks based on the question asked
    bm25_docs = bm25_retriever.get_relevant_documents(message)  #  BM25 Retrieval
    vector_docs = vector_retriever.invoke(message)  # Vector Search
    docs = bm25_docs + vector_docs  #  Combine BM25 + Vector Results

    # add all the chunks to 'knowledge'
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content + "\n\n"

    # make the call to the LLM (including prompt)
    if message is not None:
        partial_message = ""

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

        #print(rag_prompt)

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
chatbot.launch()