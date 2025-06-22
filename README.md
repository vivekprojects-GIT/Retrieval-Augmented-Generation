# 🧠 Retrieval-Augmented Generation (RAG) Project Stack

This repository contains a modular stack of Retrieval-Augmented Generation (RAG) pipelines built using multiple vector databases and deployed across different cloud environments. Each module explores optimized retrieval strategies, grounding methods, and generation techniques using both proprietary and open-source models.

---

## ✨ Features

* 🔎 Vector store support: Pinecone, ChromaDB, Weaviate, Qdrant
* 🧠 Model flexibility: OpenAI, Azure OpenAI, HuggingFace, VertexAI PaLM
* ☁️ Cloud-native deployment across Azure, AWS, GCP
* 📥 Unified ingestion, chunking, embedding, and indexing pipelines
* 📊 Built-in evaluation metrics for relevance, accuracy, and latency
* 🔄 Support for LoRA fine-tuning and QLoRA compression
* ⚡ Real-time inference APIs via FastAPI, Lambda, and Docker containers

---

## 🌐 Deployment Targets

| Stack            | Vector DB | LLM Source          | Deployment Platform      |
| ---------------- | --------- | ------------------- | ------------------------ |
| `pinecone-azure` | Pinecone  | Azure OpenAI        | Azure Container Apps     |
| `chromadb-aws`   | ChromaDB  | Bedrock/HuggingFace | AWS Lambda + API Gateway |
| `weaviate-gcp`   | Weaviate  | VertexAI (PaLM 2)   | Google Cloud Run / GKE   |
| `qdrant-local`   | Qdrant    | Local HF models     | Docker Compose           |

---

## 🚀 Getting Started

### . Clone the Repository

```bash
git clone https://github.com/your-username/rag-stack.git
cd rag-stack
```

### . Set Up a Virtual Environment


## 🔍 RAG Pipeline Overview

Each pipeline includes:

* **Document Loader** (PDF, HTML, DOCX, Markdown)
* **Text Chunking & Preprocessing**
* **Embedding Generation** (OpenAI, SentenceTransformers, Cohere)
* **Vector Indexing** (Pinecone, ChromaDB, etc.)
* **Retriever + Generator Integration**
* **API Exposure for Inference**
* **Evaluation Scripts** (precision, recall, hallucination)

---

## 📈 Evaluation Metrics

* Retrieval relevance (Top-k accuracy, MRR)
* Generation quality (BLEU, ROUGE, hallucination rate)
* Latency benchmarks across cloud platforms
* Infra cost profiling (per query and batch inference)

---

## 📚 Reference Papers

* [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Lewis et al., 2020
* [Instruct-RAG: Augmenting Instruction-Following Models with Retrieval](https://arxiv.org/abs/2310.07704) — Mialon et al., 2023
* [FiD: Fusion-in-Decoder for Open-Domain QA](https://arxiv.org/abs/2007.01282) — Izacard & Grave, 2020
* [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
* [Self-RAG: Faithful and Self-Reflective RAG](https://arxiv.org/abs/2308.03281) — Asai et al., 2023

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit and push your changes
4. Open a Pull Request

