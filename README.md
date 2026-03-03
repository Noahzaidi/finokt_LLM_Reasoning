<p align="center">
  <h1 align="center">🧠 OCR Output Reasoning Engine</h1>
  <p align="center">
    <strong>GPU-accelerated LLM microservice for intelligent OCR post-processing</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#quickstart">Quickstart</a> •
    <a href="#api-reference">API</a> •
    <a href="#contributing">Contributing</a>
  </p>
</p>

---

## Overview

**OCR Output Reasoning Engine** is a privacy-first, on-premise LLM inference microservice designed to act as an intelligent fallback layer for OCR pipelines. When an OCR system produces low-confidence results   missing fields, ambiguous classifications, or conflicting extractions   this service applies LLM-based reasoning to resolve them.

It consumes structured OCR tokens and context, reasons over them using a quantized LLM, and returns structured JSON with extracted fields, confidence scores, and human-readable reasoning traces.

> **Built for regulated industries.** No data leaves the server. No external API calls. Fully on-premise.

---

## Features

- 🔒 **Privacy-First**   All inference runs locally on GPU. Zero external calls. No PII in logs.
- ⚡ **GPU-Accelerated**   Powered by [vLLM](https://github.com/vllm-project/vllm) for high-throughput token generation on NVIDIA GPUs.
- 🎯 **Structured Output**   Uses vLLM's guided JSON decoding to guarantee valid, schema-conformant responses. No hallucinated keys. No malformed JSON.
- 🧩 **RAG-Enhanced**   Accepts field exemplars and normalization rules as context, enabling the model to learn per-tenant conventions without retraining.
- 🔐 **JWT-Authenticated**   All extraction endpoints require a valid JWT Bearer token.
- 📊 **Confidence Scoring**   Every extracted field includes a confidence score and reasoning trace for auditability.
- 🏗️ **Stateless**   No database, no sessions, no caching. Pure request → response inference.
- 🛡️ **Graceful Degradation**   Returns structured error responses on failure; never crashes the upstream pipeline.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│               Your OCR Service                       │
│  Document ingestion, OCR, layout understanding,      │
│  field extraction, quality scoring                   │
└──────────────┬───────────────────────────────────────┘
               │  HTTP POST (when confidence is low)
               ▼
┌──────────────────────────────────────────────────────┐
│          OCR Reasoning Engine (this repo)             │
│  GPU inference microservice                          │
│  Structured field extraction + classification        │
│  via quantized LLM + vLLM guided JSON                │
│                                                      │
│  ┌─────────────┐       ┌──────────────────┐          │
│  │  FastAPI     │──────▶│  vLLM Engine     │          │
│  │  Port 8200   │       │  Port 8100 (lo)  │          │
│  └─────────────┘       └──────────────────┘          │
└──────────────────────────────────────────────────────┘
```

The service exposes a FastAPI wrapper on port **8200** that internally communicates with a vLLM inference server on port **8100** (localhost only, never exposed).

---

## Getting Started

### Prerequisites

| Requirement | Minimum |
|---|---|
| NVIDIA GPU | T4 (16 GB VRAM) or better |
| CUDA | 12.x |
| Python | 3.12+ |
| OS | Ubuntu 22.04+ |

### Step 1: Clone and install dependencies

```bash
git clone https://github.com/Noahzaidi/finokt_LLM_Reasoning.git
cd finokt_LLM_Reasoning

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Download model weights

The service uses [Qwen2.5-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ), a 4-bit quantized model (~5.6 GB).

```bash
mkdir -p models
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ \
  --local-dir ./models/qwen2.5-7b-awq
```

### Step 3: Configure environment

Generate your secrets and create the `.env` file:

```bash
cp .env.example .env

# Generate secure tokens
echo "JWT_SECRET=$(openssl rand -hex 32)" >> .env
echo "INTERNAL_SERVICE_TOKEN=$(openssl rand -hex 32)" >> .env
```

Then note your `INTERNAL_SERVICE_TOKEN` value   you'll need it to call the API:

```bash
grep INTERNAL_SERVICE_TOKEN .env
```

<details>
<summary><strong>All .env variables</strong></summary>

| Variable | Description | Default |
|---|---|---|
| `FAI_LLM_MODEL` | Path to model weights directory | `./models/qwen2.5-7b-awq` |
| `FAI_LLM_VLLM_BASE_URL` | vLLM internal endpoint | `http://127.0.0.1:8100/v1` |
| `FAI_LLM_MAX_TOKENS` | Max tokens per response | `512` |
| `FAI_LLM_TEMPERATURE` | Sampling temperature (0 = deterministic) | `0.0` |
| `FAI_LLM_GPU_MEMORY_UTILIZATION` | Fraction of GPU VRAM to use | `0.85` |
| `FAI_LLM_MAX_MODEL_LEN` | Max context window | `2048` |
| `FAI_LLM_MAX_NUM_SEQS` | Max concurrent sequences | `2` |
| `JWT_SECRET` | Secret key for JWT validation | *required* |
| `INTERNAL_SERVICE_TOKEN` | Shared token for service-to-service auth | *required* |
| `LOG_LEVEL` | Logging level | `INFO` |

</details>

### Step 4: Start the services

You need **two terminals** (both with the venv activated).

**Terminal 1   Start the vLLM inference engine:**

```bash
source venv/bin/activate
python3 -m vllm.entrypoints.openai.api_server \
  --model ./models/qwen2.5-7b-awq \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --host 127.0.0.1 \
  --port 8100
```

Wait until you see `Uvicorn running on http://127.0.0.1:8100` (takes ~60-90 seconds for model loading).

**Terminal 2   Start the FastAPI application:**

```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8200
```

You should see:
```
INFO [fai_llm.main] Starting OCR Reasoning Engine...
INFO [fai_llm.main] vLLM connection verified: model loaded
INFO:     Uvicorn running on http://0.0.0.0:8200
```

### Step 5: Verify it works

**Health check (no auth required):**

```bash
curl http://localhost:8200/api/v1/health
```

Expected:
```json
{"status": "healthy", "service": "ocr-reasoning-engine", "vllm": {"vllm_status": "healthy", "models_loaded": 1}}
```

**Test extraction (requires token):**

```bash
# Set your token (from .env)
export TOKEN=$(grep INTERNAL_SERVICE_TOKEN .env | cut -d= -f2)

curl -X POST http://localhost:8200/api/v1/extract \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-001",
    "tenant_id": "tenant-demo",
    "document_id": "doc-001",
    "document_type_guess": "NATIONAL_ID",
    "classification_confidence": 0.90,
    "ocr_tokens": [
      {"text": "Nom:", "bbox": {"x": 50, "y": 120, "w": 45, "h": 16}, "confidence": 0.93, "page": 1, "line": 1, "block": 1},
      {"text": "DUPONT", "bbox": {"x": 220, "y": 120, "w": 100, "h": 18}, "confidence": 0.94, "page": 1, "line": 1, "block": 1},
      {"text": "Date de naissance:", "bbox": {"x": 50, "y": 170, "w": 160, "h": 16}, "confidence": 0.90, "page": 1, "line": 2, "block": 1},
      {"text": "15", "bbox": {"x": 220, "y": 170, "w": 20, "h": 18}, "confidence": 0.72, "page": 1, "line": 2, "block": 1},
      {"text": "O3", "bbox": {"x": 245, "y": 170, "w": 20, "h": 18}, "confidence": 0.41, "page": 1, "line": 2, "block": 1},
      {"text": "1990", "bbox": {"x": 270, "y": 170, "w": 45, "h": 18}, "confidence": 0.88, "page": 1, "line": 2, "block": 1}
    ],
    "required_fields": ["last_name", "date_of_birth"],
    "missing_fields": ["date_of_birth"],
    "locked_fields": {"last_name": "DUPONT"},
    "rag_context": {
      "field_exemplars": [
        {"field_key": "date_of_birth", "original_value": "15 O3 1990", "corrected_value": "1990-03-15"}
      ],
      "normalization_rules": [
        {"field_key": "date_of_birth", "output_format": "ISO-8601 (YYYY-MM-DD)"}
      ]
    }
  }'
```

The LLM will reason over the OCR tokens (recognizing `O3` as a misread of `03`) and return a structured JSON response with the extracted `date_of_birth`, confidence score, and reasoning trace.

### Interactive API Docs

Once the app is running, explore the full API at:
- **Swagger UI:** http://localhost:8200/docs
- **ReDoc:** http://localhost:8200/redoc

---

## API Reference

### `POST /api/v1/extract`

The primary extraction endpoint. Accepts structured OCR context and returns reasoned field extractions.

**Headers:**
```
Authorization: Bearer <INTERNAL_SERVICE_TOKEN>
Content-Type: application/json
```

**Request Body:**
```json
{
  "request_id": "uuid",
  "tenant_id": "uuid",
  "document_id": "uuid",
  "document_type_guess": "PASSPORT_MA",
  "classification_confidence": 0.6,
  "ocr_tokens": [
    {
      "text": "MARTIN",
      "bbox": {"x": 100, "y": 200, "w": 80, "h": 20},
      "confidence": 0.92,
      "page": 1,
      "line": 3,
      "block": 1
    }
  ],
  "required_fields": ["last_name", "first_name", "date_of_birth", "nationality"],
  "missing_fields": ["date_of_birth", "nationality"],
  "locked_fields": {"last_name": "MARTIN"},
  "rag_context": {
    "field_exemplars": [
      {
        "field_key": "date_of_birth",
        "original_value": "23 AVRIL 1985",
        "corrected_value": "1985-04-23"
      }
    ],
    "normalization_rules": [
      {"field_key": "date_of_birth", "output_format": "ISO-8601"}
    ]
  }
}
```

**Response:**
```json
{
  "request_id": "uuid",
  "document_id": "uuid",
  "document_type": "PASSPORT_MA",
  "classification_confidence": 0.91,
  "extracted_fields": {
    "date_of_birth": {
      "value": "1985-04-23",
      "confidence": 0.88,
      "source": "llm",
      "reasoning": "Found '23 AVRIL 1985' in token block 3, normalized to ISO-8601"
    },
    "nationality": {
      "value": "MAR",
      "confidence": 0.95,
      "source": "llm",
      "reasoning": "MRZ line 1 contains MAR at positions 14-16"
    }
  },
  "model_version": "qwen2.5-7b-awq",
  "latency_ms": 1240
}
```

### `GET /api/v1/health`

Health check endpoint (no authentication required).

Returns GPU status, model readiness, and service uptime.

### `GET /api/v1/model-info`

Returns model version and capabilities. Requires JWT authentication.

---

## Key Design Decisions

### Why Guided JSON?

In compliance-sensitive environments, free-form LLM output is a liability. The model might hallucinate field names, return malformed JSON, or include unexpected data. By using vLLM's `guided_json` parameter with a Pydantic schema, the decoding process is **structurally constrained**   the model can only produce tokens that conform to the schema. This is the primary hallucination guard.

### Why Structural RAG?

Instead of vector embeddings, this service uses **structural retrieval**: deterministic SQL lookups by `tenant_id + document_type + field_key`. This is faster, fully auditable, and requires no vector database infrastructure. Past analyst corrections are surfaced as few-shot examples in the prompt.

### Why Locked Fields?

When the upstream OCR system has already extracted a field with high confidence, that field is passed as `locked_fields`. The LLM is instructed never to override these   preventing unnecessary second-guessing and ensuring consistency between the OCR and LLM layers.

---

## Project Structure

```
.
├── app/
│   ├── main.py                   # FastAPI app entry point
│   ├── routers/
│   │   └── extraction.py         # POST /api/v1/extract
│   ├── services/
│   │   ├── llm_service.py        # vLLM call + response parsing
│   │   └── rag_service.py        # RAG context assembly
│   ├── models/
│   │   ├── request.py            # ExtractionRequest Pydantic model
│   │   └── response.py           # ExtractionResponse Pydantic model
│   └── middleware/
│       └── auth.py               # JWT validation
├── models/                       # Model weights (git-ignored)
│   └── qwen2.5-7b-awq/
├── systemd/                      # systemd service files
├── logs/                         # Runtime logs (git-ignored)
├── .env.example                  # Environment template
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Performance

| Metric | Value |
|---|---|
| Inference speed | ~28 tokens/sec (T4) |
| p95 extraction latency | < 3 seconds |
| GPU memory usage | ~12.8 GB / 15 GB |
| Model size on disk | 5.58 GB |

---

## Security

- **No external API calls**   The model runs entirely on local GPU. No data leaves the server.
- **No PII in logs**   Logs contain only `request_id`, `tenant_id`, `document_id`, latency, and status codes.
- **JWT authentication**   All extraction endpoints require a valid Bearer token.
- **Localhost-only vLLM**   The inference engine binds to `127.0.0.1`, inaccessible from the network.
- **Tenant isolation**   Every request is scoped by `tenant_id`. No cross-tenant data leakage.

---

## Contributing

Contributions are welcome! This project focuses specifically on the **OCR output reasoning** layer   improving how LLMs can be used to resolve ambiguous, low-confidence, or conflicting OCR extractions.

### Areas of Interest

- 📄 **Document type support**   Adding prompt templates for new document types
- 🧪 **Evaluation benchmarks**   Building test suites for extraction accuracy
- 🔧 **Model compatibility**   Testing with other quantized instruction-following models
- 📐 **Prompt engineering**   Optimizing extraction prompts for specific field types
- 🌍 **Multilingual support**   Improving extraction for non-Latin scripts and RTL documents

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add: your feature description'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please ensure your PR:
- Does not introduce external API calls or data exfiltration
- Does not log PII or document content
- Maintains guided JSON enforcement for all extraction endpoints
- Includes tests for any new functionality

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE). You are free to use, share, and adapt this work for non-commercial purposes with attribution.

---

## Acknowledgements

- [vLLM](https://github.com/vllm-project/vllm)   High-throughput LLM serving engine
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5)   Instruction-following language model
- [FastAPI](https://fastapi.tiangolo.com/)   Modern Python web framework
- [DocTR](https://github.com/mindee/doctr)   Deep learning for OCR

---

<p align="center">
  <sub>Built for privacy. Built for compliance. Built to reason.</sub>
</p>
