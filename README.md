# AkadVerse: YouTube Integration Recommender

**Tier 3 Neural Network / Deep Learning | Microservice Port: `8000`**

Automates personalized learning by generating context-aware YouTube playlists for students based on course registration, career paths, and academic performance.

## Table of Contents
- [What This Microservice Does](#what-this-microservice-does)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Getting Your API Key](#getting-your-api-key)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Testing with Swagger UI](#testing-with-swagger-ui)
- [Example Test Inputs](#example-test-inputs)
- [Understanding the Responses](#understanding-the-responses)
- [Generated Files](#generated-files)
- [Common Errors and Fixes](#common-errors-and-fixes)
- [Project Structure](#project-structure)

## What This Microservice Does

This service is a Tier 3 component of the AkadVerse AI-first e-learning platform. It lives within the My Learning module and serves Students.

**Core Workflow:**

1.  **Semantic Vectorization:** Converts video titles and descriptions into 384-dimensional dense vectors using a Transformer model.
2.  **Contextual Matching:** Compares student context (e.g., "struggling with calculus") against the vector library using FAISS L2 distance.
3.  **Event-Driven Curation:** Automatically triggers video fetching and indexing when it receives Kafka-style platform events.
4.  **Persistent Storage:** Syncs generated playlists to MongoDB Atlas for the student's personal dashboard.

**Key Design Decisions:**

*   **FAISS Indexing:** Built lazily and kept in memory (app.state) for sub-millisecond retrieval.
*   **Remedial Trigger:** Assessments scoring below 60 percent automatically trigger a remedial video search.
*   **Background Tasks:** Heavy ML operations and API calls are offloaded to background threads to ensure zero-latency response for the event bus.

## Architecture Overview

```
Platform Event (Kafka)
    |
    v
[FastAPI Webhook] (Python 3.10+)
    |
    v
[Sentence Transformer] (all-MiniLM-L6-v2)
    |
    v
[FAISS L2 Index] (Vector Similarity) <--- [YouTube Data API v3] (Metadata)
    |
    v
[MongoDB Atlas] (Playlist Sync)
```

| Component      | Technology             | Purpose                      |
| :------------- | :--------------------- | :--------------------------- |
| API Layer      | FastAPI                | Async REST endpoints         |
| AI Model       | Sentence Transformers  | Text-to-vector embedding     |
| Vector Store   | FAISS                  | Semantic similarity search   |
| Metadata       | YouTube Data API       | Video source retrieval       |
| DB             | MongoDB Atlas          | Permanent playlist storage   |

## Prerequisites

*   Python 3.10 or higher
*   `pip`
*   Google Cloud Project with YouTube Data API v3 enabled
*   MongoDB Atlas cluster
*   Active Internet connection (for model downloading and API calls)

## Getting Your API Key

1.  Go to [https://console.cloud.google.com/](https://console.cloud.google.com/)
2.  Enable the YouTube Data API v3.
3.  Navigate to Credentials and click Create Credentials -> API Key.
4.  Copy the key: you will paste it into your `.env` file.

## Installation

**Step 1 -- Set up your project folder**

Create dedicated folder: `akadverse-youtube-recommender/`

**Step 2 -- Create and activate a virtual environment**

*   **Windows:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
*   **macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

**Step 3 -- Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 -- Configure Environment**

Create a `.env` file in the root directory:

```env
YOUTUBE_API_KEY=your_youtube_api_key_here
MONGO_URI=your_mongodb_connection_string_here
```

## Running the Server

```bash
uvicorn recommender:app --host 127.0.0.1 --port 8000 --reload
```

Expected terminal output:

```
[INFO] AkadVerse YouTube Recommender API started.
[INFO] The video index is empty. Call POST /index-course to initialize.
```

## API Endpoints

### 1. `POST /index-course`

**What it does:** Fetches videos for a specific topic and builds the FAISS index.

| Field         | Required | Description                    |
| :------------ | :------- | :----------------------------- |
| `course_topic`| Yes      | The subject to fetch videos for|

### 2. `GET /recommend`

**What it does:** Returns recommendations based on text context and saves to MongoDB.

**Success response (200 OK):**

```json
{
  "message": "Playlist successfully generated and saved to MongoDB!",
  "data": {
    "student_id": "...",
    "recommendations": [...]
  }
}
```

### 3. `POST /webhook/event`

**What it does:** Simulates Kafka consumer for `course.registered`, `career_path.selected`, `business.registered`, and `assessment.completed`.

## Testing with Swagger UI

With the server running, open:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Example Test Inputs

**Test 1 -- Indexing**

`POST /index-course?course_topic=Machine Learning`

Expected: Successfully indexed 10 videos.

**Test 2 -- Remedial Trigger**

`POST /webhook/event`

```json
{
  "event_type": "assessment.completed",
  "student_id": "19MC022145",
  "payload": {
    "topic": "Calculus",
    "score": 45
  }
}
```

Expected: `[BACKGROUND TASK] Curating remedial resources...`

## Understanding the Responses

*   **Why is the distance score important?**
    A lower distance score in FAISS indicates a higher semantic similarity. Scores near 0.0 represent very close matches.

*   **The `[INFO] video index is empty` line**
    This is normal on startup. The microservice uses in-memory indexing for speed, so it requires at least one indexing call or event to populate.

## Generated Files

| File / Folder        | What it is              | Gitignore?              |
| :------------------- | :---------------------- | :---------------------- |
| `.env`               | Secret API keys         | Yes -- never commit     |
| `venv/`              | Virtual environment     | Yes                     |
| `__pycache__/`       | Python byte code        | Yes                     |

## Common Errors and Fixes

*   **Error: `YOUTUBE_API_KEY` is not set**
    Ensure your `.env` file is in the same folder as `recommender.py`.

*   **ModuleNotFoundError: No module named 'faiss'**
    Run `pip install faiss-cpu`.

## Project Structure

```
akadverse-youtube-recommender/
|-- recommender.py           # Main logic and API
|-- requirements.txt         # Dependencies
|-- .env                     # Secrets (Hidden)
|-- .gitignore               # Git exclusions
|-- README.md                # This documentation
```

## Part of the AkadVerse Platform

This microservice is Tier 3 in the AkadVerse AI architecture, operating within the My Learning module alongside:

*   Marketplace Recommender (Port 8021)
*   Insight Engine (Port 8010)

The playlist.generated event data stored in MongoDB is consumed by the Student Dashboard and the Tier 1 Faculty Virtual Assistant.

---

AkadVerse AI Architecture -- v1.0