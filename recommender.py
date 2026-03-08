import os #For interacting with the operating system, such as accessing environment variables.
import faiss #Facebook AI Similarity Search library for efficient similarity search and clustering of dense vectors.
import numpy as np #For numerical operations, especially for handling the vector embeddings and preparing them for FAISS.
from dotenv import load_dotenv #For loading environment variables from a .env file, such as API keys.
from sentence_transformers import SentenceTransformer #A library that provides pre-trained models to convert sentences into dense vector representations (embeddings).
from googleapiclient.discovery import build #Google API client library to interact with YouTube Data API for fetching video metadata based on search queries.
from pymongo import MongoClient #MongoDB client for connecting to a MongoDB database, which can be used for storing video metadata and user interactions in a production environment.
from fastapi import FastAPI, BackgroundTasks, HTTPException #FastAPI is a modern, fast (high-performance) web framework for building APIs with Python. BackgroundTasks allows us to run tasks in the background without blocking the main thread, which is useful for operations like fetching videos or building the FAISS index that might take some time. HTTPException is used to return proper HTTP error responses.
from pydantic import BaseModel #Pydantic is used for data validation and settings management using Python type annotations. BaseModel is the base class for creating data models that can be used to define the structure of request bodies in FastAPI endpoints.
import uvicorn #ASGI server for running FastAPI applications. It is used to serve the API when we run the script directly.

# Load environment variables (your API key)
load_dotenv()
# YouTube Data API key used by `googleapiclient`.
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# MongoDB connection URI, which can be used to connect to a MongoDB database
MONGO_URI = os.getenv('MONGO_URI')

# Validate required environment variables at startup
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY is not set. Please add it to your .env file.")
if not MONGO_URI:
    raise ValueError("MONGO_URI is not set. Please add it to your .env file.")

# Connect to MongoDB Atlas
print("Connecting to MongoDB Atlas...")
mongo_client = MongoClient(MONGO_URI)
# Create a database called 'akadverse_db'
db = mongo_client["akadverse_db"]
# Create a collection (like a table) called 'student_playlists'
playlists_collection = db["student_playlists"]

# 1. Initialize the Open-Source Embedding Model
# This model converts text into a 384-dimensional vector as specified in the architecture doc.
print("Loading the Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_youtube_videos(query, max_results=5):
    """
    Fetch video metadata for a given topic.

    Returns a list of dictionaries with:
    - `id`: YouTube video ID
    - `title`: video title
    - `text`: title + description (combined text used for embeddings)
    """
    try:
        # Build a YouTube API client once per request cycle.
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

        print(f"Fetching videos for topic: {query}...")
        # Ask YouTube Search API for videos that match the topic string.
        request = youtube.search().list(
            part="snippet",
            maxResults=max_results,
            q=query,
            type="video"
        )
        response = request.execute()
    except Exception as e:
        print(f"[ERROR] Failed to fetch YouTube videos for '{query}': {e}")
        return []

    # Extract video titles and descriptions to build our "metadata"
    videos = []
    for item in response['items']:
        title = item['snippet']['title']
        desc = item['snippet']['description']
        video_id = item['id']['videoId']
        # Combine title and description for richer contextual embeddings
        videos.append({
            "id": video_id,
            "text": f"{title}. {desc}",
            "title": title
        })
    return videos

def build_faiss_index(video_data):
    """
    Convert each video's text into an embedding and store vectors in a FAISS index.

    Uses IndexFlatL2, which performs exact nearest-neighbor search with
    Euclidean (L2) distance.
    """
    # Extract just the text from our video data list
    texts = [video['text'] for video in video_data]
    
    # Generate 384-dimensional embeddings for all videos at once
    print("Generating embeddings for video library...")
    embeddings = model.encode(texts)
    
    # FAISS requires numpy arrays in float32 format
    embeddings = np.array(embeddings).astype('float32')
    embeddings = np.ascontiguousarray(embeddings)  # Ensure C-contiguous memory layout

    # Initialize a FAISS index using L2 distance (Euclidean distance)
    dimension = embeddings.shape[1]  # This will be 384
    index = faiss.IndexFlatL2(dimension)
    
    # Load all embeddings into the in-memory index.
    index.add(embeddings)
    print(f"Successfully added {index.ntotal} videos to the FAISS index.")
    
    return index

def recommend_videos(student_context, index, video_data, top_k=3):
    """
    Takes a student's context (e.g., 'struggling with calculus limits'),
    embeds it, and finds the closest matching videos in the FAISS index.
    """
    # Convert the student's context into a vector
    context_vector = model.encode([student_context])
    context_vector = np.array(context_vector).astype('float32').reshape(1, -1)
    
    # Search the FAISS index for the 'top_k' nearest neighbors
    distances, indices = index.search(context_vector, top_k)
    
    print(f"\n--- Recommendations for context: '{student_context}' ---")
    for i in range(top_k):
        # FAISS returns row/column arrays because search supports batch queries.
        video_idx = indices[0][i]
        match_distance = distances[0][i]
        matched_video = video_data[video_idx]
        print(f"{i+1}. {matched_video['title']} (Distance score: {match_distance:.2f})")

def process_course_registration(student_id: str, course_topic: str, top_k: int = 3):
    """
    This function runs in the background. It simulates what happens
    when the YouTube Recommender catches a 'course.registered' event.
    """
    print(f"[BACKGROUND TASK] Starting setup for student {student_id} on topic '{course_topic}'")
    
    # 1. Fetch videos and build the index (acting as the setup)
    videos = fetch_youtube_videos(course_topic, max_results=5)
    if not videos:
        print(f"[BACKGROUND TASK] No videos fetched for '{course_topic}'. Aborting.")
        return
    index = build_faiss_index(videos)
    
    # 2. Simulate an immediate recommendation for their first lesson
    # We will use the course topic as the initial context
    context_vector = model.encode([f"Introduction to {course_topic}"])
    context_vector = np.array(context_vector).astype('float32').reshape(1, -1)
    distances, indices = index.search(context_vector, top_k)
    
    results = []
    for i in range(top_k):
        video_idx = indices[0][i]
        matched_video = videos[video_idx]
        results.append({
            "title": matched_video['title'],
            "video_id": matched_video['id'],
            "distance_score": float(distances[0][i])
        })
        
    # 3. Save the starter playlist to MongoDB
    playlist_document = {
        "student_id": student_id,
        "context": f"Auto-generated for course registration: {course_topic}",
        "recommendations": results,
        "trigger_event": "course.registered"
    }
    
    try:
        playlists_collection.insert_one(playlist_document)
        print(f"[BACKGROUND TASK] Successfully saved playlist for {student_id}!")
    except Exception as e:
        print(f"[ERROR] Failed to save course playlist to MongoDB: {e}")

def process_career_path(student_id: str, career_name: str, top_k: int = 3):
    """
    Background task for when a student selects a career path.
    Fetches internship and career preparation videos.
    """
    # Modify the search query to focus on internships and career advice
    search_query = f"{career_name} internship preparation and career guide"
    print(f"[BACKGROUND TASK] Curating career resources for {student_id} on '{career_name}'")
    
    # 1. Fetch videos and build the index
    videos = fetch_youtube_videos(search_query, max_results=5)
    if not videos:
        print(f"[BACKGROUND TASK] No videos fetched for '{career_name}'. Aborting.")
        return
    index = build_faiss_index(videos)
    
    # 2. Simulate an immediate recommendation for career prep
    context_vector = model.encode([f"How to get started and prepare for a {career_name} role"])
    context_vector = np.array(context_vector).astype('float32').reshape(1, -1)
    distances, indices = index.search(context_vector, top_k)
    
    results = []
    for i in range(top_k):
        video_idx = indices[0][i]
        matched_video = videos[video_idx]
        results.append({
            "title": matched_video['title'],
            "video_id": matched_video['id'],
            "distance_score": float(distances[0][i])
        })
        
    # 3. Save the career playlist to MongoDB
    playlist_document = {
        "student_id": student_id,
        "context": f"Auto-generated for career path: {career_name}",
        "recommendations": results,
        "trigger_event": "career_path.selected"
    }
    
    try:
        playlists_collection.insert_one(playlist_document)
        print(f"[BACKGROUND TASK] Successfully saved career playlist for {student_id}!")
    except Exception as e:
        print(f"[ERROR] Failed to save career playlist to MongoDB: {e}")

def process_business_registration(student_id: str, business_type: str, top_k: int = 3):
    """
    Background task for when a student registers a business in the marketplace.
    Fetches videos on how to run that specific type of business as a student.
    """
    # Tailor the search query for a student entrepreneur context
    search_query = f"How to run a {business_type} business as a student entrepreneur"
    print(f"[BACKGROUND TASK] Curating business resources for {student_id} on '{business_type}'")
    
    # 1. Fetch videos and build the index
    videos = fetch_youtube_videos(search_query, max_results=5)
    if not videos:
        print(f"[BACKGROUND TASK] No videos fetched for '{business_type}'. Aborting.")
        return
    index = build_faiss_index(videos)
    
    # 2. Simulate an immediate recommendation for their startup journey
    context_vector = model.encode([f"Tips for starting a {business_type} startup on a university campus"])
    context_vector = np.array(context_vector).astype('float32').reshape(1, -1)
    distances, indices = index.search(context_vector, top_k)
    
    results = []
    for i in range(top_k):
        video_idx = indices[0][i]
        matched_video = videos[video_idx]
        results.append({
            "title": matched_video['title'],
            "video_id": matched_video['id'],
            "distance_score": float(distances[0][i])
        })
        
    # 3. Save the business playlist to MongoDB
    playlist_document = {
        "student_id": student_id,
        "context": f"Auto-generated for marketplace business: {business_type}",
        "recommendations": results,
        "trigger_event": "business.registered"
    }
    
    try:
        playlists_collection.insert_one(playlist_document)
        print(f"[BACKGROUND TASK] Successfully saved business playlist for {student_id}!")
    except Exception as e:
        print(f"[ERROR] Failed to save business playlist to MongoDB: {e}")



#Upgrading My Script to an API

# Initialize the FastAPI application
app = FastAPI(title="AkadVerse YouTube Recommender API")

# Define a Pydantic model for incoming platform events. 
# This tells our API to expect a JSON package containing the type of event, the student's ID, and a payload dictionary containing extra details like the course name.
class PlatformEvent(BaseModel):
    event_type: str
    student_id: str
    payload: dict

# We will store our index and video library in memory for this prototype
app.state.faiss_index = None
app.state.video_library = []

@app.on_event("startup")
def startup_event():
    print("[INFO] AkadVerse YouTube Recommender API started.")
    print("[INFO] The video index is empty. Call POST /index-course to initialize it before using GET /recommend.")

@app.post("/index-course")
def index_course_videos(course_topic: str):
    """
    Endpoint to fetch videos for a course and build the FAISS index.
    Think of this as the setup phase when a new course is added to AkadVerse.
    """
    videos = fetch_youtube_videos(course_topic, max_results=10)
    index = build_faiss_index(videos)
    
    # Save them to the app state so other endpoints can access them
    app.state.faiss_index = index
    app.state.video_library = videos
    
    return {"message": f"Successfully indexed {len(videos)} videos for '{course_topic}'"}

@app.get("/recommend")
def get_recommendations(student_id: str, context: str, top_k: int = 3):
    """
    Endpoint to get video recommendations based on a student's context,
    and save them to MongoDB as a permanent playlist.
    """
    if app.state.faiss_index is None:
        raise HTTPException(status_code=503, detail="Index is empty. Please call POST /index-course first.")
        
    # 1. Embed and Search
    context_vector = model.encode([context])
    context_vector = np.array(context_vector).astype('float32').reshape(1, -1)
    distances, indices = app.state.faiss_index.search(context_vector, top_k)
    
    results = []
    for i in range(top_k):
        video_idx = indices[0][i]
        matched_video = app.state.video_library[video_idx]
        results.append({
            "title": matched_video['title'],
            "video_id": matched_video['id'],
            "distance_score": float(distances[0][i])
        })
        
    # 2. Structure the document for MongoDB
    playlist_document = {
        "student_id": student_id,
        "context": context,
        "recommendations": results
    }
    
    # 3. Save to MongoDB Atlas
    # insert_one() pushes our dictionary into the cloud database
    try:
        insert_result = playlists_collection.insert_one(playlist_document)
        # Convert the unique MongoDB ObjectId to a string so FastAPI can return it as JSON
        playlist_document["_id"] = str(insert_result.inserted_id)
    except Exception as e:
        print(f"[ERROR] Failed to save playlist to MongoDB: {e}")
        raise HTTPException(status_code=500, detail="Failed to save playlist to database.")
        
    return {
        "message": "Playlist successfully generated and saved to MongoDB!", 
        "data": playlist_document
    }

@app.post("/webhook/event")
def handle_platform_event(event: PlatformEvent, background_tasks: BackgroundTasks):
    """
    Simulates the Kafka event bus listener. 
    Receives events and hands the heavy lifting off to background tasks.
    """
    if event.event_type == "course.registered":
        # Extract the course name from the payload
        course_name = event.payload.get("course_name")
        
        if not course_name:
            return {"error": "Missing 'course_name' in payload"}
            
        # Tell FastAPI to run our ML process in the background
        background_tasks.add_task(
            process_course_registration, 
            student_id=event.student_id, 
            course_topic=course_name
        )
        
        # Return a success message immediately, before the AI finishes
        return {
            "status": "success", 
            "message": f"Event '{event.event_type}' received. Building playlist for '{course_name}' in the background."
        }
        
    #More event types can be added here later, such as "lesson.completed" or "struggle_reported"
    elif event.event_type == "career_path.selected":
        # Extract the career name from the payload
        career_name = event.payload.get("career_name")
        
        if not career_name:
            return {"error": "Missing 'career_name' in payload"}
            
        # Route to the specialized career background task
        background_tasks.add_task(
            process_career_path, 
            student_id=event.student_id, 
            career_name=career_name
        )
        
        return {
            "status": "success", 
            "message": f"Event '{event.event_type}' received. Curating career feed for '{career_name}' in the background."
        }
    
    elif event.event_type == "business.registered":
        # Extract the business type from the payload
        business_type = event.payload.get("business_type")
        
        if not business_type:
            return {"error": "Missing 'business_type' in payload"}
            
        # Route to the specialized business background task
        background_tasks.add_task(
            process_business_registration, 
            student_id=event.student_id, 
            business_type=business_type
        )
        
        return {
            "status": "success", 
            "message": f"Event '{event.event_type}' received. Curating startup resources for '{business_type}' in the background."
        }
        
    else:
        return {"error": f"Unknown event type: {event.event_type}"}

if __name__ == "__main__":
    # This tells uvicorn to run our FastAPI 'app' on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
