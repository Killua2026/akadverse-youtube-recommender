import os
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build

# Load environment variables (your API key)
load_dotenv()
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# 1. Initialize the Open-Source Embedding Model
# This model converts text into a 384-dimensional vector as specified in the architecture doc.
print("Loading the Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_youtube_videos(query, max_results=5):
    """
    Fetches videos from the YouTube Data API v3 based on a search query.
    """
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    
    print(f"Fetching videos for topic: {query}...")
    request = youtube.search().list(
        part="snippet",
        maxResults=max_results,
        q=query,
        type="video"
    )
    response = request.execute()
    
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
    Converts video text into vectors and loads them into a FAISS index for fast searching.
    """
    # Extract just the text from our video data list
    texts = [video['text'] for video in video_data]
    
    # Generate 384-dimensional embeddings for all videos at once
    print("Generating embeddings for video library...")
    embeddings = model.encode(texts)
    
    # FAISS requires numpy arrays in float32 format
    embeddings = np.array(embeddings).astype('float32')
    
    # Initialize a FAISS index using L2 distance (Euclidean distance)
    dimension = embeddings.shape[1] # This will be 384
    index = faiss.IndexFlatL2(dimension)
    
    # Add our video vectors to the index
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
    context_vector = np.array(context_vector).astype('float32')
    
    # Search the FAISS index for the 'top_k' nearest neighbors
    distances, indices = index.search(context_vector, top_k)
    
    print(f"\n--- Recommendations for context: '{student_context}' ---")
    for i in range(top_k):
        video_idx = indices[0][i]
        match_distance = distances[0][i]
        matched_video = video_data[video_idx]
        print(f"{i+1}. {matched_video['title']} (Distance score: {match_distance:.2f})")

# ==========================================
# Execution Flow (Simulating a Platform Event)
# ==========================================
if __name__ == "__main__":
    # Simulate a 'course.registered' event by pre-fetching some general course videos
    base_course_topic = "Introduction to Data Structures in C++"
    
    # Step A: Fetch videos (acting as our pre-indexed metadata database)
    video_library = fetch_youtube_videos(base_course_topic, max_results=10)
    
    # Step B: Build the vector index
    faiss_index = build_faiss_index(video_library)
    
    # Step C: Simulate an Insight Engine event where a student needs specific help
    # The system encodes the user's current context to match against the videos
    student_action_context = "How to implement a linked list in C++"
    recommend_videos(student_action_context, faiss_index, video_library)