from youtube_transcript_api import YouTubeTranscriptApi
import requests
from langchain_core.tools import tool

class YouTubeVideoLoader:
    def __init__(self, query: str, max_results: int = 5):
        self.query = query
        self.max_results = max_results
        self.video_ids = self.search_youtube_videos()
        self.transcripts = self.load_transcripts()
        self.verify_transcript_count()


    def search_youtube_videos(self):
        api_key = "AIzaSyDF5t5eIcGe5QEEAflaYZ0wSXXkOAKPLH4"  # Replace with your actual API key
        url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults={self.max_results}&q={self.query}&key={api_key}"
        response = requests.get(url)
        results = response.json()
        return [item['id']['videoId'] for item in results['items'] if item['id']['kind'] == 'youtube#video']

    def load_transcripts(self):
        transcripts = []
        for video_id in self.video_ids:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id,languages=['en'])
                transcripts.append(transcript)
            except Exception as e:
                print(f"Could not fetch transcript for {video_id}: {e}")
        return transcripts

    def verify_transcript_count(self):
        if len(self.transcripts) < self.max_results:
            print(f"Warning: Only {len(self.transcripts)} transcripts retrieved instead of {self.max_results}.")
    
    def load_language(self):
        pass

     
    
