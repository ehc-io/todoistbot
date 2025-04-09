import os
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeDataFetcher:
    """A class to interact with YouTube API and fetch various video data."""
    
    def __init__(self, api_key: str):
        """Initialize with YouTube API key.
        
        Args:
            api_key: YouTube Data API v3 key
        """
        self.api_key = api_key
        self.youtube = None
        self.api_key_valid = False
        self.initialize_youtube_client()
        
    @staticmethod
    def extract_video_id(youtube_url: str) -> Optional[str]:
        """Extract the video ID from a YouTube URL.
        
        Supports various YouTube URL formats:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://youtube.com/shorts/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        
        Args:
            youtube_url: Full YouTube URL
            
        Returns:
            Video ID or None if extraction fails
        """
        if not youtube_url:
            return None
            
        # Handle youtu.be short URLs
        if 'youtu.be' in youtube_url:
            parsed_url = urlparse(youtube_url)
            return parsed_url.path.strip('/')
            
        # Handle youtube.com/shorts/ URLs
        if '/shorts/' in youtube_url:
            pattern = r'(?:\/shorts\/)([a-zA-Z0-9_-]+)'
            match = re.search(pattern, youtube_url)
            return match.group(1) if match else None
            
        # Handle standard youtube.com URLs
        parsed_url = urlparse(youtube_url)
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query).get('v', [None])[0]
            elif '/embed/' in parsed_url.path:
                return parsed_url.path.split('/embed/')[1]
                
        return None
    
    def initialize_youtube_client(self) -> None:
        """Initialize the YouTube API client."""
        try:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            # Test with a simple request to verify API key
            self.youtube.videos().list(part="snippet", id="dQw4w9WgXcQ").execute()
            self.api_key_valid = True
            logger.info("YouTube API client initialized successfully")
        except HttpError as e:
            logger.error(f"Failed to initialize YouTube API client: {str(e)}")
            self.api_key_valid = False
            self.youtube = None
    
    def get_video_details(self, video_url_or_id: str) -> Optional[Dict[str, Any]]:
        """Fetch video details using YouTube API.
        
        Args:
            video_url_or_id: YouTube video URL or ID
            
        Returns:
            Dictionary containing video details or None if error occurs
        """
        video_id = self.extract_video_id(video_url_or_id) if '/' in video_url_or_id or '.' in video_url_or_id else video_url_or_id
        
        if not video_id:
            logger.error(f"Could not extract video ID from: {video_url_or_id}")
            return None
            
        if not self.api_key_valid or not self.youtube:
            logger.error("YouTube API key is invalid or client not initialized.")
            return None
        try:
            request = self.youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_id
            )
            response = request.execute()

            if not response.get('items'):
                logger.warning(f"No video details found for YouTube video ID: {video_id}")
                return None

            video = response['items'][0]
            snippet = video.get('snippet', {})
            statistics = video.get('statistics', {})
            content_details = video.get('contentDetails', {})

            return {
                'title': snippet.get('title', 'N/A'),
                'description': snippet.get('description', ''),
                'published_at': snippet.get('publishedAt', ''),
                'channel_id': snippet.get('channelId', ''),
                'channel_title': snippet.get('channelTitle', 'N/A'),
                'tags': snippet.get('tags', []),
                'category_id': snippet.get('categoryId', ''),
                'thumbnails': snippet.get('thumbnails', {}),
                'view_count': statistics.get('viewCount', 'N/A'),
                'like_count': statistics.get('likeCount', 'N/A'),
                'comment_count': statistics.get('commentCount', 'N/A'),
                'duration': content_details.get('duration', 'N/A'),
                'definition': content_details.get('definition', 'N/A'),  # hd or sd
                'caption': content_details.get('caption', 'false') == 'true'
            }

        except Exception as e:
            logger.error(f"Error fetching YouTube video details for ID {video_id}: {str(e)}")
            return None
    
    def get_video_comments(self, video_url_or_id: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Fetch comments for a YouTube video.
        
        Args:
            video_url_or_id: YouTube video URL or ID
            max_results: Maximum number of comments to fetch (default: 20)
            
        Returns:
            List of comments with author and text information
        """
        video_id = self.extract_video_id(video_url_or_id) if '/' in video_url_or_id or '.' in video_url_or_id else video_url_or_id
        
        if not video_id:
            logger.error(f"Could not extract video ID from: {video_url_or_id}")
            return []
            
        if not self.api_key_valid or not self.youtube:
            logger.error("YouTube API key is invalid or client not initialized.")
            return []
        
        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_results,
                textFormat="plainText"
            )
            response = request.execute()
            
            comments = []
            for item in response.get('items', []):
                comment = item.get('snippet', {}).get('topLevelComment', {}).get('snippet', {})
                comments.append({
                    'author': comment.get('authorDisplayName', 'Anonymous'),
                    'text': comment.get('textDisplay', ''),
                    'published_at': comment.get('publishedAt', ''),
                    'like_count': comment.get('likeCount', 0),
                    'author_channel_id': comment.get('authorChannelId', {}).get('value', '')
                })
            
            return comments
            
        except Exception as e:
            logger.error(f"Error fetching comments for YouTube video ID {video_id}: {str(e)}")
            return []
    
    def get_related_videos(self, video_url_or_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Fetch related videos for a YouTube video.
        
        Args:
            video_url_or_id: YouTube video URL or ID
            max_results: Maximum number of related videos to fetch (default: 10)
            
        Returns:
            List of related videos with basic details
        """
        video_id = self.extract_video_id(video_url_or_id) if '/' in video_url_or_id or '.' in video_url_or_id else video_url_or_id
        
        if not video_id:
            logger.error(f"Could not extract video ID from: {video_url_or_id}")
            return []
            
        if not self.api_key_valid or not self.youtube:
            logger.error("YouTube API key is invalid or client not initialized.")
            return []
        
        try:
            request = self.youtube.search().list(
                part="snippet",
                relatedToVideoId=video_id,
                type="video",
                maxResults=max_results
            )
            response = request.execute()
            
            related_videos = []
            for item in response.get('items', []):
                related_videos.append({
                    'video_id': item.get('id', {}).get('videoId', ''),
                    'title': item.get('snippet', {}).get('title', 'N/A'),
                    'channel_title': item.get('snippet', {}).get('channelTitle', 'N/A'),
                    'published_at': item.get('snippet', {}).get('publishedAt', ''),
                    'thumbnail': item.get('snippet', {}).get('thumbnails', {}).get('medium', {}).get('url', '')
                })
            
            return related_videos
            
        except Exception as e:
            logger.error(f"Error fetching related videos for YouTube video ID {video_id}: {str(e)}")
            return []
    
    def get_channel_details(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Fetch channel details using YouTube API.
        
        Args:
            channel_id: YouTube channel ID
            
        Returns:
            Dictionary containing channel details or None if error occurs
        """
        if not self.api_key_valid or not self.youtube:
            logger.error("YouTube API key is invalid or client not initialized.")
            return None
        
        try:
            request = self.youtube.channels().list(
                part="snippet,statistics,brandingSettings",
                id=channel_id
            )
            response = request.execute()
            
            if not response.get('items'):
                logger.warning(f"No channel details found for YouTube channel ID: {channel_id}")
                return None
                
            channel = response['items'][0]
            snippet = channel.get('snippet', {})
            statistics = channel.get('statistics', {})
            branding = channel.get('brandingSettings', {}).get('channel', {})
            
            return {
                'title': snippet.get('title', 'N/A'),
                'description': snippet.get('description', ''),
                'custom_url': snippet.get('customUrl', ''),
                'published_at': snippet.get('publishedAt', ''),
                'thumbnails': snippet.get('thumbnails', {}),
                'country': snippet.get('country', ''),
                'view_count': statistics.get('viewCount', 'N/A'),
                'subscriber_count': statistics.get('subscriberCount', 'N/A'),
                'video_count': statistics.get('videoCount', 'N/A'),
                'banner_image': branding.get('bannerImageUrl', ''),
                'keywords': branding.get('keywords', '')
            }
            
        except Exception as e:
            logger.error(f"Error fetching YouTube channel details for ID {channel_id}: {str(e)}")
            return None
    
    def search_videos(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for YouTube videos based on a query.
        
        Args:
            query: Search term
            max_results: Maximum number of videos to fetch (default: 10)
            
        Returns:
            List of videos matching the search query
        """
        if not self.api_key_valid or not self.youtube:
            logger.error("YouTube API key is invalid or client not initialized.")
            return []
        
        try:
            request = self.youtube.search().list(
                q=query,
                part="snippet",
                type="video",
                maxResults=max_results
            )
            response = request.execute()
            
            search_results = []
            for item in response.get('items', []):
                search_results.append({
                    'video_id': item.get('id', {}).get('videoId', ''),
                    'title': item.get('snippet', {}).get('title', 'N/A'),
                    'description': item.get('snippet', {}).get('description', ''),
                    'channel_title': item.get('snippet', {}).get('channelTitle', 'N/A'),
                    'published_at': item.get('snippet', {}).get('publishedAt', ''),
                    'thumbnail': item.get('snippet', {}).get('thumbnails', {}).get('high', {}).get('url', '')
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching YouTube videos with query '{query}': {str(e)}")
            return []


# Example usage
if __name__ == "__main__":
    # Replace with your actual API key
    API_KEY = os.environ.get("YOUTUBE_API_KEY", "your_api_key_here")
    
    # Create an instance of the YouTube data fetcher
    youtube_fetcher = YouTubeDataFetcher(API_KEY)
    
    # Example YouTube URLs (using different formats)
    video_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Standard format
        "https://youtu.be/dQw4w9WgXcQ",                 # Short URL format
        "dQw4w9WgXcQ"                                   # Just the ID
    ]
    
    # Use the first URL for this example
    video_url = video_urls[0]
    print(f"Using YouTube URL: {video_url}")
    
    # Extract and show the video ID (for demonstration)
    video_id = youtube_fetcher.extract_video_id(video_url)
    print(f"Extracted video ID: {video_id}")
    print("\n" + "-"*80 + "\n")
    
    # Get detailed information about the video
    video_details = youtube_fetcher.get_video_details(video_url)
    if video_details:
        print("Video Details:")
        print(f"Title: {video_details.get('title')}")
        print(f"Channel: {video_details.get('channel_title')}")
        print(f"Views: {video_details.get('view_count')}")
        print(f"Likes: {video_details.get('like_count')}")
        print(f"Duration: {video_details.get('duration')}")
        print(f"Has captions: {video_details.get('caption')}")
        print("\n" + "-"*80 + "\n")
    
    # Get comments for the video
    comments = youtube_fetcher.get_video_comments(video_url, max_results=5)
    if comments:
        print("Top 5 Comments:")
        for i, comment in enumerate(comments, 1):
            print(f"{i}. {comment.get('author')}: {comment.get('text')[:100]}...")
        print("\n" + "-"*80 + "\n")
    
    # Get related videos
    related_videos = youtube_fetcher.get_related_videos(video_url, max_results=3)
    if related_videos:
        print("Related Videos:")
        for i, video in enumerate(related_videos, 1):
            print(f"{i}. {video.get('title')} (ID: {video.get('video_id')})")
        print("\n" + "-"*80 + "\n")
    
    # Get channel details if we have video details
    if video_details and video_details.get('channel_id'):
        channel_details = youtube_fetcher.get_channel_details(video_details.get('channel_id'))
        if channel_details:
            print("Channel Details:")
            print(f"Name: {channel_details.get('title')}")
            print(f"Subscribers: {channel_details.get('subscriber_count')}")
            print(f"Total Views: {channel_details.get('view_count')}")
            print(f"Total Videos: {channel_details.get('video_count')}")
