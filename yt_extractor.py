import os
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
# from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api._errors import NoTranscriptFound

from IPython.display import display, Markdown
# Configure logging
logging.basicConfig(level=logging.WARNING)
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
    
    def list_available_transcript_languages(self, video_url_or_id: str) -> List[Dict[str, str]]:
        """List all available transcript languages for a YouTube video.
        
        Args:
            video_url_or_id: YouTube video URL or ID
            
        Returns:
            List of dictionaries containing language code, name, and whether it's auto-generated
        """
        video_id = self.extract_video_id(video_url_or_id) if '/' in video_url_or_id or '.' in video_url_or_id else video_url_or_id
        
        if not video_id:
            logger.error(f"Could not extract video ID from: {video_url_or_id}")
            return []
        
        try:
            # Get video details for title
            video_details = self.get_video_details(video_id)
            video_title = video_details['title'] if video_details else "Unknown Title"
            logger.info(f"Checking available transcript languages for video: {video_title}")
            
            # Get all available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            languages = []
            for transcript in transcript_list:
                lang_code = transcript.language_code
                is_generated = transcript.is_generated
                languages.append({
                    'language_code': lang_code,
                    'language_name': transcript.language,
                    'is_auto_generated': is_generated
                })
            
            # Sort by language name
            languages.sort(key=lambda x: x['language_name'])
            logger.info(f"Found {len(languages)} available transcript languages")
            
            return languages
            
        except Exception as e:
            logger.error(f"Error listing transcripts for video ID {video_id}: {str(e)}")
            return []

    def clean_title(self, title: str) -> str:
        """Clean video title to make it filesystem-friendly.
        
        Args:
            title: Original video title
            
        Returns:
            Cleaned title safe for filenames
        """
        clean = re.sub(r'[<>:"/\\|?*]', '', title)
        clean = clean.replace(' ', '_')
        clean = clean.replace(',', '')
        clean = clean.replace('__', '_')
        return clean
    
    def get_transcript(self, video_url_or_id: str, language: str = 'en') -> Optional[List[Dict[str, Any]]]:
        """Get the raw transcript data for a YouTube video.
        
        Args:
            video_url_or_id: YouTube video URL or ID
            language: Language code for transcript (default: 'en')
            
        Returns:
            List of transcript segments with text and timing information
        """
        video_id = self.extract_video_id(video_url_or_id) if '/' in video_url_or_id or '.' in video_url_or_id else video_url_or_id
        
        if not video_id:
            logger.error(f"Could not extract video ID from: {video_url_or_id}")
            return None
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
            logger.info(f"Successfully retrieved transcript with {len(transcript)} segments")
            return transcript
        except NoTranscriptFound:
            logger.warning(f"No transcript found in language '{language}' for video ID {video_id}")
            raise LanguageUnavailableError(f"No transcript available in '{language}' language")
        except Exception as e:
            logger.error(f"Error getting transcript for video ID {video_id}: {str(e)}")
            return None

    def download_transcript(self, video_url_or_id: str, language: str = 'en', include_metadata: bool = True) -> Optional[Dict[str, Any]]:
        """Download and save transcript for a YouTube video.
        
        Args:
            video_url_or_id: YouTube video URL or ID
            language: Language code for transcript (default: 'en')
            include_metadata: Whether to include video metadata in the output
            
        Returns:
            Dictionary with success status, file path or drive link, and any error messages
        """
        video_id = self.extract_video_id(video_url_or_id) if '/' in video_url_or_id or '.' in video_url_or_id else video_url_or_id
        result = {
            'success': False,
            'video_id': video_id,
            'language': language,
            'filepath': None,
            'drive_link': None,
            'error': None
        }
        
        if not video_id:
            error_msg = f"Could not extract video ID from: {video_url_or_id}"
            logger.error(error_msg)
            result['error'] = error_msg
            return result

        try:
            # Get video details
            video_details = self.get_video_details(video_id)
            if not video_details:
                error_msg = f"Could not fetch video details for {video_url_or_id}"
                logger.error(error_msg)
                result['error'] = error_msg
                return result

            video_title = self.clean_title(video_details['title'])
            
            # Get transcript
            try:
                transcript = self.get_transcript(video_id, language)
                if not transcript:
                    raise Exception("Empty transcript returned")
            except LanguageUnavailableError as e:
                result['error'] = str(e)
                return result
                
            # Format transcript as plain text
            formatted_transcript = "\n".join(segment["text"] for segment in transcript)
            
            # Generate filename
            timestamp = int(time.time())
            filename = f"{timestamp}_youtube_{video_title}.txt"
            
            # Prepare content
            content = ''
            if include_metadata:
                metadata = {
                    'video_id': video_id,
                    'title': video_details['title'],
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'timestamp': timestamp,
                    'published_date': video_details['published_at'],
                    'description': video_details['description'],
                    'default_language': video_details.get('language', 'en'),
                    'transcript_language': language
                }
                
                content = '--- Video Metadata ---\n'
                content += json.dumps(metadata, indent=2)
                content += '\n\n--- Transcript ---\n\n'
            
            content += formatted_transcript

            # Save the transcript
            if self.use_drive and self.drive_service:
                try:
                    drive_link = self.upload_to_drive(content, filename)
                    result['success'] = True
                    result['drive_link'] = drive_link
                    logger.info(f"Successfully uploaded transcript to Google Drive: {drive_link}")
                except Exception as e:
                    error_msg = f"Error uploading to Google Drive: {str(e)}"
                    logger.error(error_msg)
                    result['error'] = error_msg
                    return result
            else:
                try:
                    os.makedirs(self.output_dir, exist_ok=True)
                    filepath = os.path.join(self.output_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    result['success'] = True
                    result['filepath'] = filepath
                    logger.info(f"Successfully saved transcript: {filepath}")
                except Exception as e:
                    error_msg = f"Error saving transcript file: {str(e)}"
                    logger.error(error_msg)
                    result['error'] = error_msg
                    return result

            return result

        except Exception as e:
            error_msg = f"Error processing transcript download: {str(e)}"
            logger.error(error_msg)
            result['error'] = error_msg
            return result
    
    def batch_download_transcripts(self, video_urls_or_ids: List[str], language: str = 'en', 
                                  max_retries: int = 3, retry_delay: int = 5) -> List[Dict[str, Any]]:
        """Download transcripts for multiple YouTube videos with retry logic.
        
        Args:
            video_urls_or_ids: List of YouTube video URLs or IDs
            language: Language code for transcripts (default: 'en')
            max_retries: Maximum number of retry attempts per video
            retry_delay: Seconds to wait between retries
            
        Returns:
            List of dictionaries with results for each video
        """
        results = []
        for url_or_id in video_urls_or_ids:
            logger.info(f"Processing transcript download for: {url_or_id}")
            retries = 0
            success = False
            result = None
            
            while retries < max_retries and not success:
                try:
                    logger.info(f"Attempt {retries + 1} for {url_or_id}")
                    result = self.download_transcript(url_or_id, language)
                    if result and result['success']:
                        logger.info(f"Successfully processed transcript for: {url_or_id}")
                        success = True
                        results.append(result)
                        break
                except LanguageUnavailableError as e:
                    # Don't retry for language unavailability errors
                    logger.warning(f"Language unavailable error - not retrying: {str(e)}")
                    results.append({
                        'success': False,
                        'video_id': self.extract_video_id(url_or_id),
                        'language': language,
                        'error': str(e),
                        'reason': 'language_unavailable'
                    })
                    break
                except Exception as e:
                    logger.error(f"Attempt {retries + 1} failed for {url_or_id}: {str(e)}")
                    retries += 1
                    if retries < max_retries:
                        logger.info(f"Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
            
            if not success and not any(r.get('video_id') == self.extract_video_id(url_or_id) for r in results):
                results.append({
                    'success': False,
                    'video_id': self.extract_video_id(url_or_id),
                    'language': language,
                    'error': result['error'] if result and 'error' in result else "Maximum retries exceeded",
                    'reason': 'other_error'
                })
        
        return results


# Example usage
if __name__ == "__main__":
    # Replace with your actual API key
    API_KEY = os.environ.get("YOUTUBE_DATA_API_KEY", None)
    
    # Create an instance of the YouTube data fetcher
    youtube_fetcher = YouTubeDataFetcher(API_KEY)
    
    # Example YouTube URLs (using different formats)
    video_url = "https://www.youtube.com/watch?v=9zXCGnvCKWM"
    
    # Use the first URL for this example
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
    
    # Get channel details if we have video details
    if video_details and video_details.get('channel_id'):
        channel_details = youtube_fetcher.get_channel_details(video_details.get('channel_id'))
        if channel_details:
            print("Channel Details:")
            print(f"Name: {channel_details.get('title')}")
            print(f"Subscribers: {channel_details.get('subscriber_count')}")
            print(f"Total Views: {channel_details.get('view_count')}")
            print(f"Total Videos: {channel_details.get('video_count')}")

    print("\n" + "-"*80 + "\n")
    print("Getting transcript...")

    print("\n\n=== Available Transcript Languages ===")
    languages = youtube_fetcher.list_available_transcript_languages(video_url)
    if languages:
        for lang in languages:
            auto_gen = "(auto-generated)" if lang['is_auto_generated'] else ""
            print(f"{lang['language_code']}: {lang['language_name']} {auto_gen}")

    # 3. Get raw transcript data
    print("\n\n=== Raw Transcript Data (first 3 segments) ===")
    try:
        transcript = youtube_fetcher.get_transcript(video_url, language='en')
    except Exception as e:
        print(f"Error getting transcript: {e}")

    # 4. Get raw transcript data
    full_transcript_text = ""
    if transcript:
        # Combine all segments into a single text variable
        full_transcript_text = " ".join([segment['text'] for segment in transcript])
        print(f"\nFull transcript length: {len(full_transcript_text)} characters")
        print(full_transcript_text)
