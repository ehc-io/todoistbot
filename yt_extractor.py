import os
import re
import time
import sys
import logging
import argparse
import json
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Import boto3 for S3 uploads
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("Warning: boto3 is not installed. S3 upload functionality will be disabled.")
    print("Install with: pip install boto3")

# Import pytubefix for video downloads
try:
    from pytubefix import YouTube, Playlist, Channel
    from pytubefix.cli import on_progress
    PYTUBEFIX_AVAILABLE = True
except ImportError:
    PYTUBEFIX_AVAILABLE = False
    print("Warning: pytubefix is not installed. Video download functionality will be disabled.")
    print("Install with: pip install pytubefix")

from common import call_llm_completion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LanguageUnavailableError(Exception):
    """Exception raised when a requested transcript language is not available."""
    def __init__(self, message, available_languages=None):
        self.message = message
        self.available_languages = available_languages
        super().__init__(self.message)
        
    def __str__(self):
        if self.available_languages:
            lang_list = "\n".join([f"  {lang['language_code']}: {lang['language_name']}" 
                                  for lang in self.available_languages])
            return f"{self.message}\nAvailable languages:\n{lang_list}"
        return self.message

class YouTubeDataFetcher:
    """A class to interact with YouTube API and fetch various video data."""
    
    def __init__(self, api_key: str, output_dir: str = "./outputs"):
        """Initialize with YouTube API key.
        
        Args:
            api_key: YouTube Data API v3 key
            output_dir: Directory to save downloaded files
        """
        self.api_key = api_key
        self.youtube = None
        self.api_key_valid = False
        self.output_dir = output_dir
        self.use_drive = False
        self.drive_service = None
        
        # S3 configuration
        self.use_s3 = False
        self.s3_bucket = None
        self.s3_client = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize YouTube API client if API key is provided
        if api_key:
            self.initialize_youtube_client()
        
    def configure_s3(self, bucket_name: str, region_name: str = None) -> bool:
        """Configure S3 upload functionality.
        
        Args:
            bucket_name: S3 bucket name for uploads
            region_name: AWS region name (default: None, will use environment variables)
            
        Returns:
            True if configuration successful, False otherwise
        """
        if not S3_AVAILABLE:
            logger.error("boto3 is not installed. Cannot configure S3 uploads.")
            return False
            
        try:
            # Use provided region or fall back to environment variables
            self.s3_client = boto3.client('s3', region_name=region_name)
            self.s3_bucket = bucket_name
            self.use_s3 = True
            
            # Test bucket access
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Successfully configured S3 uploads to bucket: {bucket_name} in region: {region_name or 'default'}")
            return True
        except NoCredentialsError:
            logger.error("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
            self.use_s3 = False
            return False
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404':
                logger.error(f"S3 bucket {bucket_name} does not exist")
            elif error_code == '403':
                logger.error(f"Access denied to S3 bucket {bucket_name}")
            else:
                logger.error(f"S3 bucket access error: {str(e)}")
            self.use_s3 = False
            return False
        except Exception as e:
            logger.error(f"Error configuring S3: {str(e)}")
            self.use_s3 = False
            return False
    
    def upload_to_s3(self, local_file_path: str, object_name: Optional[str] = None) -> Optional[str]:
        """Upload a file to S3 bucket.
        
        Args:
            local_file_path: Path to local file to upload
            object_name: S3 object name. If not specified, file name is used
            
        Returns:
            S3 object URL if successful, None otherwise
        """
        if not self.use_s3 or not self.s3_client:
            logger.error("S3 is not configured for uploads")
            return None
            
        # If object_name not specified, use file name from local_file_path
        if object_name is None:
            object_name = os.path.basename(local_file_path)
            
        # Use week number of the year (ISO week date format: week starts on Monday, ends on Sunday)
        from datetime import datetime
        current_date = datetime.now()
        # Get ISO week number (1-53)
        week_of_year = current_date.isocalendar()[1]
        
        # Create week-based folder structure: 'week_XX' where XX is the week number
        folder_name = f"week_{week_of_year:02d}"
        
        # Final object path: week_XX/filename
        object_name = f"{folder_name}/{object_name}"
            
        try:
            logger.info(f"Uploading {local_file_path} to S3 bucket {self.s3_bucket}, folder: {folder_name}")
            self.s3_client.upload_file(local_file_path, self.s3_bucket, object_name)
            s3_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{object_name}"
            logger.info(f"S3 upload successful: {s3_url}")
            return s3_url
        except NoCredentialsError:
            logger.error("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
            return None
        except ClientError as e:
            logger.error(f"S3 upload error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error during S3 upload: {str(e)}")
            return None
    
    def direct_upload_to_s3(self, video_id: str, video_title: str) -> Optional[str]:
        """Create a reference file in S3 for a YouTube video without downloading it first.
        
        Args:
            video_id: YouTube video ID
            video_title: The title of the video for naming
            
        Returns:
            S3 object URL if successful, None otherwise
        """
        if not self.use_s3 or not self.s3_client:
            logger.error("S3 is not configured for uploads")
            return None
        
        try:
            # Use week number of the year (ISO week date format: week starts on Monday, ends on Sunday)
            from datetime import datetime
            import json
            
            current_date = datetime.now()
            timestamp = int(time.time())
            # Get ISO week number (1-53)
            week_of_year = current_date.isocalendar()[1]
            
            # Create week-based folder structure: 'week_XX' where XX is the week number
            folder_name = f"week_{week_of_year:02d}"
            
            # Clean the title for a safe filename
            clean_title = self.clean_title(video_title)
            
            # Create the reference file name
            filename = f"{timestamp}_{clean_title}_reference.json"
            
            # Create a reference JSON file with video details
            reference_data = {
                "youtube_video_id": video_id,
                "title": video_title,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "processed_at": timestamp,
                "reference_type": "youtube_video",
                "direct_embed_url": f"https://www.youtube.com/embed/{video_id}"
            }
            
            # Create temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(reference_data, temp_file, indent=4)
                temp_file_path = temp_file.name
            
            # Final object path: week_XX/filename
            object_name = f"{folder_name}/{filename}"
            
            try:
                logger.info(f"Creating S3 reference for YouTube video {video_id} in bucket {self.s3_bucket}, folder: {folder_name}")
                self.s3_client.upload_file(temp_file_path, self.s3_bucket, object_name)
                s3_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{object_name}"
                logger.info(f"S3 reference creation successful: {s3_url}")
                
                # Remove the temporary file
                os.unlink(temp_file_path)
                
                return s3_url
            except NoCredentialsError:
                logger.error("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
                # Make sure to clean up the temp file
                os.unlink(temp_file_path)
                return None
            except ClientError as e:
                logger.error(f"S3 upload error: {str(e)}")
                # Make sure to clean up the temp file
                os.unlink(temp_file_path)
                return None
            except Exception as e:
                logger.error(f"Error during S3 reference creation: {str(e)}")
                # Make sure to clean up the temp file
                os.unlink(temp_file_path)
                return None
            
        except Exception as e:
            logger.error(f"Error preparing S3 reference file: {str(e)}")
            return None
    
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
            
        Raises:
            LanguageUnavailableError: If the requested language is not available
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
            # Get available languages to provide better error information
            try:
                available_languages = self.list_available_transcript_languages(video_id)
                if available_languages:
                    raise LanguageUnavailableError(
                        f"No transcript available in '{language}' language", 
                        available_languages
                    )
                else:
                    raise LanguageUnavailableError(f"No transcripts available for this video.")
            except Exception:
                # If we can't get the language list, fall back to simple error
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
            
        The function handles language availability errors and provides helpful feedback
        on which languages are available.
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
                # Add available languages to the result if we have them
                if hasattr(e, 'available_languages') and e.available_languages:
                    result['available_languages'] = e.available_languages
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
    
    # ----- Video Download Functionality -----
    def progress_callback(self, stream, chunk, bytes_remaining):
        """Display download progress."""
        if not hasattr(self, 'download_progress_shown'):
            self.download_progress_shown = {}
        
        total_size = stream.filesize
        bytes_downloaded = total_size - bytes_remaining
        percentage = (bytes_downloaded / total_size) * 100
        
        # Only update progress every 5%
        current_progress = int(percentage / 5) * 5
        
        # Instead of using video_id which doesn't exist on stream objects,
        # use a simple string key or the stream's itag as an identifier
        stream_key = str(stream.itag)
        
        if stream_key not in self.download_progress_shown or current_progress > self.download_progress_shown[stream_key]:
            self.download_progress_shown[stream_key] = current_progress
            sys.stdout.write(f"\rProgress: {percentage:.2f}%")
            sys.stdout.flush()
    
    def download_video(self, url_or_id: str) -> Dict[str, Any]:
        """Download a single video from URL to the specified path.
        
        Args:
            url_or_id: YouTube video URL or video ID
            
        Returns:
            Dictionary with download results
        """
        if not PYTUBEFIX_AVAILABLE:
            return {
                'success': False,
                'error': "pytubefix not installed. Install with: pip install pytubefix"
            }
        
        result = {
            'success': False,
            'video_id': None,
            'title': None,
            'filepath': None,
            's3_url': None,
            'error': None
        }
        
        try:
            # Ensure we have a proper URL
            if '/' not in url_or_id and '.' not in url_or_id:
                # This is just a video ID, convert to URL
                url = f"https://www.youtube.com/watch?v={url_or_id}"
            else:
                url = url_or_id
            
            # Initialize YouTube object
            yt = YouTube(url, on_progress_callback=self.progress_callback)
            
            # Set video details in result
            result['video_id'] = yt.video_id
            result['title'] = yt.title
            
            # Get video details
            video_title = self.clean_title(yt.title)
            timestamp = int(time.time())
            
            # Format the filename
            filename = f"{timestamp}_{video_title}.mp4"
            full_path = os.path.join(self.output_dir, filename)
            
            logger.info(f"Downloading: {yt.title}")
            
            # Get highest resolution stream with both video and audio
            stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by('resolution').desc().first()
            
            if not stream:
                logger.warning(f"No suitable stream found for {yt.title}. Trying fallback method...")
                # Fallback to highest resolution mp4
                stream = yt.streams.filter(file_extension="mp4").order_by('resolution').desc().first()
            
            if not stream:
                error_msg = f"Error: No downloadable stream found for {yt.title}"
                logger.error(error_msg)
                result['error'] = error_msg
                return result
                
            # Download the video with error handling for progress callback issues
            try:
                stream.download(output_path=self.output_dir, filename=filename)
                logger.info(f"Successfully downloaded: {full_path}")
                
                result['success'] = True
                result['filepath'] = full_path
                
                # Upload to S3 if configured
                if self.use_s3 and os.path.exists(full_path):
                    s3_url = self.upload_to_s3(full_path)
                    if s3_url:
                        result['s3_url'] = s3_url
                        logger.info(f"Successfully uploaded video to S3: {s3_url}")
                    else:
                        logger.warning("Failed to upload video to S3")
            except Exception as download_error:
                error_msg = f"Error during download: {str(download_error)}"
                logger.error(error_msg)
                result['error'] = error_msg
                return result
                    
            return result
            
        except Exception as e:
            error_msg = f"Error downloading video {url_or_id}: {str(e)}"
            logger.error(error_msg)
            result['error'] = error_msg
            return result
    
    def get_video_urls(self, url: str) -> List[str]:
        """Get all video URLs from a playlist, channel, or single video.
        
        Args:
            url: YouTube URL (playlist, channel, or video)
            
        Returns:
            List of video URLs
        """
        if not PYTUBEFIX_AVAILABLE:
            logger.error("pytubefix not installed. Install with: pip install pytubefix")
            return []
            
        # If 'list=' is in the URL, assume it's a playlist
        if 'list=' in url:
            try:
                playlist = Playlist(url)
                logger.info(f"Playlist: {playlist.title}, {len(playlist.video_urls)} videos found")
                return playlist.video_urls
            except Exception as e:
                logger.error(f"Error retrieving playlist videos: {e}")
                return []
        # If '/channel/' is in the URL or '/user/' or '/@' (for new channel URLs), assume it's a channel
        elif '/channel/' in url or '/user/' in url or '/@' in url:
            try:
                channel = Channel(url)
                logger.info(f"Channel: {channel.channel_name}, {len(channel.video_urls)} videos found")
                return channel.video_urls
            except Exception as e:
                logger.error(f"Error retrieving channel videos: {e}")
                return []
        # Otherwise, treat it as a single video link
        else:
            try:
                # This call checks if the link is a valid single video
                yt = YouTube(url)
                logger.info(f"Single video: {yt.title}")
                return [url]
            except Exception as e:
                logger.error(f"Error retrieving single video: {e}")
                return []
    
    def batch_download_videos(self, url: str, max_workers: int = 2) -> Dict[str, Any]:
        """Download videos from URL (supports playlists, channels, and single videos).
        
        Args:
            url: YouTube URL (playlist, channel, or video)
            max_workers: Number of concurrent downloads
            
        Returns:
            Dictionary with download results summary
        """
        if not PYTUBEFIX_AVAILABLE:
            return {
                'success': False,
                'error': "pytubefix not installed. Install with: pip install pytubefix"
            }
            
        # Get all video URLs
        video_urls = self.get_video_urls(url)
        
        if not video_urls:
            logger.error("No videos found to download.")
            return {
                'success': False,
                'error': "No videos found to download."
            }
        
        logger.info(f"Starting download of {len(video_urls)} videos to {self.output_dir}")
        
        # Download videos
        successful = 0
        failed = 0
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            download_results = list(executor.map(lambda url: self.download_video(url), video_urls))
            
        for result in download_results:
            if result['success']:
                successful += 1
            else:
                failed += 1
            results.append(result)
        
        logger.info(f"Download complete. {successful} videos downloaded successfully, {failed} failed.")
        
        return {
            'success': True,
            'total': len(video_urls),
            'successful': successful,
            'failed': failed,
            'results': results
        }

def get_transcript_summary(transcript_text: str, text_model: str) -> str:
    """
    Get summary of YouTube transcript using LLM.
    
    Args:
        transcript_text: The full transcript text
        text_model: The LLM model to use for summarization
        
    Returns:
        A summary of the transcript (around 500 words)
    """
    if not transcript_text: 
        return "No transcript content to summarize."
    
    # Truncate text if needed (similar to PDF summarization)
    max_chars = 25000  # Same limit as webpage/PDF
    if len(transcript_text) > max_chars:
        transcript_text = transcript_text[:max_chars] + "..."
        logger.debug(f"Truncated transcript text to {max_chars} chars for summarization.")
    
    prompt = open("prompts/yt_video").read().strip() + "\nTRANSCRIPTION CONTENT: \n" + transcript_text
    return call_llm_completion(prompt, text_model, max_tokens=650)  # Increased token limit for longer summary

def enhance_youtube_processing(
    url: str, 
    text_model: str,
    download_youtube_video: bool = False,
    youtube_output_dir: str = "./downloads",
    s3_upload: bool = False,
    s3_bucket: str = None,
    s3_region: str = None
) -> dict:
    """
    Enhanced YouTube processing using YouTubeDataFetcher from yt_extractor.py
    
    Args:
        url: YouTube URL
        text_model: Text model for summarization
        download_youtube_video: Whether to download the video
        youtube_output_dir: Output directory for downloaded videos
        s3_upload: Whether to upload videos to S3
        s3_bucket: S3 bucket name for uploads
        s3_region: AWS region for S3 uploads (defaults to AWS_REGION environment variable)
        
    Returns:
        Dictionary with processed YouTube data
    """
    youtube_api_key = os.getenv('YOUTUBE_DATA_API_KEY')
    result = {
        'url': url,
        'type': 'youtube',
        'content': None,
        'error': None,
        'extraction_method': 'youtube_extractor'
    }
    
    if not youtube_api_key:
        logger.error("YOUTUBE_DATA_API_KEY not set. Cannot process YouTube URL.")
        result['error'] = 'YouTube API key missing'
        result['content'] = "[Error: YouTube API key not configured]"
        return result
    
    try:
        # Initialize YouTubeDataFetcher
        fetcher = YouTubeDataFetcher(youtube_api_key, output_dir=youtube_output_dir)
        
        # Configure S3 if requested
        if s3_upload and s3_bucket:
            if not S3_AVAILABLE:
                logger.warning("S3 upload requested but boto3 is not installed. Install with: pip install boto3")
                result['error'] = 'S3 upload requested but boto3 is not installed'
            else:
                logger.info(f"Configuring S3 upload to bucket: {s3_bucket} in region: {s3_region or 'default'}")
                s3_configured = fetcher.configure_s3(s3_bucket, region_name=s3_region)
                if not s3_configured:
                    logger.warning("Failed to configure S3 upload")
                    # Continue processing but note the error
                    result['error'] = 'Failed to configure S3 upload'
        
        # Extract video ID
        video_id = fetcher.extract_video_id(url)
        if not video_id:
            logger.error(f"Could not extract video ID from URL: {url}")
            result['error'] = 'Invalid YouTube URL format'
            result['content'] = "[Error: Could not extract video ID]"
            return result
        
        # Get video details
        video_details = fetcher.get_video_details(video_id)
        if not video_details:
            logger.error(f"Failed to fetch video details for {url}")
            result['error'] = 'Failed to fetch video details'
            result['content'] = "[Error: Could not retrieve video information]"
            return result
        
        # Get comments
        comments = fetcher.get_video_comments(video_id, max_results=5)
        
        # Get channel details
        channel_details = None
        if video_details.get('channel_id'):
            channel_details = fetcher.get_channel_details(video_details.get('channel_id'))
        
        # Get transcript
        transcript_text = ""
        transcript_summary = ""
        try:
            transcript = fetcher.get_transcript(video_id)
            if transcript:
                # Convert transcript segments to full text
                transcript_text = "\n".join(segment["text"] for segment in transcript)
                # Get transcript summary
                transcript_summary = get_transcript_summary(transcript_text, text_model)
                logger.info(f"Successfully obtained and summarized transcript for video ID: {video_id}")
            else:
                logger.warning(f"No transcript available for video ID: {video_id}")
        except Exception as e:
            logger.warning(f"Failed to get transcript for video ID {video_id}: {e}")
            # Continue processing even if transcript fails
        
        # Handle video download and/or S3 upload based on flags
        downloaded_video_path = None
        s3_url = None
        
        # Case 1: Download video locally and possibly upload to S3
        if download_youtube_video:
            try:
                logger.info(f"Downloading YouTube video: {url}")
                download_result = fetcher.download_video(url)
                if download_result and download_result.get('success'):
                    downloaded_video_path = download_result.get('filepath')
                    result['downloaded_video_path'] = downloaded_video_path
                    logger.info(f"Successfully downloaded video to: {downloaded_video_path}")
                    
                    # Upload to S3 if requested and not already uploaded during download
                    if s3_upload and not download_result.get('s3_url') and downloaded_video_path:
                        try:
                            s3_url = fetcher.upload_to_s3(downloaded_video_path)
                            if s3_url:
                                result['s3_url'] = s3_url
                                logger.info(f"Successfully uploaded downloaded video to S3: {s3_url}")
                            else:
                                logger.warning("Failed to upload video to S3 after download")
                        except Exception as s3_e:
                            logger.error(f"Error uploading downloaded video to S3: {s3_e}")
                    # If S3 URL was already set during download
                    elif s3_upload and download_result.get('s3_url'):
                        result['s3_url'] = download_result.get('s3_url')
                else:
                    error_msg = download_result.get('error') if download_result else "Unknown download error"
                    logger.warning(f"Failed to download video: {error_msg}")
            except Exception as e:
                logger.error(f"Error during video download: {e}")
        
        # Case 2: S3 upload requested without local download (create reference only)
        elif s3_upload:
            try:
                logger.info(f"Creating S3 reference for YouTube video without local download: {url}")
                s3_url = fetcher.direct_upload_to_s3(video_id, video_details.get('title', 'Unknown'))
                if s3_url:
                    result['s3_url'] = s3_url
                    logger.info(f"Successfully created S3 reference for video: {s3_url}")
                else:
                    logger.warning(f"Failed to create S3 reference for video: {url}")
            except Exception as e:
                logger.error(f"Error creating S3 reference: {e}")
        
        # Format content
        content_parts = [
            f"Title: {video_details.get('title', 'N/A')}  ",
            "---",
            f"Views: {video_details.get('view_count', 'N/A')}  ",
            f"Likes: {video_details.get('like_count', 'N/A')}  ",
            f"Duration: {video_details.get('duration', 'N/A')}  ",
            f"Has captions: {video_details.get('caption', False)}  ",
            "   ",
            "---",
        ]
        if channel_details:
            content_parts.append("Channel Details:  ")
            content_parts.append(f"Name: {channel_details.get('title', 'N/A')}  ")
            content_parts.append(f"Subscribers: {channel_details.get('subscriber_count', 'N/A')}  ")
            content_parts.append(f"Total Views: {channel_details.get('view_count', 'N/A')}  ")
            content_parts.append(f"Total Videos: {channel_details.get('video_count', 'N/A')}  ")
            content_parts.append("  ")
            content_parts.append("---")

        # Transcript section
        if transcript_text:
            content_parts.append(f"Full transcript length: {len(transcript_text)} characters  ")
            content_parts.append("  ")
            if transcript_summary:
                content_parts.append(f"Transcript Summary: {transcript_summary}  ")
                content_parts.append("  ")
        else:
            content_parts.append("Transcript: Not available")
        
        # Comments section
        if comments:
            content_parts.append("Top 5 Comments:")
            for i, comment in enumerate(comments, 1):
                comment_text = comment.get('text', '')
                # Truncate long comments
                if len(comment_text) > 100:
                    comment_text = comment_text[:97] + "..."
                content_parts.append(f"{i}. {comment.get('author', 'Anonymous')}: {comment_text}")
            content_parts.append("--")
        
        # Combine all parts
        result['content'] = "\n".join(content_parts)
        return result
        
    except Exception as e:
        logger.error(f"Error processing YouTube URL {url}: {e}", exc_info=True)
        result['error'] = f"Processing error: {str(e)}"
        result['content'] = f"[Error: Failed to process YouTube content: {str(e)}]"
        return result

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YouTube Data and Content Extractor')
    
    # Main options
    parser.add_argument('url', help='YouTube URL (video, playlist, or channel)')
    parser.add_argument('-o', '--output', default='./downloads',
                        help='Output directory for downloads (default: ./downloads)')
    parser.add_argument('-k', '--api-key', 
                        help='YouTube Data API key (required for metadata and transcript extraction)')
    
    # Extraction options
    extraction_group = parser.add_argument_group('Extraction Options')
    extraction_group.add_argument('-i', '--info', action='store_true',
                                help='Get basic video information')
    extraction_group.add_argument('-t', '--transcript', action='store_true',
                                help='Download video transcript')
    extraction_group.add_argument('-c', '--comments', action='store_true',
                                help='Get video comments')
    extraction_group.add_argument('-l', '--language', default='en',
                                help='Language code for transcript (default: en)')
    extraction_group.add_argument('--list-languages', action='store_true',
                                help='List available transcript languages')
    
    # Download options
    download_group = parser.add_argument_group('Download Options')
    download_group.add_argument('-d', '--download', action='store_true',
                              help='Download video(s)')
    download_group.add_argument('--threads', type=int, default=2,
                              help='Number of concurrent downloads (default: 2)')
    
    # S3 upload options
    s3_group = parser.add_argument_group('S3 Upload Options')
    s3_group.add_argument('--upload-to-s3', action='store_true',
                          help='Upload downloaded videos to S3')
    s3_group.add_argument('--s3-bucket', type=str, default=None,
                          help='S3 bucket name for uploads')
    s3_group.add_argument('--s3-region', type=str, default=None, 
                          help='AWS region for S3 uploads (defaults to AWS_REGION environment variable)')
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Get API key from environment variable if not provided
    api_key = args.api_key or os.environ.get("YOUTUBE_DATA_API_KEY")
    
    # Initialize YouTube data fetcher
    youtube_fetcher = YouTubeDataFetcher(api_key, output_dir=args.output)
    
    # Extract the video ID (only relevant for single videos)
    video_id = youtube_fetcher.extract_video_id(args.url)
    if video_id and not any(['/playlist' in args.url, '/channel/' in args.url, '/user/' in args.url, '/@' in args.url, 'list=' in args.url]):
        print(f"Working with video ID: {video_id}")
        
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # If download option is selected
    if args.download:
        if not PYTUBEFIX_AVAILABLE:
            print("Error: pytubefix is not installed. Install with: pip install pytubefix")
            return
            
        print(f"Downloading video(s) from {args.url} to {args.output}")
        result = youtube_fetcher.batch_download_videos(args.url, max_workers=args.threads)
        
        if result['success']:
            print(f"\nDownload Summary:")
            print(f"Total videos: {result['total']}")
            print(f"Successfully downloaded: {result['successful']}")
            print(f"Failed: {result['failed']}")
        else:
            print(f"Error: {result['error']}")
        
        # Exit if only download was requested
        if not any([args.info, args.transcript, args.comments, args.list_languages]):
            return
            
    # List available transcript languages
    if args.list_languages and api_key:
        print("\nAvailable transcript languages:")
        languages = youtube_fetcher.list_available_transcript_languages(args.url)
        if languages:
            print(f"Found {len(languages)} available languages:")
            for lang in languages:
                auto_gen = "(auto-generated)" if lang['is_auto_generated'] else ""
                print(f"{lang['language_code']}: {lang['language_name']} {auto_gen}")
        else:
            print("No transcript languages found or API key invalid.")
            print("Note: Some videos may not have any captions available.")
            
    # Get video information
    if args.info and api_key:
        print("\nVideo Information:")
        video_details = youtube_fetcher.get_video_details(args.url)
        if video_details:
            print(f"Title: {video_details.get('title')}")
            print(f"Channel: {video_details.get('channel_title')}")
            print(f"Published: {video_details.get('published_at')}")
            print(f"Views: {video_details.get('view_count')}")
            print(f"Likes: {video_details.get('like_count')}")
            print(f"Comments: {video_details.get('comment_count')}")
            print(f"Duration: {video_details.get('duration')}")
            print(f"Has captions: {video_details.get('caption')}")
        else:
            print("Could not retrieve video details or API key invalid.")
    
    # Get video comments
    if args.comments and api_key:
        print("\nTop Comments:")
        comments = youtube_fetcher.get_video_comments(args.url, max_results=10)
        if comments:
            for i, comment in enumerate(comments, 1):
                print(f"{i}. {comment.get('author')}: {comment.get('text')[:100]}...")
                if i < len(comments):
                    print("-" * 40)
        else:
            print("No comments found or API key invalid.")
            
    # Download transcript
    if args.transcript and api_key:
        print(f"\nAttempting to download transcript in language: {args.language}")
        
        # First check if this language is available
        available_languages = youtube_fetcher.list_available_transcript_languages(args.url)
        language_codes = [lang['language_code'] for lang in available_languages]
        
        if not available_languages:
            print("No transcripts available for this video.")
        elif args.language not in language_codes:
            print(f"Error: Language '{args.language}' is not available for this video.")
            print("Available language options are:")
            for lang in available_languages:
                auto_gen = "(auto-generated)" if lang['is_auto_generated'] else ""
                print(f"  {lang['language_code']}: {lang['language_name']} {auto_gen}")
            print(f"\nTry using -l or --language with one of these language codes.")
        else:
            # Proceed with download if language is available
            result = youtube_fetcher.download_transcript(args.url, language=args.language)
            if result and result['success']:
                print(f"Transcript downloaded successfully to: {result['filepath']}")
            else:
                error = result['error'] if result and 'error' in result else "Unknown error"
                print(f"Failed to download transcript: {error}")
            
    print("\nAll requested operations completed.")

if __name__ == "__main__":
    sys.exit(main())

# Test function for the progress_callback fix
def test_progress_callback():
    """Test the progress_callback method with a mock stream object."""
    print("Running progress_callback test...")
    
    class MockStream:
        def __init__(self):
            self.filesize = 1000
            self.itag = 22  # Common YouTube itag
            # Note: deliberately not setting video_id to test our fix
    
    fetcher = YouTubeDataFetcher(api_key=None)  # No API key needed for this test
    mock_stream = MockStream()
    
    # This should work without errors now
    try:
        fetcher.progress_callback(mock_stream, b"chunk", 500)  # 50% downloaded
        print("Progress callback test passed!")
        return True
    except Exception as e:
        print(f"Progress callback test failed with error: {e}")
        return False

# Allow running this test directly
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "test":
    success = test_progress_callback()
    print(f"Test result: {'PASS' if success else 'FAIL'}")