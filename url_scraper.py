#!/usr/bin/env python3
import re
import os
import base64
import json
import sys
import tempfile
import requests
import pypandoc
from typing import Optional, List, Dict, Any
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout, Error as PlaywrightError
from litellm import litellm, completion
# litellm._turn_on_debug() # Keep commented unless debugging litellm

import PyPDF2
from io import BytesIO
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs, urlencode
# --- Import New Twitter Modules ---
from twitter_session_manager import TwitterSessionManager
from twitter_api_client import TwitterAPIClient
from twitter_content_extractor import TwitterContentExtractor
from twitter_media_downloader import TwitterMediaDownloader
# ---

import logging # Ensure logging is configured if not already
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Existing YouTubeProcessor (No changes needed here unless API key handling changes) ---
class YouTubeProcessor:
    """Helper class for processing YouTube content using the YouTube Data API."""

    def __init__(self, api_key: str):
        """Initialize YouTube API client."""
        try:
            self.youtube = build('youtube', 'v3', developerKey=api_key)
            self.api_key_valid = True
        except Exception as e:
             logger.error(f"Failed to initialize YouTube API client: {e}")
             self.youtube = None
             self.api_key_valid = False

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        if 'youtu.be' in url:
            path_part = urlparse(url).path
            if path_part:
                return path_part.lstrip('/')
        parsed_url = urlparse(url)
        if 'youtube.com' in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            if 'v' in query_params:
                return query_params['v'][0]
        return None

    @staticmethod
    def extract_playlist_id(url: str) -> Optional[str]:
        """Extract playlist ID from YouTube URL."""
        parsed_url = urlparse(url)
        if 'youtube.com' in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            if 'list' in query_params:
                return query_params['list'][0]
        return None

    def get_video_details(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Fetch video details using YouTube API."""
        if not self.api_key_valid or not self.youtube:
             logger.error("YouTube API key is invalid or client not initialized.")
             return None
        try:
            request = self.youtube.videos().list(
                part="snippet,statistics",
                id=video_id
            )
            response = request.execute()

            if not response.get('items'):
                logger.warning(f"No video details found for YouTube video ID: {video_id}")
                return None

            video = response['items'][0]
            snippet = video.get('snippet', {})
            statistics = video.get('statistics', {})

            return {
                'title': snippet.get('title', 'N/A'),
                'description': snippet.get('description', ''),
                'published_at': snippet.get('publishedAt', ''),
                'channel_title': snippet.get('channelTitle', 'N/A'),
                'view_count': statistics.get('viewCount', 'N/A'),
                'like_count': statistics.get('likeCount', 'N/A')
            }

        except Exception as e:
            logger.error(f"Error fetching YouTube video details for ID {video_id}: {str(e)}")
            return None

    def get_playlist_details(self, playlist_id: str, max_items: int = 10) -> Optional[Dict[str, Any]]: # Reduced default max_items
        """Fetch playlist details and items using YouTube API."""
        if not self.api_key_valid or not self.youtube:
             logger.error("YouTube API key is invalid or client not initialized.")
             return None
        try:
            playlist_request = self.youtube.playlists().list(
                part="snippet",
                id=playlist_id
            )
            playlist_response = playlist_request.execute()

            if not playlist_response.get('items'):
                 logger.warning(f"No playlist details found for YouTube playlist ID: {playlist_id}")
                 return None

            playlist = playlist_response['items'][0]
            snippet = playlist.get('snippet', {})

            items_request = self.youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=min(max_items, 50) # API max is 50
            )
            items_response = items_request.execute()

            videos = []
            for item in items_response.get('items', []):
                video_snippet = item.get('snippet', {})
                resource_id = video_snippet.get('resourceId', {})
                if video_snippet and resource_id.get('kind') == 'youtube#video':
                    videos.append({
                        'title': video_snippet.get('title', 'N/A'),
                        'video_id': resource_id.get('videoId'),
                        'position': video_snippet.get('position', -1) + 1
                    })

            return {
                'title': snippet.get('title', 'N/A'),
                'description': snippet.get('description', ''),
                'channel_title': snippet.get('channelTitle', 'N/A'),
                'published_at': snippet.get('publishedAt', ''),
                'item_count': playlist.get('contentDetails', {}).get('itemCount', len(videos)), # Get total count if available
                'videos_preview': videos # List first few videos
            }

        except Exception as e:
            logger.error(f"Error fetching YouTube playlist details for ID {playlist_id}: {str(e)}")
            return None

# --- End YouTubeProcessor ---


class URLScraper:
    # --- Existing static methods (clean_url, is_valid_url, etc.) ---
    # Keep these as they are useful helpers.
    # Ensure is_twitter_url checks for x.com too.
    @staticmethod
    def clean_url(url: str) -> str:
        """Clean and normalize URL."""
        url = url.strip()
        # Remove trailing punctuation that might be artifacts
        url = re.sub(r'[\]\)]+$', '', url)
        # Handle markdown link leftovers if any
        url = url.strip('<>')
        # Normalize domain for Twitter/X
        url = url.replace('//x.com', '//twitter.com') # Prefer twitter.com for consistency internally? Or keep x.com? Let's keep x.com for now.
        # Remove tracking parameters, common ones
        parsed = urlparse(url)
        query = parse_qs(parsed.query, keep_blank_values=True)
        # Common trackers to remove (add more as needed)
        trackers = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 'fbclid', 'gclid', 'mc_cid', 'mc_eid', 'si']
        filtered_query = {k: v for k, v in query.items() if k.lower() not in trackers}
        # Rebuild URL without trackers and fragment unless it's a page/section marker
        fragment = ''
        if parsed.fragment and ('page=' in parsed.fragment or 'section=' in parsed.fragment or parsed.fragment.startswith('h.')): # Keep PDF page fragments or specific section markers
             fragment = parsed.fragment

        url_no_trackers = parsed._replace(query=urlencode(filtered_query, doseq=True), fragment=fragment).geturl()
        # Remove trailing slash if it's not the root path
        if url_no_trackers.endswith('/') and url_no_trackers.count('/') > 2:
             url_no_trackers = url_no_trackers.rstrip('/')

        return url_no_trackers

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Validate URL structure."""
        try:
            result = urlparse(url)
            # Requires scheme (http/https) and netloc (domain)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except ValueError: # Handle potential errors from urlparse on weird inputs
            return False

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract and clean URLs from text. Handles markdown."""
        if not text: return []

        # Enhanced regex to capture URLs within markdown and plain text
        # Avoids capturing markdown syntax itself, captures URLs ending with ) if part of the URL
        url_pattern = r'https?://(?:[a-zA-Z0-9\-_./?=&%~+:@!$,]|(?<=\()\S+(?=\))|\(\S+\))+'
        urls = re.findall(url_pattern, text)

        # Extract from explicit markdown links [text](url)
        markdown_pattern = r'\[[^\]]*\]\((?P<url>https?://[^\)]+)\)'
        md_urls = [match.group('url') for match in re.finditer(markdown_pattern, text)]

        # Combine, clean, and deduplicate
        all_urls = list(dict.fromkeys([URLScraper.clean_url(u) for u in urls + md_urls]))
        return [u for u in all_urls if URLScraper.is_valid_url(u)] # Final validity check

    @staticmethod
    def is_twitter_url(url: str) -> bool:
        """Check if URL is a Twitter/X status/post URL."""
        # Match twitter.com/user/status/id or x.com/user/status/id
        return bool(re.match(r'https?://(?:www\.)?(?:twitter\.com|x\.com)/[^/]+/status/\d+', url))

    @staticmethod
    def is_github_repo_url(url: str) -> bool:
        """Check if URL is a GitHub repository or sub-path."""
        # Matches github.com/user/repo or github.com/user/repo/path...
        pattern = r'https?://github\.com/[^/]+/[^/]+(/.*)?'
        return bool(re.match(pattern, url))

    @staticmethod
    def is_pdf_url(url: str, check_headers: bool = True) -> bool:
        """Check if URL likely points to a PDF."""
        url_lower = url.lower().split('?')[0].split('#')[0] # Check path before query/fragment
        if url_lower.endswith('.pdf'):
            return True
        if 'arxiv.org/pdf/' in url_lower:
             return True

        if check_headers:
            try:
                # Use HEAD request to check Content-Type without downloading body
                response = requests.head(url, allow_redirects=True, timeout=10)
                response.raise_for_status() # Check for HTTP errors
                content_type = response.headers.get('Content-Type', '').lower()
                if 'application/pdf' in content_type:
                    return True
            except requests.exceptions.RequestException as e:
                 # Log warning but don't fail if HEAD request has issues (e.g., HEAD not allowed)
                 logger.debug(f"HEAD request for PDF check failed for {url}: {e}. Relying on URL pattern.")
            except Exception as e:
                 logger.debug(f"Unexpected error during PDF check HEAD request for {url}: {e}")

        return False # Default to false if pattern and headers don't confirm

    @staticmethod
    def is_youtube_url(url: str) -> bool:
        """Check if URL is a YouTube video or playlist."""
        # Updated patterns for common YouTube URL formats
        video_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?.*v=[\w-]+',
            r'https?://youtu\.be/[\w-]+',
            r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
            r'https?://(?:www\.)?youtube\.com/v/[\w-]+',
            r'https?://m\.youtube\.com/watch\?.*v=[\w-]+' # Mobile
        ]
        playlist_patterns = [
            r'https?://(?:www\.)?youtube\.com/playlist\?.*list=[\w-]+',
            r'https?://m\.youtube\.com/playlist\?.*list=[\w-]+' # Mobile
        ]
        return any(re.match(p, url) for p in video_patterns + playlist_patterns)

    # --- Keep PDF processing methods ---
    @staticmethod
    def download_pdf(url: str) -> Optional[bytes]:
        logger.debug(f"Downloading PDF from: {url}")
        try:
            response = requests.get(url, timeout=30, headers={ # Add a user-agent
                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                 })
            response.raise_for_status()
            # Check content type again just in case
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' not in content_type:
                 logger.warning(f"Downloaded content from {url} does not have PDF Content-Type: {content_type}")
                 # Decide whether to proceed or return None
                 # Let's proceed but log, PyPDF2 will likely fail if it's not PDF
            return response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None

    @staticmethod
    def extract_pdf_text(pdf_content: bytes) -> str:
        """Extract text content from PDF bytes."""
        logger.debug("Extracting text from PDF content...")
        text_content = []
        try:
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file, strict=False) # Use strict=False for leniency

            # Check if PDF is encrypted and cannot be read
            if pdf_reader.is_encrypted:
                 logger.warning("PDF is encrypted and cannot be processed.")
                 # Try decrypting with an empty password (common for some PDFs)
                 try:
                     if pdf_reader.decrypt('') == 0: # 0 means decryption failed
                         return "[Error: PDF is encrypted]"
                     # If decryption succeeded (returns 1 or 2), proceed
                 except NotImplementedError:
                      logger.error("PDF decryption algorithm not supported by PyPDF2.")
                      return "[Error: Unsupported PDF encryption]"
                 except Exception as decrypt_e:
                      logger.error(f"Error during PDF decryption attempt: {decrypt_e}")
                      return "[Error: Failed to decrypt PDF]"


            num_pages = len(pdf_reader.pages)
            logger.debug(f"PDF has {num_pages} pages.")
            for i, page in enumerate(pdf_reader.pages):
                 try:
                     page_text = page.extract_text()
                     if page_text:
                          text_content.append(page_text)
                     # else: logger.debug(f"Page {i+1} yielded no text.")
                 except Exception as page_e:
                      logger.warning(f"Error extracting text from PDF page {i+1}: {page_e}")
                      # Continue with other pages

            full_text = "\n".join(text_content).strip()
            logger.debug(f"Extracted {len(full_text)} characters from PDF.")
            # Basic cleanup
            full_text = re.sub(r'\s+\n', '\n', full_text) # Remove trailing spaces before newlines
            full_text = re.sub(r'\n{3,}', '\n\n', full_text) # Collapse excessive newlines
            return full_text

        except PyPDF2.errors.PdfReadError as pdf_err:
             logger.error(f"PyPDF2 error reading PDF: {pdf_err}")
             return "[Error: Invalid or corrupt PDF file]"
        except Exception as e:
            logger.error(f"Unexpected error extracting PDF text: {str(e)}")
            return "[Error: Failed to extract text from PDF]"

    # --- Keep LLM summarization methods ---
    @staticmethod
    def _call_llm_completion(prompt: str, text_model: str, max_tokens: int = 150) -> str:
         """Helper function to call the LLM completion API."""
         try:
             messages = [{"content": prompt, "role": "user"}]
             # Use context manager for potential temporary env var setting if needed
             # with litellm.utils.set_verbose(False): # Reduce litellm verbosity if desired
             response = completion(
                 model=text_model,
                 messages=messages,
                 max_tokens=max_tokens, # Limit output size
                 temperature=0.5, # Lower temperature for factual summary
                 # Add other parameters like top_p if needed
             )
             # Accessing content correctly for LiteLLM v1+
             if response.choices and response.choices[0].message and response.choices[0].message.content:
                  summary = response.choices[0].message.content.strip()
                  # Optional: Post-process summary (remove boilerplate, etc.)
                  return summary
             else:
                  logger.warning(f"LLM response structure invalid or content missing. Response: {response}")
                  return "Summary generation failed (Invalid LLM response)."

         except Exception as e:
             # Catch specific LiteLLM errors if possible
             logger.error(f"Error calling LLM ({text_model}) for summary: {e}")
             # Log traceback for detailed debugging if needed
             # import traceback
             # logger.error(traceback.format_exc())
             return f"Error generating summary: {e}"


    @staticmethod
    def get_webpage_summary(text: str, text_model: str = "ollama/llama3.2:3b") -> str:
        """Get summary of webpage content using LLM."""
        if not text: return "No content to summarize."

        # Truncate text (use a token estimator or char count)
        # Average token length is ~4 chars. Target ~3500 tokens = ~14000 chars.
        max_chars = 14000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.debug(f"Truncated webpage text to {max_chars} chars for summarization.")

        prompt = f"""Please provide a concise summary (around 100 words) of the following webpage content. Focus on the main topic, key points, and conclusions.

Webpage Content:
\"\"\"
{text}
\"\"\"

Summary:"""
        return URLScraper._call_llm_completion(prompt, text_model, max_tokens=150)


    @staticmethod
    def get_pdf_summary(text: str, text_model: str = "ollama/llama3.2:3b") -> str:
        """Get summary of PDF content using LLM."""
        if not text: return "No content to summarize."
        if text.startswith("[Error:"): return text # Pass through extraction errors

        max_chars = 14000 # Same limit as webpage
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.debug(f"Truncated PDF text to {max_chars} chars for summarization.")

        prompt = f"""Please provide a concise summary (around 100 words) of the following document content, likely from a PDF. Focus on the main topic, key findings, arguments, or purpose.

Document Content:
\"\"\"
{text}
\"\"\"

Summary:"""
        return URLScraper._call_llm_completion(prompt, text_model, max_tokens=150)


    @staticmethod
    def get_github_summary(repo_info: Dict[str, Any], text_model: str = "ollama/llama3.2:3b") -> str:
        """Get summary of GitHub repo content using LLM."""
        # repo_info expected keys: url, about, readme_text, stats (dict with stars, forks, watchers)
        if not repo_info: return "No GitHub info provided."

        content_for_summary = f"""
Repository URL: {repo_info.get('url', 'N/A')}
About: {repo_info.get('about', 'N/A')}

README/Content Snippet:
\"\"\"
{repo_info.get('readme_text', 'N/A')[:2000]}...
\"\"\"

Repository Statistics:
- Stars: {repo_info.get('stats', {}).get('stars', 'N/A')}
- Forks: {repo_info.get('stats', {}).get('forks', 'N/A')}
- Watchers: {repo_info.get('stats', {}).get('watchers', 'N/A')}
"""
        # Truncate for LLM
        max_chars = 14000
        if len(content_for_summary) > max_chars:
             content_for_summary = content_for_summary[:max_chars] + "..."
             logger.debug("Truncated GitHub combined info for summarization.")


        prompt = f"""Please provide a concise summary (around 100 words) of this GitHub project based on the provided information. Focus on its purpose, key features (if mentioned), and target audience. Ignore GitHub interface elements.

Repository Information:
\"\"\"
{content_for_summary}
\"\"\"

Summary:"""
        return URLScraper._call_llm_completion(prompt, text_model, max_tokens=150)

    @staticmethod
    def get_youtube_summary(details: Dict[str, Any], content_type: str, text_model: str) -> str:
        """Generate a summary string for YouTube content."""
        if not details: return "Could not retrieve YouTube details."

        if content_type == 'video':
             # Limit description length for summary prompt
             description_snippet = (details.get('description', '')[:500] + '...') if len(details.get('description', '')) > 500 else details.get('description', '')
             summary_text = (
                 f"YouTube Video: \"{details.get('title', 'N/A')}\"\n"
                 f"Channel: {details.get('channel_title', 'N/A')}\n"
                 f"Views: {details.get('view_count', 'N/A')}, Likes: {details.get('like_count', 'N/A')}\n"
                 f"Published: {details.get('published_at', '')}\n"
                 f"Description Snippet: {description_snippet}"
             )
             # Optional: Could feed this to an LLM for a more narrative summary
             return summary_text
        elif content_type == 'playlist':
             description_snippet = (details.get('description', '')[:500] + '...') if len(details.get('description', '')) > 500 else details.get('description', '')
             video_titles = [f"- \"{v['title']}\"" for v in details.get('videos_preview', [])[:5]] # Preview first 5 videos
             summary_text = (
                 f"YouTube Playlist: \"{details.get('title', 'N/A')}\"\n"
                 f"Channel: {details.get('channel_title', 'N/A')}\n"
                 f"Total Videos: {details.get('item_count', 'N/A')}\n"
                 f"Published: {details.get('published_at', '')}\n"
                 f"Description Snippet: {description_snippet}\n"
                 f"Video Preview:\n" + "\n".join(video_titles)
             )
             return summary_text
        else:
             return "Unknown YouTube content type."

    # --- Keep Pandoc extraction ---
    @staticmethod
    def extract_content_with_pandoc(html: str) -> str:
        """Extract text content from HTML using Pandoc."""
        if not html: return ""
        logger.debug("Extracting content using Pandoc...")
        try:
            # Use text input directly if possible, avoiding temp file for performance
            text = pypandoc.convert_text(
                html,
                'plain',
                format='html',
                extra_args=['--wrap=none', '--strip-comments', '--reference-links'] # reference-links might help clean link clutter
            )
            # Basic cleaning
            text = re.sub(r'\n{3,}', '\n\n', text) # Collapse excessive newlines
            text = re.sub(r' {2,}', ' ', text) # Collapse multiple spaces
            text = text.strip()
            logger.debug(f"Pandoc extracted {len(text)} characters.")
            return text

        except OSError as e:
             # Handle common Pandoc not found error
             if "No such file" in str(e) or "cannot find" in str(e).lower():
                  logger.error("Pandoc command not found. Please ensure Pandoc is installed and in your system's PATH.")
                  return "[Error: Pandoc not installed or not found in PATH]"
             else:
                  logger.error(f"Pandoc OS Error: {e}")
                  return f"[Error: Pandoc execution failed ({e})]"
        except Exception as e:
            logger.error(f"Error converting HTML with Pandoc: {str(e)}")
            return f"[Error: Pandoc conversion failed ({e})]"

    # --- Keep Google Search fallback ---
    @staticmethod
    def get_url_info_from_search(url: str) -> Optional[str]:
        """Get information about a URL using Google Custom Search API."""
        logger.debug(f"Attempting fallback search for URL: {url}")
        api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        cx = os.getenv('GOOGLE_SEARCH_ENGINE_ID')

        if not api_key or not cx:
            logger.warning("Google Search API credentials (GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_ENGINE_ID) not configured. Skipping search fallback.")
            return None

        try:
            service = build("customsearch", "v1", developerKey=api_key)
            # Search specifically for the URL
            query = f'"{url}"' # Exact URL search
            result = service.cse().list(q=query, cx=cx, num=1).execute() # Fetch only top result

            if 'items' in result and result['items']:
                item = result['items'][0]
                title = item.get('title')
                snippet = item.get('snippet')
                info = []
                if title: info.append(f"Title: {title}")
                if snippet: info.append(f"Description: {snippet.replace('...', '').strip()}") # Clean snippet

                if info:
                    logger.info(f"Found info via Google Search for {url}")
                    return " | ".join(info)
                else:
                    logger.info(f"Google Search returned result for {url}, but no usable title/snippet.")
            else:
                 logger.info(f"Google Search found no results for {url}.")
            return None

        except Exception as e:
            logger.error(f"Error performing Google Search for {url}: {str(e)}")
            return None

    # --- Console message filter (keep as is) ---
    @staticmethod
    def filter_console_message(msg):
        # ... (keep the existing implementation) ...
        ignore_patterns = [
            "Failed to load resource", "net::ERR_", "status of 403",
            "status of 404", "status of 429", "favicon.ico"
        ]
        msg_text = msg.text().lower() if hasattr(msg, 'text') else ''
        msg_type = msg.type().lower() if hasattr(msg, 'type') else ''

        if any(pattern.lower() in msg_text for pattern in ignore_patterns):
            return False # Silence common errors

        # Silence specific noisy warnings (add more as needed)
        if "Duplicate key 'aria-labelledby'" in msg.text(): return False
        if "DevTools failed to load source map" in msg.text(): return False

        # Log important types, filter others
        if msg_type in ["error", "warning"]:
             # Log JS errors and non-resource warnings
             if "failed to load resource" not in msg_text:
                  return True # Log it
             return False # Silence resource load errors/warnings
        elif msg_type == "log":
            # Decide if general logs are needed (can be very noisy)
            return False # Silence general console.log messages by default
        else:
            return True # Log other types like 'info', 'debug' if they occur


    # --- NEW process_twitter_content method ---
    @staticmethod
    def process_twitter_content(
        url: str,
        session_manager: TwitterSessionManager,
        download_media: bool = False,
        media_output_dir: str = "./downloads"
        ) -> Optional[Dict[str, Any]]:
        """
        Processes Twitter/X content using the new modular structure.
        Ensures valid session, extracts page content, optionally fetches API data and downloads media.
        """
        logger.info(f"Processing Twitter/X URL: {url}")

        # 1. Ensure Session is Valid
        if not session_manager.ensure_valid_session():
            logger.error("Failed to ensure a valid Twitter/X session. Cannot process tweet.")
            return {
                'type': 'twitter',
                'url': url,
                'error': 'Session validation/refresh failed.',
                'content': "[Error: Could not authenticate Twitter/X session]"
            }
        session_path = session_manager.get_session_path()

        # 2. Extract Content Directly from Page
        content_extractor = TwitterContentExtractor()
        page_details = content_extractor.extract_tweet_details_from_page(url, session_path)

        if not page_details or page_details.get('error'):
            error_msg = page_details.get('error') if page_details else "Failed to extract page details (None returned)"
            logger.error(f"Failed to extract base tweet details from page: {error_msg}")
            return {
                'type': 'twitter',
                'url': url,
                'error': f'Page extraction failed: {error_msg}',
                'content': f"[Error: Could not extract tweet content from page - {error_msg}]"
            }

        tweet_id = page_details.get('tweet_id')
        if not tweet_id:
            logger.error("Could not determine Tweet ID from page details.")
            return {
                'type': 'twitter',
                'url': url,
                'error': 'Could not determine Tweet ID',
                'content': "[Error: Could not determine Tweet ID]"
            }

        # 3. Prepare Result Dictionary (start with page details)
        result = {
            'type': 'twitter',
            'url': url,
            'tweet_id': tweet_id,
            'details': page_details, # Contains text, user, time, urls etc.
            'media_urls': [],
            'downloaded_media_paths': [],
            'content': f"Tweet by @{page_details.get('user_handle', 'unknown')}: {page_details.get('text', '[No text content found]')}" # Basic content string
        }


        # 4. Fetch API Data & Media if needed (currently needed for reliable media URLs)
        #    Decide if API call is always needed or only for downloads. Let's call for media URLs always for now.
        media_items = None
        try:
             api_client = TwitterAPIClient(session_path)
             api_data = api_client.fetch_tweet_data_api(tweet_id)
             if api_data:
                  media_items = api_client.extract_media_urls_from_api_data(api_data)
                  if media_items:
                       result['media_urls'] = [item['url'] for item in media_items]
                       logger.info(f"Found {len(media_items)} media items via API.")
                  else:
                       logger.info("No media items found via API.")
             else:
                  logger.warning("Failed to fetch API data for tweet. Media information might be incomplete.")
                  result['error'] = result.get('error', '') + '; API data fetch failed'

        except Exception as api_e:
             logger.error(f"Error during API fetch or media extraction: {api_e}")
             result['error'] = result.get('error', '') + f'; API processing error: {api_e}'


        # 5. Download Media if Requested and Available
        if download_media and media_items:
            logger.info("Media download requested, proceeding with download...")
            try:
                 downloader = TwitterMediaDownloader(output_dir=media_output_dir)
                 downloaded_files_info = downloader.download_media_items(media_items, page_details, tweet_id)
                 result['downloaded_media_paths'] = [f['path'] for f in downloaded_files_info]
                 if result['downloaded_media_paths']:
                      # Append download info to the main content string
                      result['content'] += f"\n[Downloaded {len(result['downloaded_media_paths'])} media file(s)]"
            except Exception as dl_e:
                 logger.error(f"Error during media download process: {dl_e}")
                 result['error'] = result.get('error', '') + f'; Media download error: {dl_e}'
        elif download_media and not media_items:
             logger.info("Media download requested, but no media items were found via API.")


        # 6. Finalize Content String (Could add more details here if needed)
        # The basic content string is already set. Could enhance with media counts etc.
        if result['media_urls']:
             result['content'] += f"\n[Media detected: {len(result['media_urls'])} item(s)]"


        logger.info(f"✓ Successfully processed Twitter URL: {url}")
        return result


    # --- Updated scrape_url Method ---
    @staticmethod
    def scrape_url(
        url: str,
        text_model: str = "ollama/llama3.2:3b",
        vision_model: str = "ollama/llava:7b", # Keep vision model for non-specialized cases
        download_media: bool = False, # Flag for media download (esp. Twitter)
        media_output_dir: str = "./downloads",
        use_search_fallback: bool = True # Control search fallback
        ) -> Optional[Dict[str, Any]]:
        """
        Scrapes a given URL, handling different content types and using LLMs for summarization.
        Integrates specialized processing for Twitter, YouTube, GitHub, PDFs.
        """
        logger.info(f"Processing URL: {url} (TextModel: {text_model}, VisionModel: {vision_model}, DownloadMedia: {download_media})")

        cleaned_url = URLScraper.clean_url(url)
        if not URLScraper.is_valid_url(cleaned_url):
            logger.error(f"Invalid URL after cleaning: {cleaned_url} (Original: {url})")
            return None

        result = {'url': cleaned_url, 'type': 'unknown', 'content': None, 'error': None}

        # --- Handle Special URL Types First ---
        if URLScraper.is_twitter_url(cleaned_url):
            session_manager = TwitterSessionManager() # Initialize session manager
            # Pass download flag and output dir
            twitter_result = URLScraper.process_twitter_content(
                cleaned_url,
                session_manager,
                download_media=download_media,
                media_output_dir=media_output_dir
                )
            return twitter_result # Return directly, process_twitter_content handles format

        elif URLScraper.is_youtube_url(cleaned_url):
            result['type'] = 'youtube'
            youtube_api_key = os.getenv('YOUTUBE_DATA_API_KEY')
            if not youtube_api_key:
                logger.error("YOUTUBE_DATA_API_KEY not set. Cannot process YouTube URL.")
                result['error'] = 'YouTube API key missing'
                result['content'] = "[Error: YouTube API key not configured]"
            else:
                processor = YouTubeProcessor(youtube_api_key)
                video_id = processor.extract_video_id(cleaned_url)
                playlist_id = processor.extract_playlist_id(cleaned_url)

                if video_id:
                    details = processor.get_video_details(video_id)
                    result['content'] = URLScraper.get_youtube_summary(details, 'video', text_model)
                elif playlist_id:
                    details = processor.get_playlist_details(playlist_id)
                    result['content'] = URLScraper.get_youtube_summary(details, 'playlist', text_model)
                else:
                    result['error'] = "Could not extract YouTube video or playlist ID."
                    result['content'] = "[Error: Invalid YouTube URL format]"
            return result

        elif URLScraper.is_pdf_url(cleaned_url):
            result['type'] = 'pdf'
            pdf_content = URLScraper.download_pdf(cleaned_url)
            if pdf_content:
                 pdf_text = URLScraper.extract_pdf_text(pdf_content)
                 result['content'] = URLScraper.get_pdf_summary(pdf_text, text_model=text_model)
            else:
                 result['error'] = 'Failed to download or process PDF'
                 result['content'] = "[Error: Failed to download or extract PDF content]"
            return result

        # --- Generic Webpage Processing (including GitHub) ---
        # Uses Playwright -> Pandoc -> LLM Summary / Search Fallback
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=[ # Standard args
                    '--disable-gpu', '--disable-dev-shm-usage', '--disable-setuid-sandbox', '--no-sandbox'
                    ])
                context = browser.new_context(
                     user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                     java_script_enabled=True, # Keep JS enabled for rendering, block resources instead
                     # Block images/media/fonts for faster loading by default
                     # Note: This was handled per-page before, context level is simpler
                )
                # Route to block resources at context level
                context.route("**/*", lambda route: route.abort() if route.request.resource_type in ['image', 'stylesheet', 'font', 'media'] else route.continue_())

                page = context.new_page()
                 # Setup console/error logging
                page.on("pageerror", lambda err: logger.warning(f"Page JS error ({cleaned_url}): {err}"))
                page.on("console", lambda msg: print(f"Console [{msg.type()}]: {msg.text()}") if URLScraper.filter_console_message(msg) else None) # Conditional print

                extraction_method = "unknown"
                try:
                    logger.debug(f"Navigating to generic URL: {cleaned_url}")
                    page.goto(cleaned_url, wait_until='domcontentloaded', timeout=30000) # Allow 30s for load
                    # Add small wait for any dynamic content rendering after DOM load
                    page.wait_for_timeout(2000)

                    # Check if it's GitHub after navigation (in case of redirects)
                    final_url = page.url
                    if URLScraper.is_github_repo_url(final_url):
                         result['type'] = 'github'
                         extraction_method = 'github_playwright'
                         logger.info("Processing as GitHub URL...")
                         # Simplified GitHub Extraction: Get Readme/About, Generate Summary
                         repo_info = {'url': final_url, 'about': '', 'readme_text': '', 'stats': {}}
                         try:
                              # Extract "About"
                              about_el = page.query_selector('.BorderGrid--spacious .f4.my-3') # Updated selector? Inspect needed.
                              if about_el: repo_info['about'] = about_el.text_content().strip()

                              # Extract README (common selector)
                              readme_el = page.query_selector('#readme article.markdown-body')
                              if readme_el: repo_info['readme_text'] = readme_el.text_content().strip()

                               # Extract Stats (selectors might change)
                              stars_el = page.query_selector('#repo-stars-counter-star')
                              forks_el = page.query_selector('#repo-network-counter')
                              watchers_el = page.query_selector('#repo-watchers-counter')
                              if stars_el: repo_info['stats']['stars'] = stars_el.get_attribute('title')
                              if forks_el: repo_info['stats']['forks'] = forks_el.get_attribute('title')
                              if watchers_el: repo_info['stats']['watchers'] = watchers_el.get_attribute('title')

                              result['content'] = URLScraper.get_github_summary(repo_info, text_model)
                         except PlaywrightError as gh_err:
                              logger.error(f"Playwright error during GitHub element extraction: {gh_err}")
                              result['error'] = f"GitHub processing error: {gh_err}"
                              result['content'] = "[Error: Could not extract GitHub content]"

                    else:
                         # Generic Webpage: Try Pandoc -> LLM
                         result['type'] = 'webpage'
                         logger.info("Processing as generic webpage...")
                         html_content = page.content()
                         text_content = URLScraper.extract_content_with_pandoc(html_content)

                         if text_content and not text_content.startswith("[Error:") and len(text_content) > 50: # Basic check for valid content
                              extraction_method = 'pandoc_llm'
                              result['content'] = URLScraper.get_webpage_summary(text_content, text_model)
                         else:
                              logger.warning(f"Pandoc extraction yielded minimal or error content ({len(text_content)} chars). Error: {text_content if text_content.startswith('[Error:') else 'None'}")
                              result['error'] = "Pandoc extraction failed or insufficient content."
                              # Trigger search fallback if enabled
                              if use_search_fallback:
                                   logger.info("Attempting Google Search fallback...")
                                   search_content = URLScraper.get_url_info_from_search(cleaned_url)
                                   if search_content:
                                        result['content'] = search_content
                                        extraction_method = 'google_search'
                                        result['error'] = None # Clear previous error if search succeeded
                                   else:
                                        result['content'] = "[Error: Content extraction failed, and search fallback yielded no results]"
                              else:
                                   result['content'] = "[Error: Content extraction failed, search fallback disabled]"


                except PlaywrightTimeout as e:
                    logger.error(f"Timeout loading page: {cleaned_url} - {e}")
                    result['error'] = f"Page load timeout: {e}"
                    # Trigger search fallback if enabled
                    if use_search_fallback:
                         logger.info("Attempting Google Search fallback due to timeout...")
                         search_content = URLScraper.get_url_info_from_search(cleaned_url)
                         if search_content:
                              result['content'] = search_content
                              extraction_method = 'google_search'
                              result['error'] = None # Clear timeout error if search succeeded
                         else:
                              result['content'] = "[Error: Page load timed out, and search fallback yielded no results]"
                    else:
                         result['content'] = "[Error: Page load timed out, search fallback disabled]"

                except PlaywrightError as e:
                     logger.error(f"Playwright error processing {cleaned_url}: {e}")
                     result['error'] = f"Playwright error: {e}"
                     result['content'] = f"[Error: Playwright failed - {e}]" # Keep specific error if possible

                except Exception as e:
                     logger.error(f"Unexpected error during Playwright processing of {cleaned_url}: {e}", exc_info=True)
                     result['error'] = f"Unexpected Playwright error: {e}"
                     result['content'] = "[Error: Unexpected failure during scraping]"

                finally:
                     result['extraction_method'] = extraction_method # Record how content was obtained
                     page.close()
                     context.close()
                     browser.close()

        except Exception as e:
            logger.error(f"General error setting up Playwright for {cleaned_url}: {e}", exc_info=True)
            result['error'] = f"Playwright setup failed: {e}"
            result['content'] = "[Error: Scraper setup failed]"


        # Final check for content
        if result['content'] is None and result['error'] is None:
             logger.warning(f"Processing finished for {cleaned_url}, but no content was generated and no error reported.")
             result['content'] = "[Error: Unknown processing failure - no content extracted]"
        elif result.get('error'):
             logger.error(f"Finished processing {cleaned_url} with error: {result['error']}")
        else:
             logger.info(f"✓ Finished processing {cleaned_url} (Type: {result['type']}, Method: {result.get('extraction_method', 'N/A')})")


        return result
