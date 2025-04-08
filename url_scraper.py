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
from twitter_content_extractor import TweetExtractor
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
    def call_llm_completion(prompt: str, text_model: str, max_tokens: int = 250) -> str:
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
        return URLScraper.call_llm_completion(prompt, text_model, max_tokens=150)

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
        return URLScraper.call_llm_completion(prompt, text_model, max_tokens=150)

    def _scrape_github_repo(
        url: str, 
        browser, 
        text_model: str,
        save_screenshot: bool = True,  # New parameter to control screenshot saving
        screenshot_dir: str = "screenshots"  # Default directory for saved screenshots
    ) -> Dict[str, Any]:
        """
        Scrapes a GitHub repository by using Text LLM Model
        Args:
            url: The GitHub repository URL
            browser: Playwright browser instance
            text_model: The text LLM model to use
            save_screenshot: Whether to save a permanent copy of the screenshot for debugging
            screenshot_dir: Directory where screenshots should be saved
            
        Returns:
            Dictionary with scraped information
        """
        logger.info(f"Processing GitHub repository with vision approach: {url}")
        result = {'url': url, 'type': 'github', 'content': None, 'error': None}
        screenshot_path = None  # Track the saved screenshot path
        
        try:
            # Configure context for better screenshot quality
            context_options = {
                'viewport': {'width': 1280, 'height': 1600},  # Larger viewport to capture more content
                'device_scale_factor': 1,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            }
            
            context = browser.new_context(**context_options)    
            page = context.new_page()
            
            # Navigate to the repo and wait for content to load
            logger.debug(f"Navigating to GitHub repo: {url}")
            page.goto(url, wait_until='domcontentloaded', timeout=30000)
            
            # Wait for README to be visible (if present)
            page.wait_for_timeout(3000)  # Additional time for dynamic content
            
            # Create a unique filename for the screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            repo_parts = url.rstrip('/').split('/')
            repo_identifier = f"{repo_parts[-2]}_{repo_parts[-1]}" if len(repo_parts) >= 2 else "unknown_repo"
            screenshot_filename = f"github_{repo_identifier}_{timestamp}.png"
            
            # Prepare screenshot directory if saving is enabled
            if save_screenshot:
                os.makedirs(screenshot_dir, exist_ok=True)
                screenshot_path = os.path.join(screenshot_dir, screenshot_filename)
                logger.info(f"Will save debug screenshot to: {screenshot_path}")

            # Instead of taking a screenshot for vision model processing
            logger.debug(f"Extracting GitHub repo content using CSS selector")
            
            # Extract the main content using CSS selector
            main_content = None
            try:
                main_element = page.query_selector('main')
                if main_element:
                    main_content = main_element.inner_text()
                    
                    # Also extract repository statistics
                    stats = {}
                    
                    # Try to extract stars, forks, watchers
                    stats_selectors = {
                        'stars': 'a[href$="/stargazers"]',
                        'forks': 'a[href$="/forks"]',
                        'watchers': 'a[href$="/watchers"]',
                        'issues': 'a[href$="/issues"]'
                    }
                    
                    for stat_name, selector in stats_selectors.items():
                        try:
                            stat_element = page.query_selector(selector)
                            if stat_element:
                                stat_text = stat_element.inner_text().strip()
                                stats[stat_name] = stat_text
                        except Exception as stat_err:
                            logger.warning(f"Failed to extract {stat_name} stat: {stat_err}")
                    
                    # Try to get repository title/description
                    repo_title = page.title()
                    repo_description = ""
                    description_element = page.query_selector('p[itemprop="description"]')
                    if description_element:
                        repo_description = description_element.inner_text().strip()
                    
                    # Combine all content for summary
                    content_for_summary = f"Title: {repo_title}\n"
                    content_for_summary += f"Description: {repo_description}\n\n"
                    content_for_summary += f"Statistics:\n"
                    for stat_name, stat_value in stats.items():
                        content_for_summary += f"- {stat_name.capitalize()}: {stat_value}\n"
                    content_for_summary += f"\nMain Content:\n{main_content}"
                    
                    # Create prompt for text-based LLM
                    prompt = f"""
                        Based on the content_for_summary provided to you as follows, generate a concise summary (around 200 words) of this github project page.
                        Disregard whatever is related to the Github service itself. Consider only information related to the particular project this page points to.
                        Make sure you provide Repository Statistics such as: Stars, Forks, and Commit activity as part of the summary.
                        ---
                        Repository Content:
                        {content_for_summary}
                        ---
                        Output must in the following format:
---
Title of the Github Repo:
Summary:
Technologies: (which languages, frameworks, tools were used)
Forks: 
Stars:
Commit Activity: (how many commits and when the last commit)
                    """
                    # Call text-based LLM for summary
                    logger.debug(f"Calling text model ({text_model}) for GitHub repo analysis")
                    summary = URLScraper.call_llm_completion(prompt, text_model, max_tokens=250)
                    
                    # Update result with the summary
                    result['content'] = summary
                    result['extraction_method'] = 'github_text_selector'
                    logger.info(f"Successfully analyzed GitHub repo via text model")
                else:
                    result['error'] = "Could not find main content element on GitHub page"
                    result['content'] = "[Error: Failed to extract GitHub repository content]"
                    logger.error("Could not find main content element on GitHub page")
            except Exception as extract_err:
                logger.error(f"Error extracting GitHub content: {extract_err}", exc_info=True)
                result['error'] = f"Content extraction error: {extract_err}"
                result['content'] = "[Error: Failed to extract GitHub repository content]"
                                
                  
        except PlaywrightTimeout as e:
            logger.error(f"Timeout accessing GitHub repo {url}: {e}")
            result['error'] = f"GitHub page load timeout: {e}"
            result['content'] = "[Error: Timed out loading GitHub repository page]"
            
        except Exception as e:
            logger.error(f"Unexpected error analyzing GitHub repo with vision: {e}", exc_info=True)
            result['error'] = f"Vision analysis error: {e}"
            result['content'] = "[Error: Unexpected failure during GitHub vision analysis]"

        finally:
            # Clean up Playwright resources
            if 'page' in locals():
                page.close()
            if 'context' in locals():
                context.close()
                
        return result

    def scrape_github_repo(
            url: str, 
            browser, 
            text_model: str,
            save_screenshot: bool = True,  # Parameter to control screenshot saving
            screenshot_dir: str = "screenshots"  # Default directory for saved screenshots
        ) -> Dict[str, Any]:
            """
            Scrapes a GitHub repository using a mix of web scraping and LLM inference
            Args:
                url: The GitHub repository URL
                browser: Playwright browser instance
                text_model: The text LLM model to use
                save_screenshot: Whether to save a permanent copy of the screenshot for debugging
                screenshot_dir: Directory where screenshots should be saved
                
            Returns:
                Dictionary with scraped information
            """
            logger.info(f"Processing GitHub repository: {url}")
            result = {'url': url, 'type': 'github', 'content': None, 'error': None}
            screenshot_path = None  # Track the saved screenshot path
            
            try:
                # Configure context for better screenshot quality
                context_options = {
                    'viewport': {'width': 1280, 'height': 1600},  # Larger viewport to capture more content
                    'device_scale_factor': 1,
                    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                }
                
                context = browser.new_context(**context_options)    
                page = context.new_page()
                
                # Navigate to the repo and wait for content to load
                logger.debug(f"Navigating to GitHub repo: {url}")
                page.goto(url, wait_until='domcontentloaded', timeout=30000)
                
                # Wait for content to be visible
                page.wait_for_timeout(3000)  # Additional time for dynamic content
                
                # Create a unique filename for the screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                repo_parts = url.rstrip('/').split('/')
                repo_identifier = f"{repo_parts[-2]}_{repo_parts[-1]}" if len(repo_parts) >= 2 else "unknown_repo"
                screenshot_filename = f"github_{repo_identifier}_{timestamp}.png"
                
                # Prepare screenshot directory if saving is enabled
                if save_screenshot:
                    os.makedirs(screenshot_dir, exist_ok=True)
                    screenshot_path = os.path.join(screenshot_dir, screenshot_filename)
                    logger.info(f"Will save debug screenshot to: {screenshot_path}")

                # Extract information through web scraping
                logger.debug(f"Extracting GitHub repo content using CSS selectors")
                
                # 1. Extract repository statistics
                stats = {}
                
                # Stars count
                stars_elem = page.query_selector('a[href$="/stargazers"] .text-bold')
                if stars_elem:
                    stats['stars'] = stars_elem.inner_text().strip()
                else:
                    stats['stars'] = "0"
                
                # Forks count
                forks_elem = page.query_selector('a[href$="/forks"] .text-bold')
                if forks_elem:
                    stats['forks'] = forks_elem.inner_text().strip()
                else:
                    stats['forks'] = "0"
                
                # Watching count
                watching_elem = page.query_selector('a[href$="/watchers"] .text-bold')
                if watching_elem:
                    stats['watching'] = watching_elem.inner_text().strip()
                else:
                    stats['watching'] = "0"
                
                # 2. Extract repository title and description
                repo_title = ""
                title_elem = page.query_selector('strong[itemprop="name"] a')
                if title_elem:
                    repo_title = title_elem.inner_text().strip()
                else:
                    # Fallback to URL if title element not found
                    repo_title = url.split('/')[-1]
                
                repo_description = ""
                description_elem = page.query_selector('p.f4.mb-3')
                if description_elem:
                    repo_description = description_elem.inner_text().strip()
                
                # 3. Extract commit activity information
                commit_message = ""
                commit_date = ""
                commit_count = "Unknown"
                
                # Last commit message
                message_elem = page.query_selector('[data-testid="latest-commit-html"] a')
                if message_elem:
                    commit_message = message_elem.inner_text().strip()
                
                # Last commit date
                date_elem = page.query_selector('[data-testid="latest-commit"] relative-time')
                if date_elem:
                    commit_date = date_elem.get_attribute('title') or date_elem.inner_text().strip()
                
                # Commit count using the specific XPath
                try:
                    xpath_selector = '//*[@id="repo-content-pjax-container"]/div/div/div/div[1]/react-partial/div/div/div[3]/div[1]/table/tbody/tr[1]/td/div/div[2]/div[2]/a/span/span[2]/span'
                    commit_count_elem = page.query_selector(f"xpath={xpath_selector}")
                    if commit_count_elem:
                        commit_count = commit_count_elem.inner_text().strip()
                        # Remove any text, keep only the number
                        commit_count = ''.join(filter(str.isdigit, commit_count))
                except Exception as e:
                    logger.warning(f"Failed to extract commit count using XPath: {e}")
                    # Fallback to previous method
                    commit_elem = page.query_selector('a.Link--primary span.fgColor-default')
                    if commit_elem and "Commits" in commit_elem.inner_text():
                        commit_count = commit_elem.inner_text().split()[0].strip()
                
                # 4. Extract languages used
                languages = []
                lang_elements = page.query_selector_all('.d-inline .d-inline-flex.flex-items-center')
                for lang_elem in lang_elements:
                    lang_name = lang_elem.query_selector('.text-bold')
                    if lang_name:
                        languages.append(lang_name.inner_text().strip())
                
                # 5. Get README content for summary
                readme_content = ""
                readme_elem = page.query_selector('article.markdown-body')
                if readme_elem:
                    readme_content = readme_elem.inner_text()
                
                # Construct content_for_summary from all gathered information
                content_for_summary = f"Title: {repo_title}  "
                content_for_summary += f"Description: {repo_description}\n\n"
                content_for_summary += f"Statistics:\n"
                content_for_summary += f"- Stars: {stats.get('stars', '0')}\n"
                content_for_summary += f"- Forks: {stats.get('forks', '0')}\n"
                content_for_summary += f"- Watchers: {stats.get('watching', '0')}\n"
                content_for_summary += f"Last Commit: {commit_message} ({commit_date})\n"
                
                if languages:
                    content_for_summary += f"Languages: {', '.join(languages)}\n\n"
                
                content_for_summary += f"README Content:\n{readme_content}"
                
                # Create LLM prompt for summary generation
                prompt = f"""
    Based on the content_for_summary provided to you as follows, generate a concise summary (around 200 words) of this github project page.
    Disregard whatever is related to the Github service itself. Consider only information related to the particular project this page points to.
    ---
    Repository Content:
    {content_for_summary}
    ---
    """
                
                # Call text-based LLM for summary
                logger.debug(f"Calling text model ({text_model}) for GitHub repo analysis")
                summary = URLScraper.call_llm_completion(prompt, text_model, max_tokens=250)
                
                # Format the output according to requirements
                technologies = ", ".join(languages) if languages else "Not specified"
                commit_activity = f"{commit_count} commits, last commit on {commit_date}"
                
                formatted_output = f"""
Title: {repo_title}  
    
Summary:   
{summary}  

Technologies: {technologies}  
Forks: {stats.get('forks', '0')}  
Stars: {stats.get('stars', '0')}  
Commit Activity: {commit_activity}  
"""
                
                # Update result with the formatted summary
                result['content'] = formatted_output.strip()
                result['extraction_method'] = 'github_web_scraping_with_llm'
                logger.info(f"Successfully analyzed GitHub repo")
                
                # Save screenshot if enabled
                if save_screenshot and screenshot_path:
                    page.screenshot(path=screenshot_path)
                    logger.debug(f"Saved screenshot to {screenshot_path}")
                    
            except PlaywrightTimeout as e:
                logger.error(f"Timeout accessing GitHub repo {url}: {e}")
                result['error'] = f"GitHub page load timeout: {e}"
                result['content'] = "[Error: Timed out loading GitHub repository page]"
                
            except Exception as e:
                logger.error(f"Unexpected error analyzing GitHub repo: {e}", exc_info=True)
                result['error'] = f"Analysis error: {e}"
                result['content'] = "[Error: Unexpected failure during GitHub analysis]"

            finally:
                # Clean up Playwright resources
                if 'page' in locals():
                    page.close()
                if 'context' in locals():
                    context.close()
                    
            return result

    @staticmethod
    def get_youtube_summary(details: Dict[str, Any], content_type: str, text_model: str) -> str:
        """Generate a summary string for YouTube content."""
        if not details: return "Could not retrieve YouTube details."

        if content_type == 'Video':
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

    @staticmethod
    def filter_console_message(msg):
        # Check if 'text' is a method or attribute
        if hasattr(msg, 'text'):
            if callable(msg.text):
                msg_text = msg.text().lower()
            else:
                msg_text = msg.text.lower()
        else:
            msg_text = ''
            
        # Same check for 'type'
        if hasattr(msg, 'type'):
            if callable(msg.type):
                msg_type = msg.type().lower()
            else:
                msg_type = msg.type.lower()
        else:
            msg_type = ''

        ignore_patterns = [
            "Failed to load resource", "net::ERR_", "status of 403",
            "status of 404", "status of 429", "favicon.ico"
        ]

        if any(pattern.lower() in msg_text for pattern in ignore_patterns):
            return False # Silence common errors

        # Silence specific noisy warnings (add more as needed)
        if "Duplicate key 'aria-labelledby'" in msg_text: return False
        if "DevTools failed to load source map" in msg_text: return False

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
        Processes Twitter/X content using the modular structure.
        Ensures valid session, extracts page content, fetches API data and downloads media if requested.
        """
        logger.info(f"Processing Twitter/X URL: {url}")

        # 1. Ensure Session is Valid
        if not session_manager.ensure_valid_session():
            logger.error("Failed to ensure a valid Twitter/X session. Cannot process tweet.")
            return {
                'type': 'Twitter',
                'url': url,
                'error': 'Session validation/refresh failed.',
                'content': "[Error: Could not authenticate Twitter/X session]"
            }
        session_path = session_manager.get_session_path()

        # 2. Extract Content Directly from Page
        content_extractor = TweetExtractor()
        page_details = content_extractor.extract_tweet(url, session_path)

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

        result = {
            'type': 'twitter',
            'url': url,
            'tweet_id': tweet_id,
            'details': page_details, # Contains text, user, time, urls etc.
            'media_urls': [],
            'downloaded_media_paths': [],
            'extraction_method': 'playwright_bs4',
        }
        
        # Create a formatted content string similar to _format_tweet_output
        user_handle = page_details.get('user_handle', 'unknown')
        user_name = page_details.get('user_name', '')
        tweet_text = page_details.get('text', '[No text content found]')
        timestamp = page_details.get('timestamp', '')
        urls = page_details.get('urls', [])
        embedded_urls = page_details.get('embedded_urls', [])
        
        if timestamp:
            try:
                # Check if it's a numeric timestamp (unix timestamp)
                if str(timestamp).isdigit() or (isinstance(timestamp, (int, float))):
                    # Convert unix timestamp to datetime and format
                    dt = datetime.fromtimestamp(int(timestamp))
                    formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
                else:
                    # If it's already a string format, use it as is
                    formatted_timestamp = timestamp
            except Exception as e:
                logger.warning(f"Error formatting timestamp {timestamp}: {e}")
                formatted_timestamp = str(timestamp)  # Fallback to 
    
        # Format the output with the requested style
        content_parts = [
            f"User: @{user_handle} ({user_name})  ",
            f"Tweet ID: {tweet_id}  "  ,
            f"Posted on: {formatted_timestamp}  " if formatted_timestamp else "Posted on: Unknown",
            "",
            "---",
            tweet_text,
        ]
        
        # Add URLs section if any are found (combining both regular and embedded URLs)
        all_urls = []
        if urls and len(urls) > 0:
            all_urls.extend(urls)
        
        if embedded_urls and len(embedded_urls) > 0:
            all_urls.extend(embedded_urls)
            
        if all_urls:
            content_parts.append("URLs in tweet:")
            for url in all_urls:
                content_parts.append(f" - {url}")
            # content_parts.append("---")
                
        # Create the content string
        result['content'] = "\n".join(content_parts)

        # 4. Fetch API Data & Media URLs
        media_items = []  # Initialize as empty list instead of None
        try:
            api_client = TwitterAPIClient(session_path)
            # CRITICAL: First get the tokens
            if not api_client._get_tokens():
                logger.error("Failed to get authentication tokens")
                result['error'] = 'Failed to get authentication tokens'
            else:
                # Now fetch tweet data for media extraction
                api_data = api_client.fetch_tweet_data_api(tweet_id)
                
                if api_data:
                    # Extract media URLs - this is the critical step for downloading
                    media_items = api_client.extract_media_urls_from_api_data(api_data)
                    if media_items:
                        result['media_urls'] = [item['url'] for item in media_items if item.get('url')]
                        logger.info(f"Found {len(media_items)} media items via API.")
                        
                        # Add media count to content
                        result['content'] += f"\n\nMedia: {len(media_items)} items detected"
                    else:
                        logger.info("No media items found via API.")
                else:
                    logger.warning("Failed to fetch API data for tweet. Media information might be incomplete.")
                    result['error'] = result.get('error', '') + '; API data fetch failed'

        except Exception as api_e:
            logger.error(f"Error during API fetch or media extraction: {api_e}", exc_info=True)
            result['error'] = result.get('error', '') + f'; API processing error: {api_e}'

        # 5. Download Media if Requested and Available
        # CRITICAL: Debug log to check media_items content
        logger.info(f"Media items to download: {len(media_items)} (Download requested: {download_media})")
        if media_items:
            logger.debug(f"First media item: {media_items[0] if media_items else 'None'}")

        if download_media and media_items:
            logger.info(f"Media download requested for {len(media_items)} items, proceeding with download...")
            try:
                # Create output directory if it doesn't exist
                os.makedirs(media_output_dir, exist_ok=True)
                
                downloader = TwitterMediaDownloader(output_dir=media_output_dir)
                downloaded_files_info = downloader.download_media_items(media_items, page_details, tweet_id)
                
                if downloaded_files_info:
                    result['downloaded_media_paths'] = [f['path'] for f in downloaded_files_info]
                    
                    # Update content with download information
                    media_types = {}
                    for file_info in downloaded_files_info:
                        media_type = file_info.get('type', 'unknown')
                        media_types[media_type] = media_types.get(media_type, 0) + 1
                    
                    download_info = [f"Downloaded {len(downloaded_files_info)} media files:"]
                    for media_type, count in media_types.items():
                        download_info.append(f"- {media_type.capitalize()}: {count}")
                    
                    result['content'] += "\n\n" + "\n".join(download_info)
                    logger.info(f"Successfully downloaded {len(downloaded_files_info)} media files")
                else:
                    logger.warning("No media files were successfully downloaded")
                    result['content'] += "\n\n[Note: Media download was attempted but no files were successfully downloaded]"
                    
            except Exception as dl_e:
                logger.error(f"Error during media download process: {dl_e}", exc_info=True)
                result['error'] = result.get('error', '') + f'; Media download error: {dl_e}'
                result['content'] += "\n\n[Error during media download: Check logs for details]"
        elif download_media and not media_items:
            logger.info("Media download requested, but no media items were found via API.")
            result['content'] += "\n\n[Note: No media files found to download]"

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
                        extraction_method = 'github_playwright_llm_model'
                        logger.info("Processing as GitHub URL...")
                        # Simplified GitHub Extraction: Get Readme/About, Generate Summary
                        # repo_info = {'url': final_url, 'about': '', 'readme_text': '', 'stats': {}}
                        try:
                        # Extract "About"
                            github_result = URLScraper.scrape_github_repo(final_url, browser, vision_model)
                            result.update(github_result)
                        except:
                            logger.error("Error during LLM github summarization")
                            result['error'] = "Error during LLM github summarization"
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
