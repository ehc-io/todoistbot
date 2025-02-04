#!/usr/bin/env python3
import re
import os
import base64
import tempfile
import requests
import pypandoc
from typing import Optional, List, Dict, Any
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from litellm import completion
import PyPDF2
from io import BytesIO
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

class TwitterProcessor:
    """Helper class for processing Twitter content."""
    
    @staticmethod
    def extract_urls_from_html(html_content: str) -> List[str]:
        """Extract all external URLs from tweet HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        urls = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if not href.startswith('/') and not href.startswith('https://twitter.com'):
                urls.append(href)
        
        return list(set(urls))

    @staticmethod
    def extract_tweet_text(html_content: str) -> str:
        """Extract clean text content from tweet HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        tweet_text_div = soup.find('div', {'data-testid': 'tweetText'})
        
        if not tweet_text_div:
            return ""
            
        text_parts = []
        for element in tweet_text_div.descendants:
            if element.name == 'a':
                if element.get('href', '').startswith('/'):
                    text_parts.append(element.text)
            elif isinstance(element, str):
                text_parts.append(element.strip())
                
        return ' '.join(filter(None, text_parts))

    @staticmethod
    def extract_metadata(html_content: str) -> Dict[str, Any]:
        """Extract additional metadata from tweet HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        metadata = {}
        
        timestamp_elem = soup.find('time')
        if timestamp_elem and timestamp_elem.get('datetime'):
            metadata['timestamp'] = timestamp_elem['datetime']
        
        author_elem = soup.find('div', {'data-testid': 'User-Name'})
        if author_elem:
            metadata['author'] = {
                'name': author_elem.text,
                'handle': author_elem.find('span', text=re.compile(r'^@')).text if author_elem.find('span', text=re.compile(r'^@')) else None
            }
        
        return metadata

class YouTubeProcessor:
    """Helper class for processing YouTube content using the YouTube Data API."""
    
    def __init__(self, api_key: str):
        """Initialize YouTube API client.
        
        Args:
            api_key (str): YouTube Data API key
        """
        self.youtube = build('youtube', 'v3', developerKey=api_key)
    
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        # Handle youtu.be URLs
        if 'youtu.be' in url:
            return url.split('/')[-1].split('?')[0]
        
        # Handle youtube.com URLs
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
        try:
            request = self.youtube.videos().list(
                part="snippet,statistics",
                id=video_id
            )
            response = request.execute()
            
            if not response['items']:
                return None
            
            video = response['items'][0]
            snippet = video['snippet']
            statistics = video['statistics']
            
            return {
                'type': 'youtube_video',
                'video_id': video_id,
                'url': f'https://youtube.com/watch?v={video_id}',
                'content': {
                    'title': snippet['title'],
                    'description': snippet['description'],
                    'published_at': snippet['publishedAt'],
                    'channel_title': snippet['channelTitle'],
                    'view_count': statistics.get('viewCount', 'N/A'),
                    'like_count': statistics.get('likeCount', 'N/A')
                }
            }
            
        except Exception as e:
            print(f"Error fetching video details: {str(e)}")
            return None
    
    def get_playlist_details(self, playlist_id: str, max_items: int = 50) -> Optional[Dict[str, Any]]:
        """Fetch playlist details and items using YouTube API."""
        try:
            # First get playlist details
            playlist_request = self.youtube.playlists().list(
                part="snippet",
                id=playlist_id
            )
            playlist_response = playlist_request.execute()
            
            if not playlist_response['items']:
                return None
            
            playlist = playlist_response['items'][0]
            snippet = playlist['snippet']
            
            # Then get playlist items
            items_request = self.youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=max_items
            )
            items_response = items_request.execute()
            
            videos = []
            for item in items_response.get('items', []):
                video_snippet = item['snippet']
                videos.append({
                    'title': video_snippet['title'],
                    'video_id': video_snippet['resourceId']['videoId'],
                    'position': video_snippet['position'] + 1
                })
            
            return {
                'type': 'youtube_playlist',
                'playlist_id': playlist_id,
                'url': f'https://youtube.com/playlist?list={playlist_id}',
                'content': {
                    'title': snippet['title'],
                    'description': snippet['description'],
                    'channel_title': snippet['channelTitle'],
                    'published_at': snippet['publishedAt'],
                    'video_count': len(videos),
                    'videos': videos
                }
            }
            
        except Exception as e:
            print(f"Error fetching playlist details: {str(e)}")
            return None 
    
class URLScraper:
    @staticmethod
    def clean_url(url: str) -> str:
        """Clean and normalize URL."""
        url = re.sub(r'[)\]]$', '', url)
        if '#' in url and not any(x in url for x in ['#page=', '#section=']):
            url = url.split('#')[0]
        url = url.replace('x.com', 'twitter.com')
        return url.strip()

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Validate URL and check if it's scrapeable."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract and clean URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        return [URLScraper.clean_url(url) for url in urls]

    @staticmethod
    def is_twitter_url(url: str) -> bool:
        """Check if URL is a Twitter/X post."""
        return bool(re.match(r'https?://(twitter\.com|x\.com)/\w+/status/\d+', url))

    @staticmethod
    def is_github_repo_url(url: str) -> bool:
        """Check if URL is a GitHub repository or sub-path."""
        pattern = r'https?://github\.com/[^/]+/[^/]+'
        return bool(re.match(pattern, url))

    @staticmethod
    def is_pdf_url(url: str, check_headers: bool = True) -> bool:
        # First check patterns
        url_lower = url.lower()
        
        # Quick pattern check
        pdf_patterns = [
            lambda u: u.split('#')[0].split('?')[0].endswith('.pdf'),
            lambda u: 'arxiv.org/pdf/' in u,
            # ... other patterns ...
        ]
        
        if any(pattern(url_lower) for pattern in pdf_patterns):
            return True
        
        # If pattern check fails and headers check is enabled, try HEAD request
        if check_headers:
            try:
                response = requests.head(url, allow_redirects=True, timeout=5)
                content_type = response.headers.get('Content-Type', '').lower()
                return 'application/pdf' in content_type
            except Exception:
                # If request fails, fall back to pattern matching result
                return False
                
        return False

    @staticmethod
    def is_youtube_url(url: str) -> bool:
        """Check if URL is a YouTube video or playlist."""
        youtube_patterns = [
            # Video patterns (including mobile)
            r'https?://(?:(?:www|m)\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://youtu\.be/[\w-]+',
            
            # Playlist patterns (including mobile)
            r'https?://(?:(?:www|m)\.)?youtube\.com/playlist\?(?:[\w=&-]+&)?list=[\w-]+(?:&[\w=&-]+)?'
        ]
        return any(re.match(pattern, url) for pattern in youtube_patterns)

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def download_pdf(url: str) -> Optional[bytes]:
        # print('downloading PDF')
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            # print(response.content)
            return response.content
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error downloading PDF: {str(e)}")
            return None

    @staticmethod
    def extract_pdf_text(pdf_content: bytes) -> str:
        """Extract text content from PDF."""
        # print('trying to extract pdf text')
        try:
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = []
            
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())
            
            return "\n".join(text_content)
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error extracting PDF text: {str(e)}")
            return ""

    @staticmethod
    def get_webpage_summary(text: str, model: str = "gpt-4o-mini") -> str:
        """Get summary of webpage content using LLM."""
        try:
            # Truncate text if too long (adjust limit based on model's context window)
            max_chars = 14000  # Adjust based on model's limits
            if len(text) > max_chars:
                text = text[:max_chars] + "..."

            prompt = f"""Please provide a 100 word summary of this webpage content. 
            Focus on the main points and key information.
            
            Content:
            {text}"""
            
            messages = [{ "content": prompt, "role": "user"}]
            
            response = completion(model=model, messages=messages)
            return response.choices[0].message.content if response.choices else "Summary generation failed."
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error generating webpage summary: {str(e)}")
            return "Error generating summary"

    @staticmethod
    def get_pdf_summary(text: str, model: str = "gpt-4o-mini") -> str:
        """Get summary of PDF content using LLM."""
        try:
            # Truncate text if too long (adjust limit based on model's context window)
            max_chars = 14000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."

            prompt = f"""Please provide a 100 word summary of the content. 
            
            Content:
            {text}"""
            
            messages = [{ "content": prompt, "role": "user"}]
            
            response = completion(model=model, messages=messages)
            return response.choices[0].message.content if response.choices else "Summary generation failed."
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error generating summary: {str(e)}")
            return "Error generating summary"

    @staticmethod
    def process_pdf_content(url: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """Process PDF content by extracting text and generating summary."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing PDF from URL: {url}")
        
        # Download PDF
        pdf_content = URLScraper.download_pdf(url)
        if not pdf_content:
            return {
                'type': 'pdf',
                'url': url,
                'error': 'Failed to download PDF'
            }

        # Extract text
        text_content = URLScraper.extract_pdf_text(pdf_content)
        if not text_content:
            return {
                'type': 'pdf',
                'url': url,
                'error': 'Failed to extract text from PDF'
            }

        # Generate summary using specified model
        summary = URLScraper.get_pdf_summary(text_content, model=model)
        # print(summary)
        return {
            'type': 'pdf',
            'url': url,
            'content': summary
        }

    @staticmethod
    def process_github_content(page, url: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """Process GitHub repository content (exact path) and generate summary."""
        try:
            # Navigate directly to the GitHub URL (no root repo fallback)
            page.goto(url, wait_until='domcontentloaded', timeout=30000)
            
            # Wait for repository content to load (10-second timeout)
            page.wait_for_selector('div#repo-content-pjax-container', timeout=10000)
            
            # Extract "About" text if available
            about_section = page.query_selector('.Layout-sidebar .f4')
            about_text = about_section.text_content().strip() if about_section else ""
            
            # Extract README or page content if available
            readme = page.query_selector('article.markdown-body')
            readme_text = readme.text_content().strip() if readme else ""
            
            # Attempt to gather basic repository stats if visible
            stars_el = page.query_selector('#repo-stars-counter-star')
            forks_el = page.query_selector('#repo-network-counter')
            watchers_el = page.query_selector('#repo-watchers-counter')
            stats = {
                'stars': stars_el.text_content().strip() if stars_el else "N/A",
                'forks': forks_el.text_content().strip() if forks_el else "N/A",
                'watchers': watchers_el.text_content().strip() if watchers_el else "N/A",
            }
            
            # Look for any additional content on this specific path
            specific_content = page.query_selector('article.markdown-body, .Box-body')
            if specific_content:
                specific_text = specific_content.text_content().strip()
                # Combine "specific" content with the existing readme text
                readme_text = f"Specific path content:\n{specific_text}\n\nRepository overview:\n{readme_text}"
            
            # Combine content for summarization
            content_for_summary = f"""
            Repository URL: {url}
            About: {about_text}
            
            Content: {readme_text}
            
            Repository Statistics:
            - Stars: {stats['stars']}
            - Forks: {stats['forks']}
            - Watchers: {stats['watchers']}
            """
            
            # Generate summary using LLM
            prompt = f"""Please provide a 100 word summary of this GitHub project.
            Disregard whatever is related to the Github service itself. Consider only information related to the particular project this page points to.
            
            Repository Content:
            {content_for_summary}"""
            
            messages = [{"content": prompt, "role": "user"}]
            response = completion(model=model, messages=messages)
            summary = response.choices[0].message.content if response.choices else "Summary generation failed."
            
            return {
                'type': 'github',
                'url': url,
                'content': summary,
                'stats': stats
            }
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error processing GitHub content: {str(e)}")
            return None

    @staticmethod
    def process_twitter_content(page, url: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """Process Twitter/X content using both screenshot analysis and HTML parsing."""
        try:
            # Wait for tweet to load
            tweet_selector = 'article[data-testid="tweet"]'
            page.wait_for_selector(tweet_selector, timeout=10000)
            
            # Get the HTML content first
            tweet_element = page.query_selector(tweet_selector)
            html_content = tweet_element.inner_html()
            
            # Extract text, URLs, and metadata from HTML
            tweet_text = TwitterProcessor.extract_tweet_text(html_content)
            urls = TwitterProcessor.extract_urls_from_html(html_content)
            metadata = TwitterProcessor.extract_metadata(html_content)
            
            # Take screenshot for vision model analysis
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tweet_element.screenshot(path=tmp.name)
                base64_image = URLScraper.encode_image(tmp.name)
                os.unlink(tmp.name)
                
                # Use vision model for comprehensive analysis
                response = completion(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Please analyze this tweet and provide: 1) Main topic/subject, 2) Any media content description, 3) Notable engagement metrics if visible, 4) Any hashtags or key mentions"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                                }
                            ]
                        }
                    ]
                )
                
                vision_analysis = response.choices[0].message.content if response.choices else None
                
                # Combine all information into a comprehensive summary
                content_parts = []
                
                # Add metadata if available
                if metadata.get('author'):
                    author_info = metadata['author']
                    content_parts.append(f"Author: {author_info.get('name')} ({author_info.get('handle')})")
                
                if metadata.get('timestamp'):
                    content_parts.append(f"Posted: {metadata['timestamp']}")
                
                # Add the original tweet text
                if tweet_text:
                    content_parts.append(f"\nTweet text:\n{tweet_text}")
                
                # Add vision model analysis
                if vision_analysis:
                    content_parts.append(f"\nAnalysis:\n{vision_analysis}")
                
                # Add external links
                if urls:
                    content_parts.append("\nExternal links:")
                    for url in urls:
                        content_parts.append(f"- {url}")
                
                # Combine all parts into a single content string
                combined_content = "\n".join(content_parts)
                
                return {
                    'type': 'twitter',
                    'url': url,
                    'content': combined_content
                }
                    
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error processing Twitter content: {str(e)}")
            return None

    @staticmethod
    def process_youtube_content(url: str, youtube_processor: YouTubeProcessor) -> Optional[Dict[str, Any]]:
        """Process YouTube video or playlist content."""
        # Check for video ID first
        video_id = youtube_processor.extract_video_id(url)
        if video_id:
            video_details = youtube_processor.get_video_details(video_id)
            if video_details:
                return {
                    'type': 'youtube',
                    'url': url,
                    'content': f"{video_details['content']['title']} | {video_details['content']['description']}"
                }
        
        # Check for playlist ID
        playlist_id = youtube_processor.extract_playlist_id(url)
        if playlist_id:
            playlist_details = youtube_processor.get_playlist_details(playlist_id)
            if playlist_details:
                return {
                    'type': 'youtube',
                    'url': url,
                    'content': f"{playlist_details['content']['title']} | {playlist_details['content']['description']}"
                }
        
        return None
    
    @staticmethod
    def extract_content_with_pandoc(html: str) -> str:
        """Extract text content from HTML using Pandoc."""
        try:
            # Create a temporary file for the HTML content
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', encoding='utf-8', delete=False) as tmp:
                tmp.write(html)
                tmp_path = tmp.name

            try:
                # Convert HTML to plain text using Pandoc
                text = pypandoc.convert_file(
                    tmp_path,
                    'plain',
                    format='html',
                    extra_args=['--wrap=none', '--strip-comments']
                )
                
                # Clean up the text
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error converting HTML with Pandoc: {str(e)}")
            return ""

    @staticmethod
    def _scrape_url(url: str, model: str = "gpt-4o-mini", selectors: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
        """Extended scrape_url method with YouTube support."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] → Starting to scrape URL: {url}")
        
        if not URLScraper.is_valid_url(url):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Invalid URL: {url}")
            return None
            
        # Check if it's a YouTube URL
        if URLScraper.is_youtube_url(url):
            youtube_api_key = os.getenv('YOUTUBE_API_KEY')
            if not youtube_api_key:
                print("YOUTUBE_API_KEY environment variable is not set")
                return None
                
            youtube_processor = YouTubeProcessor(youtube_api_key)
            return URLScraper.process_youtube_content(url, youtube_processor)

        # If it's a PDF, remove any fragment before processing
        if URLScraper.is_pdf_url(url):
            # Strip off anything after '#'
            url = re.sub(r'#.*$', '', url)
            # print(f"my_url_1: {url}")
            return URLScraper.process_pdf_content(url, model=model)
            
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={'width': 1280, 'height': 800},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
                page = context.new_page()
                
                try:
                    page.goto(url, wait_until='domcontentloaded', timeout=30000)
                except PlaywrightTimeout:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Timeout loading page: {url}")
                    return None
                
                # Check URL type and process accordingly
                if URLScraper.is_twitter_url(url):
                    result = URLScraper.process_twitter_content(page, url, model=model)
                elif URLScraper.is_github_repo_url(url):
                    result = URLScraper.process_github_content(page, url, model=model)
                else:
                    # Process regular webpage using Pandoc
                    try:
                        if selectors:
                            result = {}
                            for name, selector in selectors.items():
                                elements = page.query_selector_all(selector)
                                texts = []
                                for el in elements:
                                    html = el.evaluate('el => el.outerHTML')
                                    if html:
                                        text = URLScraper.extract_content_with_pandoc(html)
                                        if text and text.strip():
                                            texts.append(text.strip())
                                result[name] = texts
                        else:
                            html = page.content()
                            text_content = URLScraper.extract_content_with_pandoc(html)
                            summary = URLScraper.get_webpage_summary(text_content, model=model)
                            
                            if text_content:
                                result = {
                                    'type': 'webpage',
                                    'url': url,
                                    'content': summary
                                }
                            else:
                                result = None
                    
                    except Exception as e:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error extracting content: {str(e)}")
                        return None
                
                browser.close()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Successfully scraped URL: {url}")
                return result
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error scraping {url}:")
            print(f"├── Error type: {type(e).__name__}")
            print(f"└── Details: {str(e)}")
            return None

    @staticmethod
    def get_url_info_from_search(url: str) -> Optional[str]:
        """Get information about a URL using Google Custom Search API."""
        try:
            # Get API credentials from environment
            api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
            cx = os.getenv('SEARCH_ENGINE_ID')  # Custom Search Engine ID
            
            if not api_key or not cx:
                print("[{datetime.now().strftime('%H:%M:%S')}] Google API credentials not configured")
                return None
            
            # Initialize the Custom Search API service
            service = build("customsearch", "v1", developerKey=api_key)
            
            # Extract domain for better search results
            domain = urlparse(url).netloc
            
            # Perform the search
            query = f"site:{domain} {url}"  # Search for content from this specific URL and domain
            result = service.cse().list(q=query, cx=cx, num=5).execute()
            
            if 'items' not in result:
                return None
                
            # Collect relevant information from search results
            search_info = []
            
            # Try to get description of the exact URL
            exact_match = next((item for item in result['items'] 
                              if item['link'].startswith(url)), None)
            
            if exact_match and 'snippet' in exact_match:
                search_info.append(f"Page description: {exact_match['snippet']}")
            
            # Get site description if available
            site_info = next((item for item in result['items'] 
                            if item['link'].startswith(f"https://{domain}")), None)
                            
            if site_info and 'snippet' in site_info:
                search_info.append(f"Site information: {site_info['snippet']}")
            
            # Combine the information
            if search_info:
                return " | ".join(search_info)
            return None
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error getting search results: {str(e)}")
            return None

    @staticmethod
    def scrape_url(url: str, model: str = "gpt-4o-mini", selectors: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
        """Extended scrape_url method with screenshot fallback."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] → Starting to scrape URL: {url}")
        
        if not URLScraper.is_valid_url(url):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Invalid URL: {url}")
            return None
            
        # Check for special URL types (YouTube, PDF, etc.)
        if URLScraper.is_youtube_url(url):
            youtube_api_key = os.getenv('YOUTUBE_API_KEY')
            if not youtube_api_key:
                print("YOUTUBE_API_KEY environment variable is not set")
                return None
            youtube_processor = YouTubeProcessor(youtube_api_key)
            return URLScraper.process_youtube_content(url, youtube_processor)

        if URLScraper.is_pdf_url(url):
            url = re.sub(r'#.*$', '', url)
            return URLScraper.process_pdf_content(url, model=model)
            
        try:
            with sync_playwright() as p:
                # Launch browser with specific arguments
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        '--disable-gpu',
                        '--disable-dev-shm-usage',
                        '--disable-setuid-sandbox',
                        '--no-sandbox',
                    ]
                )
                
                # Configure context with resource handling
                context = browser.new_context(
                    viewport={'width': 1280, 'height': 800},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    # Reduce memory usage
                    device_scale_factor=1,
                    # Disable features that might slow down loading
                    is_mobile=False,
                    has_touch=False
                )
                page = context.new_page()
                
                # Set up request interception to block non-essential resources
                page.route("**/*", lambda route: route.abort() 
                    if route.request.resource_type in ['image', 'stylesheet', 'font', 'media'] 
                    else route.continue_())
                
                # Configure page for optimal loading
                page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})
                
                # Enable JavaScript error handling
                page.on("pageerror", lambda err: print(f"Page error: {err}"))
                page.on("console", lambda msg: print(f"Console {msg.type}: {msg.text}"))
                
                try:
                    page.goto(url, wait_until='domcontentloaded', timeout=20000)
                except PlaywrightTimeout:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Page failed to load: {url}")
                    print("Attempting to get information from search...")
                    
                    # Try to get information about the URL from search
                    search_info = URLScraper.get_url_info_from_search(url)
                    if search_info:
                        result = {
                            'type': 'webpage',
                            'url': url,
                            'content': search_info,
                            'extraction_method': 'web_search'
                        }
                        return result
                    return None
                except PlaywrightTimeout:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Timeout loading page: {url}")
                    return None
                
                # Process based on URL type
                if URLScraper.is_twitter_url(url):
                    result = URLScraper.process_twitter_content(page, url, model=model)
                elif URLScraper.is_github_repo_url(url):
                    result = URLScraper.process_github_content(page, url, model=model)
                else:
                    # Try Pandoc extraction first
                    try:
                        if selectors:
                            result = {}
                            for name, selector in selectors.items():
                                elements = page.query_selector_all(selector)
                                texts = []
                                for el in elements:
                                    html = el.evaluate('el => el.outerHTML')
                                    if html:
                                        text = URLScraper.extract_content_with_pandoc(html)
                                        if text and text.strip():
                                            texts.append(text.strip())
                                result[name] = texts
                        else:
                            html = page.content()
                            text_content = URLScraper.extract_content_with_pandoc(html)
                            
                            if not text_content or len(text_content.strip()) < 100:  # Check if content extraction failed or returned minimal content
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] Pandoc extraction failed or returned minimal content. Falling back to web search.")
                                
                                # Fallback to web search
                                search_info = URLScraper.get_url_info_from_search(url)
                                if search_info:
                                    result = {
                                        'type': 'webpage',
                                        'url': url,
                                        'content': search_info,
                                        'extraction_method': 'web_search'
                                    }
                                else:
                                    result = None
                            else:
                                summary = URLScraper.get_webpage_summary(text_content, model=model)
                                result = {
                                    'type': 'webpage',
                                    'url': url,
                                    'content': summary,
                                    'extraction_method': 'pandoc'
                                }
                    
                    except Exception as e:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error with Pandoc extraction, falling back to screenshot: {str(e)}")
                        # Fallback to web search
                        search_info = URLScraper.get_url_info_from_search(url)
                        if search_info:
                            result = {
                                'type': 'webpage',
                                'url': url,
                                'content': search_info,
                                'extraction_method': 'web_search'
                            }
                        else:
                            result = None
                
                browser.close()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Successfully scraped URL: {url}")
                return result
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error scraping {url}:")
            print(f"├── Error type: {type(e).__name__}")
            print(f"└── Details: {str(e)}")
            return None