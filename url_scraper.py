#!/usr/bin/env python3
import re
import os
import base64
import tempfile
import requests
import pypandoc
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from litellm import completion
import PyPDF2
from io import BytesIO
from bs4 import BeautifulSoup

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
    def is_pdf_url(url: str) -> bool:
        base_url = url.split('#')[0]
        return base_url.lower().endswith('.pdf')

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def download_pdf(url: str) -> Optional[bytes]:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error downloading PDF: {str(e)}")
            return None

    @staticmethod
    def extract_pdf_text(pdf_content: bytes) -> str:
        """Extract text content from PDF."""
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
    def scrape_url(url: str, model: str = "gpt-4o-mini", selectors: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
        """Scrape content from URL with improved error handling and content type detection."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] → Starting to scrape URL: {url}")
        
        # First check URL type
        if not URLScraper.is_valid_url(url):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Invalid URL: {url}")
            return None

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
