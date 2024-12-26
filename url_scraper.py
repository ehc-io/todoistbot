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

class URLScraper:
    @staticmethod
    def clean_url(url: str) -> str:
        """Clean and normalize URL."""
        # Remove trailing parentheses and other common artifacts
        url = re.sub(r'[)\]]$', '', url)
        # Remove hash fragments unless they're meaningful
        if '#' in url and not any(x in url for x in ['#page=', '#section=']):
            url = url.split('#')[0]
        # Normalize Twitter/X URLs
        url = url.replace('x.com', 'twitter.com')
        return url.strip()

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Validate URL and check if it's scrapeable."""
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False
            return True
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
        # Remove URL fragment if present (everything after #)
        base_url = url.split('#')[0]
        return base_url.lower().endswith('.pdf')

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def download_pdf(url: str) -> Optional[bytes]:
        # """Download PDF file from URL."""
        # print(f"my_url: {url}")
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
        print(summary)
        return {
            'type': 'pdf',
            'url': url,
            'summary': summary
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
                'content': content_for_summary,
                'summary': summary,
                'stats': stats
            }
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error processing GitHub content: {str(e)}")
            return None

    @staticmethod
    def process_twitter_content(page, url: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """Process Twitter/X content by taking screenshot and using vision model."""
        try:
            # Wait for tweet to load
            page.wait_for_selector('article[data-testid="tweet"]', timeout=10000)
            
            # Create temporary file for screenshot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                # Take screenshot of the tweet
                tweet_element = page.query_selector('article[data-testid="tweet"]')
                tweet_element.screenshot(path=tmp.name)
                
                # Encode image to base64
                base64_image = URLScraper.encode_image(tmp.name)
                
                # Remove temporary file
                os.unlink(tmp.name)
                
                # Use vision model to describe tweet
                response = completion(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Please describe this tweet in detail, including the author, content, images if any, and engagement metrics if visible."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                                }
                            ]
                        }
                    ]
                )
                
                return {
                    'type': 'twitter',
                    'url': url,
                    'content': response.choices[0].message.content if response.choices else None
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
                            
                            if text_content:
                                result = {
                                    'type': 'webpage',
                                    'url': url,
                                    'content': text_content
                                }
                                
                                # If content is longer than 2000 characters, generate summary
                                if len(text_content) > 2000:
                                    summary = URLScraper.get_webpage_summary(text_content, model=model)
                                    result['summary'] = summary
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
