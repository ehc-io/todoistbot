#!/usr/bin/env python3
import re
import os
import base64
import tempfile
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from litellm import completion

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
    def is_pdf_url(url: str) -> bool:
        """Check if URL points to a PDF file."""
        return url.lower().endswith('.pdf')

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def process_twitter_content(page, url: str) -> Dict[str, Any]:
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
                    model="gpt-4o-mini",
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
    def process_pdf_content(url: str) -> Dict[str, Any]:
        """Process PDF content (placeholder for future implementation)."""
        return {
            'type': 'pdf',
            'url': url,
            'content': 'PDF processing not yet implemented'
        }

    @staticmethod
    def scrape_url(url: str, selectors: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
        """Scrape content from URL with improved error handling and content type detection."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] → Starting to scrape URL: {url}")
        
        # First check URL type
        if not URLScraper.is_valid_url(url):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Invalid URL: {url}")
            return None
            
        if URLScraper.is_pdf_url(url):
            return URLScraper.process_pdf_content(url)
            
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
                
                # Check if it's a Twitter/X URL
                if URLScraper.is_twitter_url(url):
                    result = URLScraper.process_twitter_content(page, url)
                else:
                    # Process regular webpage
                    result = {}
                    try:
                        if selectors:
                            for name, selector in selectors.items():
                                elements = page.query_selector_all(selector)
                                texts = []
                                for el in elements:
                                    text = el.text_content()
                                    if text and text.strip():
                                        texts.append(text.strip())
                                result[name] = texts
                        else:
                            # Get main content
                            main_content = page.evaluate("""() => {
                                const selectors = [
                                    'main article',
                                    'main',
                                    'article',
                                    '[role="main"]',
                                    '.content',
                                    '.main',
                                    '#content',
                                    '#main',
                                    'body'
                                ];
                                
                                for (const selector of selectors) {
                                    const element = document.querySelector(selector);
                                    if (element) {
                                        return element.textContent
                                            .replace(/\\s+/g, ' ')
                                            .trim();
                                    }
                                }
                                return '';
                            }""")
                            
                            if main_content:
                                result = {
                                    'type': 'webpage',
                                    'url': url,
                                    'content': main_content
                                }
                    
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