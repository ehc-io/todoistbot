import logging
import re
from datetime import datetime
import time
from playwright.sync_api import sync_playwright, Page
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class TweetExtractor:
    """Extract tweet content and metadata using Playwright."""
    
    @staticmethod
    def extract_tweet_id_from_url(url: str) -> Optional[str]:
        """Extract tweet ID from a Twitter/X URL."""
        match = re.search(r"(?:twitter\.com|x\.com)/[^/]+/status/(\d+)", url)
        if match:
            return match.group(1)
        logger.warning(f"Could not extract tweet ID from URL: {url}")
        return None
    
    def extract_tweet(self, tweet_url: str, session_path: Optional[Path] = None, auth_token: str = None, 
                      csrf_token: str = None) -> Dict[str, Any]:
        """
        Extract tweet information using Playwright.
        
        Args:
            tweet_url: URL of the tweet to extract
            session_path: Optional path to a stored browser session
            auth_token: Optional X/Twitter auth_token for authentication
            csrf_token: Optional X/Twitter csrf token for authentication
            
        Returns:
            Dictionary containing the formatted tweet data
        """
        logger.info(f"Extracting tweet content from URL: {tweet_url}")
        
        tweet_data = {
            'tweet_id': self.extract_tweet_id_from_url(tweet_url),
            'user_name': '',
            'user_handle': '',
            'created_at': '',
            'timestamp': 0,
            'text': '',
            'urls': [],
            'error': None
        }
        
        with sync_playwright() as playwright:
            # Launch browser
            browser = playwright.chromium.launch(headless=True)
            
            # Set up context with authentication
            if session_path and session_path.exists():
                context = browser.new_context(storage_state=str(session_path))
            else:
                context = browser.new_context()
                
                # Add authentication cookies if provided
                if auth_token and csrf_token:
                    context.add_cookies([
                        {
                            "name": "auth_token",
                            "value": auth_token,
                            "domain": ".x.com",
                            "path": "/"
                        },
                        {
                            "name": "ct0",
                            "value": csrf_token,
                            "domain": ".x.com",
                            "path": "/"
                        }
                    ])
            
            # Create a new page
            page = context.new_page()
            
            try:
                # Navigate to the tweet URL with timeout
                logger.info(f"Loading tweet page: {tweet_url}")
                page.goto(tweet_url, timeout=30000)
                page.wait_for_timeout(5000)  # Wait for dynamic content to load
                
                # Wait for tweet content to be visible
                tweet_selector = 'article[data-testid="tweet"]'
                page.wait_for_selector(tweet_selector, state='visible', timeout=15000)
                
                # Extract all tweet data using JavaScript for better reliability
                tweet_data = self._extract_tweet_data(page, tweet_data)
                
                # Format the output
                formatted_output = self._format_tweet_output(tweet_data)
                tweet_data['formatted_output'] = formatted_output
                
            except Exception as e:
                error_msg = f"Error extracting tweet content: {str(e)}"
                logger.error(error_msg)
                tweet_data['error'] = error_msg
            finally:
                # Close browser resources
                page.close()
                context.close()
                browser.close()
        
        return tweet_data
    
    def _extract_tweet_data(self, page: Page, tweet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all tweet data from the page using JavaScript evaluation."""
        try:
            # Extract tweet text
            tweet_data['text'] = page.evaluate('''() => {
                const elements = document.querySelectorAll('[data-testid="tweetText"] > span');
                let text = '';
                for (const element of elements) {
                    text += element.textContent + ' ';
                }
                return text.trim();
            }''')
            
            # Extract URLs
            tweet_data['urls'] = page.evaluate('''() => {
                const links = document.querySelectorAll('[data-testid="tweetText"] a[href]');
                const urls = [];
                for (const link of links) {
                    const href = link.getAttribute('href');
                    if (href && !href.startsWith('/hashtag/') && !href.startsWith('/@')) {
                        if (href.includes('t.co')) {
                            const expandedUrl = link.getAttribute('title') || href;
                            urls.push(expandedUrl);
                        } else {
                            urls.push(href);
                        }
                    }
                }
                return urls;
            }''')
            
            # Extract user information
            user_info = page.evaluate('''() => {
                const userElement = document.querySelector('[data-testid="User-Name"]');
                if (!userElement) return { handle: '', name: '' };
                
                const nameElement = userElement.querySelector('span:first-child');
                const handleElement = userElement.querySelector('div[dir="ltr"]');
                
                return {
                    name: nameElement ? nameElement.textContent.trim() : '',
                    handle: handleElement ? handleElement.textContent.trim().replace('@', '') : ''
                };
            }''')
            
            tweet_data['user_name'] = user_info.get('name', '')
            tweet_data['user_handle'] = user_info.get('handle', '')
            
            # Extract timestamp
            timestamp_info = page.evaluate('''() => {
                const timeElement = document.querySelector('time');
                if (!timeElement) return { created_at: '', timestamp: 0 };
                
                const datetime = timeElement.getAttribute('datetime');
                const displayTime = timeElement.textContent.trim();
                
                return {
                    created_at: datetime,
                    display_time: displayTime
                };
            }''')
            
            # Format timestamp data
            created_at = timestamp_info.get('created_at', '')
            
            if created_at:
                tweet_data['created_at'] = created_at
                try:
                    date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    tweet_data['timestamp'] = int(date_obj.timestamp())
                except ValueError:
                    logger.warning(f"Could not parse date: {created_at}")
                    tweet_data['timestamp'] = int(time.time())  # Use current time as fallback
            else:
                tweet_data['timestamp'] = int(time.time())
            
            logger.info(f"Successfully extracted tweet content: {tweet_data['text'][:10]}...")
            
        except Exception as e:
            logger.error(f"Error during data extraction: {e}")
            tweet_data['error'] = f"Extraction error: {str(e)}"
            
        return tweet_data
    
    def _format_tweet_output(self, tweet_data: Dict[str, Any]) -> str:
        """Format the extracted tweet data into the requested output format."""
        # Format date string from ISO to readable format
        timestamp_str = "Unknown"
        if tweet_data['created_at']:
            try:
                dt = datetime.fromisoformat(tweet_data['created_at'].replace('Z', '+00:00'))
                timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
            except ValueError:
                pass
        
        # Build the formatted output
        output_parts = [
            "TWEET DOWNLOAD REPORT",
            "=" * 56,
            f"User: @{tweet_data['user_handle']} ({tweet_data['user_name']})",
            f"Tweet ID: {tweet_data['tweet_id']}",
            f"Posted on: {timestamp_str}",
            "-" * 80,
            "CONTENT: ",
            tweet_data['text'],
            "-" * 80
        ]
        
        # Add URLs section if any are found
        if tweet_data['urls'] and len(tweet_data['urls']) > 0:
            output_parts.append("URLs in tweet:")
            for url in tweet_data['urls']:
                output_parts.append(f"  - {url}")
            output_parts.append("-" * 80)
        
        # Join all parts into the final formatted output
        return "\n".join(output_parts)
    
    def print_tweet(self, tweet_data: Dict[str, Any]) -> None:
        """Print the formatted tweet data to console."""
        if tweet_data.get('error'):
            print(f"Error extracting tweet: {tweet_data['error']}")
            return
            
        print(tweet_data.get('formatted_output', 'No formatted output available'))
        

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set up command-line argument parsing
    import argparse
    parser = argparse.ArgumentParser(description='Extract content from a Twitter/X tweet.')
    parser.add_argument('--url', type=str, required=True,
                        help='URL of the tweet to extract (default: example URL)')
    parser.add_argument('--session-path', type=str, default="session-data/session.json",
                        help='Path to a saved Twitter session file (optional)')
    args = parser.parse_args()
    tweet_url = args.url
    
    # Create extractor
    extractor = TweetExtractor()
    
    # Extract tweet based on provided authentication options
    if args.session_path:
        # Option 2: Using a saved session file
        from pathlib import Path
        session_path = Path(args.session_path)
        tweet_data = extractor.extract_tweet(tweet_url, session_path=session_path)
    else:
        sys.exit("Error: Please provide a session path using the --session-path option.")
    
    # Print the tweet
    extractor.print_tweet(tweet_data)
    

if __name__ == "__main__":
    main()