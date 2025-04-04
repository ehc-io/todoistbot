# twitter_content_extractor.py
import logging
import re
from pathlib import Path
from datetime import datetime
import time
from playwright.sync_api import sync_playwright, Page, Error as PlaywrightError
from typing import Optional, Dict, Any, List
from bs4 import BeautifulSoup # Use BS4 for robust HTML parsing

logger = logging.getLogger(__name__)

class TwitterContentExtractor:
    """Extracts visible tweet content directly from the page using Playwright."""

    @staticmethod
    def _extract_tweet_id_from_url(url: str) -> Optional[str]:
        """Extract tweet ID from a Twitter/X URL."""
        match = re.search(r"(?:twitter\.com|x\.com)/[^/]+/status/(\d+)", url)
        if match:
            return match.group(1)
        logger.warning(f"Could not extract tweet ID from URL: {url}")
        return None

    def _parse_html_with_bs4(self, page: Page, url: str) -> Dict[str, Any]:
        """Uses BeautifulSoup to parse the tweet page HTML for robust extraction."""
        details = {
            'tweet_id': self._extract_tweet_id_from_url(url),
            'text': '',
            'user_name': '',
            'user_handle': '',
            'created_at_iso': '',
            'display_time': '',
            'timestamp_ms': 0,
            'embedded_urls': [],
            'hashtags': [],
            'mentions': [],
            'error': None
        }

        try:
            # First try JavaScript approach similar to original script for better text extraction
            try:
                # This is similar to the original script's approach
                tweet_text = page.evaluate('''() => {
                    const elements = document.querySelectorAll('[data-testid="tweetText"] > span');
                    let text = '';
                    for (const element of elements) {
                        text += element.textContent + ' ';
                    }
                    return text.trim();
                }''')
                
                if tweet_text:
                    details['text'] = tweet_text
                    
                # Also extract URLs using JavaScript as in original
                urls = page.evaluate('''() => {
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
                
                if urls:
                    details['embedded_urls'] = urls
            except Exception as js_e:
                logger.warning(f"JavaScript extraction failed, falling back to BeautifulSoup: {js_e}")
                # Continue to BeautifulSoup approach as fallback
            
            # Ensure the main tweet article exists
            main_tweet_selector = 'article[data-testid="tweet"]'
            page.wait_for_selector(main_tweet_selector, state='visible', timeout=15000)
            html_content = page.content()
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find the main tweet article again in soup
            tweet_article = soup.find('article', attrs={'data-testid': 'tweet'})
            if not tweet_article:
                 details['error'] = "Could not find main tweet article element in HTML."
                 logger.error(details['error'])
                 return details

            # --- User Info ---
            user_name_div = tweet_article.find('div', attrs={'data-testid': 'User-Name'})
            if user_name_div:
                # Name is usually the first span inside
                name_span = user_name_div.find('span', recursive=False) # Avoid nested spans
                if name_span:
                     # Find the actual name text node(s) within the span
                     name_parts = [elem.strip() for elem in name_span.find_all(string=True, recursive=False) if elem.strip()]
                     details['user_name'] = " ".join(name_parts) if name_parts else name_span.get_text(strip=True) # Fallback

                # Handle is usually in a div with dir="ltr" containing @
                handle_div = user_name_div.find('div', {'dir': 'ltr'})
                if handle_div:
                     handle_text = handle_div.get_text(strip=True)
                     if handle_text.startswith('@'):
                          details['user_handle'] = handle_text[1:]

            # --- Timestamp ---
            time_tag = tweet_article.find('time')
            if time_tag and time_tag.has_attr('datetime'):
                details['created_at_iso'] = time_tag['datetime']
                details['display_time'] = time_tag.get_text(strip=True)
                try:
                    # Convert ISO string to Unix timestamp (milliseconds)
                    dt_obj = datetime.fromisoformat(details['created_at_iso'].replace('Z', '+00:00'))
                    details['timestamp_ms'] = int(dt_obj.timestamp() * 1000)
                except ValueError:
                    logger.warning(f"Could not parse timestamp: {details['created_at_iso']}")
                    details['timestamp_ms'] = int(time.time() * 1000) # Fallback to current time

            # --- Tweet Text & Embedded Entities (if JS approach failed) ---
            if not details['text']:
                tweet_text_div = tweet_article.find('div', attrs={'data-testid': 'tweetText'})
                if tweet_text_div:
                    text_parts = []
                    processed_urls = set() # Avoid duplicates from t.co and title attr

                    for element in tweet_text_div.descendants:  # Use descendants instead of children to get nested elements
                        if element.name == 'span' and not element.find_parent('a'):  # Regular text parts not inside links
                             text_parts.append(element.get_text())
                        elif element.name == 'a': # Links (URLs, Hashtags, Mentions)
                            href = element.get('href', '')
                            link_text = element.get_text(strip=True)
                            text_parts.append(link_text) # Include link text in main text flow

                            if href.startswith('/hashtag/'):
                                details['hashtags'].append(link_text.lstrip('#'))
                            elif href.startswith('/') and '@' in link_text: # Basic mention check
                                details['mentions'].append(link_text.lstrip('@'))
                            elif href.startswith('http'):
                                # Handle t.co links and try to get expanded URL from title or text
                                if 't.co' in href:
                                    expanded_url = element.get('title') or element.get_text(strip=True)
                                    if expanded_url and expanded_url.startswith('http') and expanded_url not in processed_urls:
                                         details['embedded_urls'].append(expanded_url)
                                         processed_urls.add(expanded_url)
                                    elif href not in processed_urls: # Fallback to t.co if no expansion found
                                         details['embedded_urls'].append(href)
                                         processed_urls.add(href)
                                elif href not in processed_urls:
                                    details['embedded_urls'].append(href)
                                    processed_urls.add(href)
                        elif element.name == 'img': # Emojis / Alt text
                             alt_text = element.get('alt')
                             if alt_text:
                                  text_parts.append(alt_text)

                    # Join text parts, clean up extra whitespace
                    details['text'] = re.sub(r'\s+', ' ', "".join(text_parts)).strip()

                else:
                     details['error'] = "Could not find tweet text container."
                     logger.error(details['error'])

            # Clean up potential duplicates
            details['hashtags'] = sorted(list(set(details['hashtags'])))
            details['mentions'] = sorted(list(set(details['mentions'])))
            details['embedded_urls'] = sorted(list(set(details['embedded_urls'])))

        except Exception as e:
             details['error'] = f"Error during content extraction: {e}"
             logger.error(details['error'])

        return details

    def extract_tweet_details_from_page(self, url: str, session_path: Path) -> Optional[Dict[str, Any]]:
        """
        Extracts tweet details (text, user, time, urls) directly from the tweet page HTML.
        Uses the provided session for authentication context.
        """
        logger.info(f"Extracting tweet details from page: {url}")
        if not session_path.exists():
            logger.error("Cannot extract page details: Session file missing.")
            return None

        details = None
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    storage_state=str(session_path),
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36' # Consistent UA
                )
                 # Block non-essential resources for faster loading
                context.route("**/*", lambda route: route.abort() if route.request.resource_type in ['font', 'media', 'websocket'] else route.continue_())

                page = context.new_page()
                try:
                    page.goto(url, timeout=30000, wait_until='domcontentloaded')
                    # Give time for dynamic content rendering after DOM load
                    page.wait_for_timeout(5000)

                    # Use BS4 for parsing robustness
                    details = self._parse_html_with_bs4(page, url)

                except PlaywrightError as e:
                    logger.error(f"Playwright error navigating to or processing page {url}: {e}")
                    details = {'error': f"Playwright error: {e}"} # Ensure error is reported
                except Exception as e:
                     logger.error(f"Unexpected error during page extraction for {url}: {e}")
                     details = {'error': f"Unexpected error: {e}"}
                finally:
                    page.close()
                    context.close()
                    browser.close()

        except Exception as e:
            logger.error(f"Error setting up Playwright for page extraction: {e}")
            return {'error': f"Playwright setup error: {e}"}

        if details and not details.get('error'):
             logger.info(f"Successfully extracted page details for tweet {details.get('tweet_id', 'N/A')}.")
        elif details:
             logger.error(f"Failed to extract page details for {url}. Error: {details.get('error')}")
        else:
             logger.error(f"Extraction failed for {url}, details object is None.")

        return details
        """
        Extracts all media URLs (images, GIFs, videos) from the tweet data.
        Follows exactly the same path traversal as the original script.
        """
        logger.info("Extracting media URLs from tweet data")
        
        media_items = []
        
        try:
            # Navigate the nested structure to find the tweet information - EXACT same path as original
            tweet_result = tweet_data['data']['threaded_conversation_with_injections_v2']['instructions'][0]['entries'][0]['content']['itemContent']['tweet_results']['result']
            
            # Extract the extended entities which contain media
            extended_entities = tweet_result.get('legacy', {}).get('extended_entities', {})
            
            if not extended_entities or 'media' not in extended_entities:
                logger.warning("No media found in tweet")
                return []
            
            # Process all media items
            for index, media in enumerate(extended_entities['media']):
                media_type = media.get('type', '')
                media_item = {
                    'type': media_type,
                    'index': index,
                    'url': None,
                    'extension': None
                }
                
                if media_type == 'photo':
                    # For photos, use the highest quality version
                    media_item['url'] = media.get('media_url_https', '')
                    media_item['extension'] = 'jpg'  # Most Twitter images are JPGs
                    
                elif media_type == 'video':
                    # For videos, find the highest quality MP4
                    video_info = media.get('video_info', {})
                    variants = video_info.get('variants', [])
                    
                    # Find the highest quality MP4 variant
                    mp4_variants = [v for v in variants if v.get('content_type') == 'video/mp4']
                    if mp4_variants:
                        best_variant = max(mp4_variants, key=lambda v: v.get('bitrate', 0))
                        media_item['url'] = best_variant['url']
                        media_item['extension'] = 'mp4'
                    
                elif media_type == 'animated_gif':
                    # For GIFs, get the MP4 version (Twitter converts GIFs to MP4)
                    video_info = media.get('video_info', {})
                    variants = video_info.get('variants', [])
                    
                    if variants:
                        # There's usually only one variant for GIFs
                        media_item['url'] = variants[0]['url']
                        media_item['extension'] = 'mp4'  # Twitter serves GIFs as MP4s
                
                # Add to the list if we found a URL
                if media_item['url']:
                    media_items.append(media_item)
                    logger.info(f"Found {media_type} URL: {media_item['url']}")
                else:
                    logger.warning(f"Could not extract URL for {media_type} at index {index}")
            
            return media_items
            
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract media URLs: {e}")
            # Log the structure for debugging
            try:
                if 'data' in tweet_data and 'threaded_conversation_with_injections_v2' in tweet_data['data']:
                    instructions = tweet_data['data']['threaded_conversation_with_injections_v2'].get('instructions', [])
                    if instructions:
                        logger.debug(f"Instructions count: {len(instructions)}")
                        if len(instructions) > 0:
                            entries = instructions[0].get('entries', [])
                            logger.debug(f"Entries count: {len(entries)}")
                            if entries and len(entries) > 0:
                                logger.debug(f"First entry content keys: {entries[0].get('content', {}).keys()}")
                                item_content = entries[0].get('content', {}).get('itemContent', {})
                                if item_content:
                                    logger.debug(f"ItemContent keys: {item_content.keys()}")
                                    tweet_results = item_content.get('tweet_results', {})
                                    if tweet_results:
                                        logger.debug(f"Tweet_results keys: {tweet_results.keys()}")
            except Exception as debug_e:
                logger.error(f"Error while debugging structure: {debug_e}")
            
            return []  # Return empty list on error