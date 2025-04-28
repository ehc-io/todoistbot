import requests
import json
import logging
from pathlib import Path
from playwright.sync_api import sync_playwright, Error as PlaywrightError, Route, Request
from typing import Optional, Dict, Any, Tuple, List
import time
# ---> FIX: Add urlparse needed for extract_media_urls_from_api_data <---
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class TwitterAPIClient:
    """Handles authenticated API calls to Twitter/X."""

    def __init__(self, session_path: Path):
        self.session_path = session_path
        self.auth_tokens: Optional[Tuple[str, str, str]] = None # (auth_token, csrf_token, bearer_token)

    def _extract_auth_tokens(self) -> Optional[Tuple[str, str, str]]:
        """
        Uses Playwright to load the session and extract necessary authentication tokens.
        Closely matches the approach from the original extractor script.
        Returns (auth_token, csrf_token, bearer_token) or None if extraction fails.
        """
        if not self.session_path.exists():
            logger.error("Cannot extract tokens: Session file does not exist.")
            return None

        logger.info("Extracting auth tokens from session using Playwright...")
        captured_bearer_token = None
        auth_token = None
        csrf_token = None
        bearer_token = None
        last_token = None  # Track last seen token to avoid duplicates

        try:
            with sync_playwright() as playwright:
                # Launch browser with the session state
                browser = playwright.chromium.launch(headless=True)
                context = browser.new_context(storage_state=str(self.session_path))
                page = context.new_page()
                
                # Define request handler to capture bearer token
                def handle_request(request):
                    nonlocal captured_bearer_token, last_token
                    
                    # Look for API requests to Twitter/X endpoints
                    if ('api.twitter.com' in request.url or 
                        'twitter.com/i/api' in request.url or 
                        'x.com/i/api' in request.url):
                        headers = request.headers
                        auth_header = headers.get('authorization')
                        
                        # Check if this is a Bearer token and not the same as the last one we logged
                        if auth_header and auth_header.startswith('Bearer '):
                            token = auth_header.replace('Bearer ', '')
                            # Only capture if it's a new token
                            if token != last_token:
                                captured_bearer_token = token
                                last_token = token
                                logger.info(f"Intercepted Bearer token: {token[:20]}...")
                
                # Listen to all requests
                page.on('request', handle_request)

                # Navigate to Twitter/X home page
                logger.info("Navigating to Twitter/X home page")
                page.goto("https://x.com/home")
                
                # Wait for more API calls to happen
                logger.info("Waiting for API calls...")
                page.wait_for_timeout(5000)  # Wait for 5 seconds

                # Extract cookies
                cookies = context.cookies()
                
                # Find auth_token and csrf_token from cookies
                auth_token = next((cookie["value"] for cookie in cookies if cookie["name"] == "auth_token"), None)
                csrf_token = next((cookie["value"] for cookie in cookies if cookie["name"] == "ct0"), None)
                
                # If we didn't capture a bearer token through request interception, try JS context
                if not captured_bearer_token:
                    logger.info("No bearer token captured from requests, trying JavaScript context...")
                    
                    # Try to extract bearer token from JavaScript context
                    try:
                        js_bearer_token = page.evaluate('''() => {
                            // Look in various places where Twitter might store the token
                            
                            // Method 1: Look in localStorage
                            for (let key of Object.keys(localStorage)) {
                                if (key.includes('token') || key.includes('auth')) {
                                    let value = localStorage.getItem(key);
                                    if (value && value.includes('AAAA')) return value;
                                }
                            }
                            
                            // Method 2: Try to find in main JS objects
                            try {
                                if (window.__INITIAL_STATE__ && window.__INITIAL_STATE__.authentication) {
                                    return window.__INITIAL_STATE__.authentication.bearerToken;
                                }
                                
                                for (let key in window) {
                                    try {
                                        let obj = window[key];
                                        if (obj && typeof obj === 'object' && obj.authorization && obj.authorization.bearerToken) {
                                            return obj.authorization.bearerToken;
                                        }
                                    } catch (e) {}
                                }
                            } catch (e) {}
                            
                            return null;
                        }''')
                        
                        if js_bearer_token:
                            # Clean up the token if needed
                            if isinstance(js_bearer_token, str) and js_bearer_token.startswith('Bearer '):
                                js_bearer_token = js_bearer_token.replace('Bearer ', '')
                            bearer_token = js_bearer_token
                            logger.info(f"Found bearer token in JavaScript context")
                    except Exception as e:
                        logger.warning(f"Error extracting bearer token from JavaScript context: {e}")
                else:
                    # Use the bearer token we captured from request interception
                    bearer_token = captured_bearer_token
                
                # Close browser
                browser.close()
                
                if not auth_token or not csrf_token or not bearer_token:
                    logger.error("Failed to extract all required authentication tokens")
                    missing = []
                    if not auth_token: missing.append("auth_token")
                    if not csrf_token: missing.append("csrf_token")
                    if not bearer_token: missing.append("bearer_token")
                    
                    raise ValueError(f"Missing authentication tokens: {', '.join(missing)}")
                
                logger.info("Successfully extracted all authentication tokens")
                return auth_token, csrf_token, bearer_token
        except Exception as e:
            logger.error(f"Failed to extract auth tokens: {e}")
            raise ValueError(f"Failed to extract authentication tokens: {e}")

    def _get_tokens(self) -> bool:
        """Ensures auth tokens are loaded."""
        if self.auth_tokens:
            return True
        extracted_tokens = self._extract_auth_tokens()
        if extracted_tokens is not None:
            self.auth_tokens = extracted_tokens  # FIXED: Store the extracted tokens
            return True
        return False

    def fetch_tweet_data_api(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetches detailed tweet data using the GraphQL API.
        Uses parameters derived from observed working requests.
        """
        logger.info(f"Fetching tweet data via API for ID: {tweet_id}")
        if not self._get_tokens():
            logger.error("Cannot fetch tweet data: Auth tokens not available.")
            return None

        if self.auth_tokens is None:
            logger.error("Auth tokens tuple is None, cannot proceed.")
            return None

        auth_token, csrf_token, bearer_token = self.auth_tokens

        # --- UPDATED API Endpoint and Parameters (Based on Burp Capture Apr 2025) ---
        api_url = "https://x.com/i/api/graphql/0hWvDhmW8YQ-S_ib3azIrw/TweetResultByRestId"

        params = {
            "variables": json.dumps({
                "tweetId": tweet_id, # Changed from focalTweetId
                "withCommunity": False,
                "includePromotedContent": False,
                "withVoice": False
                # Removed several older/unnecessary params like with_rux_injections, withV2Timeline etc.
            }),
            "features": json.dumps({ # UPDATED features based on Burp capture
                "creator_subscriptions_tweet_preview_api_enabled": False, # Was false, now false
                "tweetypie_unmention_optimization_enabled": True, # Was false, now true
                "responsive_web_edit_tweet_api_enabled": True, # Was false, now true
                "graphql_is_translatable_rweb_tweet_is_translatable_enabled": False, # Was false, now false
                "view_counts_everywhere_api_enabled": False, # Was false, now false
                "longform_notetweets_consumption_enabled": True, # Was false, now true
                "responsive_web_twitter_article_tweet_consumption_enabled": False, # Was false, now false
                "tweet_awards_web_tipping_enabled": False, # Was false, now false
                "freedom_of_speech_not_reach_fetch_enabled": True, # Was false, now true
                "standardized_nudges_misinfo": False, # Was false, now false
                "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True, # Was false, now true
                "longform_notetweets_rich_text_read_enabled": False, # Was false, now false
                "longform_notetweets_inline_media_enabled": False, # Was false, now false
                "responsive_web_graphql_exclude_directive_enabled": True, # Was false, now true
                "verified_phone_label_enabled": False, # Was false, now false
                "responsive_web_media_download_video_enabled": False, # New parameter
                "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False, # Was false, now false
                "responsive_web_graphql_timeline_navigation_enabled": False, # Was false, now false
                "responsive_web_enhance_cards_enabled": False # Was false, now false
                # Several old features removed
            }),
            "fieldToggles": json.dumps({ # Kept same as before, matches Burp
                "withArticleRichContentState": False,
                "withAuxiliaryUserLabels": False
            })
        }
        # --- END UPDATED PARAMETERS ---

        headers = {
            "Host": "x.com",
            "Cookie": f"auth_token={auth_token}; ct0={csrf_token}",
            "X-Twitter-Active-User": "yes",
            "Authorization": f"Bearer {bearer_token}",
            "X-Csrf-Token": csrf_token,
            "X-Twitter-Auth-Type": "OAuth2Session",
            # Update User-Agent slightly to match Burp capture if desired, though current one likely works
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "Content-Type": "application/json",
            "Accept": "*/*"
            # Add other headers from Burp if needed, but these are likely sufficient
        }

        try:
            logger.debug(f"Attempting API request to: {api_url} with params: {params}")
            response = requests.get(api_url, params=params, headers=headers, timeout=15)

            logger.debug(f"API Request URL (final): {response.url}")

            response.raise_for_status() # Will raise HTTPError for 4xx/5xx

            data = response.json()
            logger.info("Successfully fetched tweet data from API.")

            # --- UPDATED Validation for new response structure ---
            if 'data' not in data or 'tweetResult' not in data['data'] or 'result' not in data['data']['tweetResult']:
                logger.warning("API response structure may have changed. Missing expected 'data.tweetResult.result' path.")
                logger.debug(f"Response keys: {list(data.keys())}")
                if 'data' in data:
                    logger.debug(f"Data keys: {list(data['data'].keys())}")
            # --- END UPDATED Validation ---

            return data

        except requests.exceptions.HTTPError as e:
            # Existing enhanced logging...
            logger.error(f"API request failed with HTTP status {e.response.status_code} for URL: {e.request.url}")
            try:
                response_text = e.response.text
                logger.error(f"Raw API error response text: {response_text}")
                error_details = e.response.json()
                logger.error(f"API error details (JSON parsed): {json.dumps(error_details)}")
            except json.JSONDecodeError:
                 logger.error("API error response was not valid JSON.")
            except Exception as parse_e:
                logger.error(f"Could not parse API error response: {parse_e}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch tweet data for tweet_id {tweet_id}: {e}", exc_info=True)
            return None

    def extract_media_urls_from_api_data(self, tweet_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts all media URLs (images, GIFs, videos) from the tweet data,
        using the path observed in TweetResultByRestId responses.
        """
        logger.info("Extracting media URLs from tweet data")
        media_items = []

        try:
            # --- UPDATED Path Traversal (Based on Burp Response Apr 2025) ---
            # Old Path: ['data']['threaded_conversation_with_injections_v2']['instructions'][0]['entries'][0]['content']['itemContent']['tweet_results']['result']
            # New Path: ['data']['tweetResult']['result']
            if 'data' not in tweet_data or 'tweetResult' not in tweet_data['data'] or 'result' not in tweet_data['data']['tweetResult']:
                 logger.error("Could not find 'data.tweetResult.result' in API response for media extraction.")
                 return []

            tweet_result = tweet_data['data']['tweetResult']['result']

            # Check if the result itself indicates an issue (e.g., tweet not found, though unlikely if API returned 200)
            if tweet_result.get('__typename') == 'TweetUnavailable' or not tweet_result.get('legacy'):
                logger.warning(f"Tweet result indicates unavailability or missing legacy data: {tweet_result.get('reason', 'Unknown reason')}")
                return []
            # --- END UPDATED Path Traversal ---


            # The rest of the logic relies on the 'legacy' structure which seems consistent in the Burp response
            legacy_data = tweet_result.get('legacy', {})
            extended_entities = legacy_data.get('extended_entities', {})

            if not extended_entities or 'media' not in extended_entities:
                # Check if media might be in top-level entities for simple images
                entities = legacy_data.get('entities', {})
                if entities and 'media' in entities:
                     logger.info("Found media in legacy.entities (likely simple image).")
                     extended_entities = entities # Treat entities.media like extended_entities.media
                else:
                     logger.info("No extended_entities.media or entities.media found in tweet legacy data.")
                     return [] # Return empty list if no media found

            # Process all media items (logic remains the same as it depends on the 'media' array structure)
            for index, media in enumerate(extended_entities.get('media', [])): # Use .get for safety
                media_type = media.get('type', '')
                media_item = {
                    'type': media_type,
                    'index': index,
                    'url': None,
                    'extension': None
                }

                if media_type == 'photo':
                    media_item['url'] = media.get('media_url_https', '')
                    media_item['extension'] = 'jpg'
                elif media_type == 'video':
                    video_info = media.get('video_info', {})
                    variants = video_info.get('variants', [])
                    mp4_variants = [v for v in variants if v.get('content_type') == 'video/mp4']
                    if mp4_variants:
                        best_variant = max(mp4_variants, key=lambda v: v.get('bitrate', 0))
                        media_item['url'] = best_variant['url']
                        media_item['extension'] = 'mp4'
                elif media_type == 'animated_gif':
                    video_info = media.get('video_info', {})
                    variants = video_info.get('variants', [])
                    if variants:
                        media_item['url'] = variants[0]['url']
                        media_item['extension'] = 'mp4'

                if media_item['url']:
                    # Basic URL cleaning - remove query params like ?tag=10
                    parsed_url = urlparse(media_item['url'])
                    cleaned_url = parsed_url._replace(query='').geturl()
                    media_item['url'] = cleaned_url

                    # Determine extension more reliably
                    path_part = parsed_url.path
                    if '.' in path_part:
                         potential_ext = path_part.split('.')[-1].lower()
                         if potential_ext in ['jpg', 'jpeg', 'png', 'webp', 'mp4', 'gif']:
                              media_item['extension'] = potential_ext
                         elif media_item['type'] == 'photo': # Fallback for photos
                              media_item['extension'] = 'jpg'
                         elif media_item['type'] in ['video', 'animated_gif']: # Fallback for videos/gifs
                              media_item['extension'] = 'mp4'


                    media_items.append(media_item)
                    logger.info(f"Found {media_type} URL: {media_item['url']} (Extension: {media_item['extension']})")
                else:
                    logger.warning(f"Could not extract URL for {media_type} at index {index}")

            return media_items

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to extract media URLs due to path error: {e}", exc_info=True)
            # Log structure for debugging
            logger.debug(f"API Data structure (keys): {list(tweet_data.keys())}")
            if 'data' in tweet_data:
                 logger.debug(f"API Data['data'] structure (keys): {list(tweet_data['data'].keys())}")
                 if 'tweetResult' in tweet_data['data']:
                      logger.debug(f"API Data['data']['tweetResult'] structure (keys): {list(tweet_data['data']['tweetResult'].keys())}")


            return [] # Return empty list on error