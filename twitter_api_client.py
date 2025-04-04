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
        Exactly matches the parameters from the original working script.
        """
        logger.info(f"Fetching tweet data via API for ID: {tweet_id}")
        if not self._get_tokens():
            logger.error("Cannot fetch tweet data: Auth tokens not available.")
            return None

        # Ensure tokens are available
        if self.auth_tokens is None:
            logger.error("Auth tokens tuple is None, cannot proceed.")
            return None
        
        auth_token, csrf_token, bearer_token = self.auth_tokens

        # API Endpoint (same as original script)
        api_url = "https://x.com/i/api/graphql/zJvfJs3gSbrVhC0MKjt_OQ/TweetDetail"
        
        # Use EXACTLY the same parameters as the original script
        params = {
            "variables": json.dumps({
                "focalTweetId": tweet_id,
                "with_rux_injections": False,
                "includePromotedContent": False,
                "withCommunity": False,
                "withQuickPromoteEligibilityTweetFields": False,
                "withBirdwatchNotes": False,
                "withVoice": False,
                "withV2Timeline": True
            }),
            "features": json.dumps({
                "rweb_tipjar_consumption_enabled": False,
                "responsive_web_graphql_exclude_directive_enabled": False,
                "verified_phone_label_enabled": False,
                "creator_subscriptions_tweet_preview_api_enabled": False,
                "responsive_web_graphql_timeline_navigation_enabled": False,
                "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
                "communities_web_enable_tweet_community_results_fetch": False,
                "c9s_tweet_anatomy_moderator_badge_enabled": False,
                "articles_preview_enabled": True,
                "tweetypie_unmention_optimization_enabled": False,
                "responsive_web_edit_tweet_api_enabled": False,
                "graphql_is_translatable_rweb_tweet_is_translatable_enabled": False,
                "view_counts_everywhere_api_enabled": False,
                "longform_notetweets_consumption_enabled": False,
                "responsive_web_twitter_article_tweet_consumption_enabled": False,
                "tweet_awards_web_tipping_enabled": False,
                "creator_subscriptions_quote_tweet_preview_enabled": False,
                "freedom_of_speech_not_reach_fetch_enabled": False,
                "standardized_nudges_misinfo": False,
                "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": False,
                "tweet_with_visibility_results_prefer_gql_media_interstitial_enabled": False,
                "rweb_video_timestamps_enabled": False,
                "longform_notetweets_rich_text_read_enabled": False,
                "longform_notetweets_inline_media_enabled": False,
                "responsive_web_enhance_cards_enabled": False
            }),
            "fieldToggles": json.dumps({
                "withArticleRichContentState": False,
                "withAuxiliaryUserLabels": False
            })
        }
        
        # Use EXACTLY the same headers as the original script
        headers = {
            "Host": "x.com",
            "Cookie": f"auth_token={auth_token}; ct0={csrf_token}",
            "X-Twitter-Active-User": "yes",
            "Authorization": f"Bearer {bearer_token}",
            "X-Csrf-Token": csrf_token,
            "X-Twitter-Auth-Type": "OAuth2Session",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            "Content-Type": "application/json",
            "Accept": "*/*"
        }
        
        try:
            # Make the API request
            response = requests.get(api_url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            logger.info("Successfully fetched tweet data from API.")
            
            # Basic validation of expected structure
            if 'data' not in data or 'threaded_conversation_with_injections_v2' not in data['data']:
                logger.warning("API response structure may have changed. Missing expected data paths.")
                logger.debug(f"Response keys: {list(data.keys())}")
                if 'data' in data:
                    logger.debug(f"Data keys: {list(data['data'].keys())}")
            
            return data
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"API request failed with HTTP status {e.response.status_code}")
            try:
                error_details = e.response.json()
                logger.error(f"API error details: {json.dumps(error_details)[:500]}")
            except:
                logger.error(f"API error response: {e.response.text[:500]}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch tweet data: {e}")
            return None

    def extract_media_urls_from_api_data(self, tweet_data: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            except Exception as debug_e:
                logger.error(f"Error while debugging structure: {debug_e}")
            
            return []  # Return empty list on error