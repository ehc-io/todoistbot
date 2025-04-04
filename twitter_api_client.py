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
        Assumes the session file is valid.
        Returns (auth_token, csrf_token, bearer_token) or None if extraction fails.
        """
        if not self.session_path.exists():
            logger.error("Cannot extract tokens: Session file does not exist.")
            return None

        logger.info("Extracting auth tokens from session using Playwright...")
        browser = None
        context = None
        page = None
        captured_bearer_token = None
        auth_token = None
        csrf_token = None
        bearer_token = None
        last_intercepted_token = None # To avoid logging duplicates

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                # ---> FIX: Ensure context loads storage state correctly <---
                context = browser.new_context(storage_state=str(self.session_path))
                page = context.new_page()

                def handle_request(route: Route):
                    nonlocal captured_bearer_token, last_intercepted_token
                    request: Request = route.request
                    # Don't process headers if token already found
                    if captured_bearer_token:
                        try: route.continue_() # Still need to continue
                        except PlaywrightError as cont_err: logger.debug(f"Error continuing route {request.url} after token found: {cont_err}")
                        return

                    # ---> FIX: Match extractor's interception logic more closely <---
                    is_api_call = ('api.twitter.com' in request.url or
                                   'twitter.com/i/api' in request.url or
                                   'x.com/i/api' in request.url)

                    if is_api_call:
                        auth_header = request.headers.get('authorization', '')
                        if auth_header.startswith('Bearer AAAA'): # Check for the specific bearer token format
                            token = auth_header.split(' ')[1]
                            # Only log/capture if it's different from the last one seen
                            if token != last_intercepted_token:
                                if not captured_bearer_token: # Capture the first valid one we see
                                    logger.info(f"Intercepted Bearer token: {token[:20]}...")
                                    captured_bearer_token = token
                                last_intercepted_token = token # Update last seen token

                    # Ensure the request continues regardless of interception logic
                    try:
                        route.continue_()
                    except PlaywrightError as cont_err:
                        logger.debug(f"Error continuing route {request.url}: {cont_err}")

                page.route("**/*", handle_request)

                try:
                    logger.info("Navigating to x.com/home...")
                    # Navigate without strict wait_until, then wait for a specific element
                    page.goto("https://x.com/home", timeout=30000, wait_until="domcontentloaded") # Use domcontentloaded

                    # ---> FIX: Use a more reliable indicator or longer wait after navigation <---
                    # Instead of just compose button, wait for timeline or increase general wait
                    logger.info("Waiting after navigation for potential API calls...")
                    page.wait_for_timeout(5000) # Wait 5 seconds, similar to extractor logic

                    # Optional: Wait for a known element like timeline if needed
                    # try:
                    #     timeline_selector = '[data-testid="primaryColumn"]' # Example selector
                    #     page.wait_for_selector(timeline_selector, timeout=15000)
                    #     logger.info("Timeline indicator found.")
                    # except PlaywrightError:
                    #     logger.warning("Timeline indicator not found within timeout, proceeding anyway.")


                    # Now wait specifically for the bearer token if not found yet via interception
                    start_time = time.time()
                    wait_token_timeout = 5 # Wait max 5 *additional* seconds for token
                    while not captured_bearer_token and time.time() - start_time < wait_token_timeout:
                         logger.debug("Waiting for bearer token capture...")
                         page.wait_for_timeout(500) # Check every 500ms

                    if not captured_bearer_token:
                         logger.warning(f"Bearer token not captured via interception after waiting.")
                         # NOTE: Fallback logic (like JS evaluation from extractor) could be added here if interception remains unreliable.
                         # For now, we rely on interception working during the initial load + wait.

                    bearer_token = captured_bearer_token

                    # Extract cookies
                    cookies = context.cookies()
                    auth_token = next((c["value"] for c in cookies if c["name"] == "auth_token"), None)
                    csrf_token = next((c["value"] for c in cookies if c["name"] == "ct0"), None)

                except PlaywrightError as e:
                    logger.error(f"Playwright error during token extraction page load/wait: {e}")
                    # Attempt to capture screenshot on error
                    try:
                        screenshot_path = f"playwright_error_{time.strftime('%Y%m%d_%H%M%S')}.png"
                        page.screenshot(path=screenshot_path, full_page=True)
                        logger.info(f"Screenshot saved to {screenshot_path} on error.")
                    except Exception as se:
                        logger.error(f"Failed to take screenshot on error: {se}")


        except Exception as e:
            logger.error(f"Unexpected error during token extraction setup: {e}", exc_info=True)

        finally:
            # Graceful cleanup
            if page:
                try: page.close()
                except Exception as e_page: logger.debug(f"Error closing page: {e_page}")
            if context:
                try: context.close()
                except Exception as e_context: logger.debug(f"Error closing context: {e_context}")
            if browser:
                try: browser.close()
                except Exception as e_browser: logger.debug(f"Error closing browser: {e_browser}")

        # Validation logic
        if not all([auth_token, csrf_token, bearer_token]):
            missing = [name for name, val in [('auth_token', auth_token), ('csrf_token', csrf_token), ('bearer_token', bearer_token)] if not val]
            logger.error(f"Failed to extract all required tokens. Missing: {', '.join(missing)}")
            return None

        logger.info("Successfully extracted auth tokens.")
        self.auth_tokens = (auth_token, csrf_token, bearer_token) # Store tokens on success
        return self.auth_tokens

    def _get_tokens(self) -> bool:
        """Ensures auth tokens are loaded."""
        if self.auth_tokens:
            return True
        extracted_tokens = self._extract_auth_tokens()
        return extracted_tokens is not None

    def fetch_tweet_data_api(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetches detailed tweet data using the GraphQL API.
        Requires valid auth tokens.
        """
        logger.info(f"Fetching tweet data via API for ID: {tweet_id}")
        if not self._get_tokens():
            logger.error("Cannot fetch tweet data: Auth tokens not available.")
            return None

        # Ensure self.auth_tokens is not None before unpacking
        if self.auth_tokens is None:
             logger.error("Auth tokens tuple is None, cannot proceed.")
             return None
        auth_token, csrf_token, bearer_token = self.auth_tokens

        # API Endpoint (same as extractor)
        api_url = "https://x.com/i/api/graphql/zJvfJs3gSbrVhC0MKjt_OQ/TweetDetail"

        # --- VVV FIX: Use EXACT parameters/features/toggles from the working extractor script ---
        params = {
            "variables": json.dumps({
                "focalTweetId": tweet_id,
                "with_rux_injections": False,
                "includePromotedContent": False,
                "withCommunity": False,
                "withQuickPromoteEligibilityTweetFields": False,
                "withBirdwatchNotes": False,
                "withVoice": False,
                "withV2Timeline": True # Kept from original, seems important and present in extractor
            }),
            # Use the features dict from twitter-media-extractor.py
            "features": json.dumps({
                "rweb_tipjar_consumption_enabled": False, # Aligned with extractor
                "responsive_web_graphql_exclude_directive_enabled": False, # Aligned with extractor
                "verified_phone_label_enabled": False, # Aligned with extractor
                "creator_subscriptions_tweet_preview_api_enabled": False, # Aligned with extractor
                "responsive_web_graphql_timeline_navigation_enabled": False, # Aligned with extractor
                "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False, # Aligned with extractor
                "communities_web_enable_tweet_community_results_fetch": False, # Aligned with extractor
                "c9s_tweet_anatomy_moderator_badge_enabled": False, # Aligned with extractor
                "articles_preview_enabled": True, # Aligned with extractor
                "tweetypie_unmention_optimization_enabled": False, # Aligned with extractor
                "responsive_web_edit_tweet_api_enabled": False, # Aligned with extractor
                "graphql_is_translatable_rweb_tweet_is_translatable_enabled": False, # Aligned with extractor
                "view_counts_everywhere_api_enabled": False, # Aligned with extractor
                "longform_notetweets_consumption_enabled": False, # Aligned with extractor
                "responsive_web_twitter_article_tweet_consumption_enabled": False, # Aligned with extractor
                "tweet_awards_web_tipping_enabled": False, # Aligned with extractor
                "creator_subscriptions_quote_tweet_preview_enabled": False, # Aligned with extractor
                "freedom_of_speech_not_reach_fetch_enabled": False, # Aligned with extractor
                "standardized_nudges_misinfo": False, # Aligned with extractor
                "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": False, # Aligned with extractor
                "tweet_with_visibility_results_prefer_gql_media_interstitial_enabled": False, # Aligned with extractor
                "rweb_video_timestamps_enabled": False, # Aligned with extractor
                "longform_notetweets_rich_text_read_enabled": False, # Aligned with extractor
                "longform_notetweets_inline_media_enabled": False, # Aligned with extractor
                "responsive_web_enhance_cards_enabled": False # Aligned with extractor
            }),
             # Use the fieldToggles dict from twitter-media-extractor.py
            "fieldToggles": json.dumps({
                "withArticleRichContentState": False, # Aligned with extractor
                "withAuxiliaryUserLabels": False # Aligned with extractor
            })
        }

        headers = {
            # Use headers exactly as in twitter-media-extractor.py
            "Host": "x.com", # Added from extractor
            "Cookie": f"auth_token={auth_token}; ct0={csrf_token}", # Crucial cookies
            "X-Twitter-Active-User": "yes",
            "Authorization": f"Bearer {bearer_token}",
            "X-Csrf-Token": csrf_token,
            "X-Twitter-Auth-Type": "OAuth2Session",
             # Use the User-Agent from the extractor script
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            "Content-Type": "application/json",
            "Accept": "*/*",
            # Removed Referer and X-Client-Uuid as they are not in the extractor's request
        }
        # --- ^^^ END FIX ---

        try:
            # Increased timeout slightly
            response = requests.get(api_url, params=params, headers=headers, timeout=20)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            data = response.json()
            logger.info("Successfully fetched tweet data from API.")

            # Basic validation of response structure
            # ---> FIX: Adapt validation based on the expected structure from extractor's features <---
            if 'data' not in data or 'threaded_conversation_with_injections_v2' not in data['data']:
                 logger.warning("API response structure might have changed or differs from expectation. Expected keys ('data', 'threaded_conversation_with_injections_v2') not found directly.")
                 # Log more details for debugging if structure is unexpected
                 logger.debug(f"API Response keys: {list(data.keys())}")
                 if 'data' in data: logger.debug(f"Data keys: {list(data['data'].keys())}")
                 else: logger.debug("Response has no 'data' key.")
                 # Consider returning data even if structure is unexpected, let caller handle it
                 # return data # Optional: return potentially malformed data

            return data

        except requests.exceptions.HTTPError as e:
            logger.error(f"API request failed with HTTPError: {e}")
            logger.error(f"Response status: {e.response.status_code}")
            # Log response body for 4xx/5xx errors to understand the reason
            try:
                error_details = e.response.json()
                logger.error(f"Response JSON body: {json.dumps(error_details, indent=2)}")
            except json.JSONDecodeError:
                logger.error(f"Response text body: {e.response.text[:1000]}...") # Log snippet
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed with RequestException: {e}")
            return None
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode API response JSON: {e}")
             # Log the raw response text that failed to parse
             if 'response' in locals() and response:
                 logger.error(f"Raw response text causing JSON error: {response.text[:1000]}...")
             return None

    def extract_media_urls_from_api_data(self, tweet_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Extracts media URLs (image, video, gif) from the raw API tweet data.
        Prioritizes higher quality formats.
        Returns a list of media items or None.
        """
        # ---> FIX: Adjust JSON path traversal based on the expected response structure
        # The structure used in twitter-media-extractor.py seems more direct:
        # data -> threaded_conversation_with_injections_v2 -> instructions -> [0] -> entries -> [0] -> content -> itemContent -> tweet_results -> result -> legacy -> extended_entities -> media
        # This assumes the first instruction and first entry contain the main tweet. This is common but might break for complex conversations.

        if not tweet_data or 'data' not in tweet_data:
            logger.warning("Cannot extract media: Invalid or empty tweet_data provided.")
            return None

        media_items = []
        try:
            # Navigate the expected structure
            instructions = tweet_data.get('data', {}).get('threaded_conversation_with_injections_v2', {}).get('instructions', [])
            if not instructions:
                logger.warning("Could not find 'instructions' in API response data.")
                return None

            # Find the 'TimelineAddEntries' instruction, often the first one
            entries = []
            for instruction in instructions:
                if instruction.get('type') == 'TimelineAddEntries':
                    entries.extend(instruction.get('entries', []))
                    # Break if we assume the first AddEntries is sufficient, or iterate all
                    # break # Let's stick to iterating all for robustness

            if not entries:
                 logger.warning("Could not find 'TimelineAddEntries' instructions or they were empty.")
                 return None

            tweet_results = None
            # Iterate through entries to find the one containing the main tweet item
            for entry in entries:
                # Check different potential paths as structure can vary slightly
                content = entry.get('content', {})
                item_content = content.get('itemContent', {})
                tweet_results_candidate = item_content.get('tweet_results', {}).get('result')

                # Check if it's a valid tweet result (not a tombstone placeholder etc.)
                # and potentially matches the requested tweet_id if needed (though TweetDetail usually focuses on one)
                if tweet_results_candidate and tweet_results_candidate.get('__typename') == 'Tweet':
                    # Found the main tweet entry
                    tweet_results = tweet_results_candidate
                    break
                elif content.get('entryType') == 'TimelineTimelineItem' and item_content.get('itemType') == 'Tweet':
                     # Alternative structure check
                     if tweet_results_candidate and tweet_results_candidate.get('__typename') == 'Tweet':
                         tweet_results = tweet_results_candidate
                         break


            if not tweet_results:
                 logger.warning("Could not find valid 'tweet_results' (__typename: Tweet) in API response entries.")
                 # Log structure for debugging
                 # logger.debug(f"Entries structure sample: {json.dumps(entries[0] if entries else {}, indent=2)[:1000]}")
                 return None

            # Check for tombstone (deleted/unavailable tweet) - Already checked __typename above, but double check legacy if needed
            if tweet_results.get('__typename') != 'Tweet': # e.g., 'TweetTombstone'
                reason = "Unknown"
                if tweet_results.get('__typename') == 'TweetTombstone':
                    reason = tweet_results.get('tombstone',{}).get('text',{}).get('text', 'Unavailable')
                logger.warning(f"Tweet is unavailable ({tweet_results.get('__typename')}). Reason: {reason}")
                return [] # Return empty list for unavailable tweets

            # Navigate to media entities (use legacy path as in extractor)
            legacy_data = tweet_results.get('legacy', {})
            extended_entities = legacy_data.get('extended_entities', {})
            media_list = extended_entities.get('media', [])

            if not media_list:
                logger.info("No media found in the 'legacy.extended_entities.media' of the API data.")
                return [] # Return empty list if no media

            for index, media in enumerate(media_list):
                media_type = media.get('type')
                media_item = {'index': index, 'type': media_type, 'url': None, 'extension': None}

                if media_type == 'photo':
                    # Use original quality URL format if possible
                    base_url = media.get('media_url_https')
                    if base_url:
                         # Extractor just uses media_url_https directly. Let's stick to that for simplicity unless proven otherwise.
                         # Twitter often appends ?format=jpg&name=orig or similar via params, not ':orig' suffix now.
                         # The base URL itself is usually the highest res available via this API path.
                         media_item['url'] = base_url
                         # Determine extension from URL path, default to jpg
                         parsed_path = urlparse(base_url).path
                         ext = Path(parsed_path).suffix[1:].lower() if Path(parsed_path).suffix else None
                         media_item['extension'] = ext if ext in ['jpg', 'jpeg', 'png', 'webp'] else 'jpg'
                    else:
                         logger.warning(f"Photo media item at index {index} missing 'media_url_https'.")

                elif media_type in ['video', 'animated_gif']:
                    video_info = media.get('video_info', {})
                    variants = video_info.get('variants', [])
                    # Filter for mp4 and sort by bitrate (higher is better)
                    mp4_variants = [v for v in variants if v.get('content_type') == 'video/mp4' and v.get('bitrate') is not None]
                    if mp4_variants:
                        # Sort by bitrate descending
                        mp4_variants.sort(key=lambda v: v['bitrate'], reverse=True)
                        best_variant = mp4_variants[0]
                        media_item['url'] = best_variant['url']
                        media_item['extension'] = 'mp4'
                    elif variants: # Fallback if no MP4 found
                         non_mp4_variants = [v for v in variants if v.get('url')]
                         if non_mp4_variants:
                            # Try to find HLS/M3U8 - not directly downloadable but maybe useful info
                            hls_variant = next((v for v in non_mp4_variants if v.get('content_type') == 'application/x-mpegURL'), None)
                            if hls_variant:
                                logger.warning(f"No MP4 variant found for {media_type} at index {index}. Found HLS playlist: {hls_variant['url']}")
                                # Don't set URL as it's not directly downloadable
                            else:
                                logger.warning(f"No MP4 variant found for {media_type} at index {index}. Found other types: {[v.get('content_type') for v in non_mp4_variants]}")
                                # Could potentially try the first available URL as a last resort, but likely not MP4
                                # best_fallback = max(non_mp4_variants, key=lambda v: v.get('bitrate', 0)) if any('bitrate' in v for v in non_mp4_variants) else non_mp4_variants[0]
                                # media_item['url'] = best_fallback['url']
                                # media_item['extension'] = 'ts' # guess?
                         else:
                            logger.warning(f"No usable video variants with URLs found for {media_type} at index {index}.")
                    else:
                        logger.warning(f"{media_type.capitalize()} media item at index {index} missing 'video_info' or 'variants'.")

                else:
                    logger.warning(f"Unsupported media type '{media_type}' encountered at index {index}.")

                if media_item.get('url'):
                    media_items.append(media_item)
                    logger.info(f"Found {media_type} URL (index {index}): ...{media_item['url'][-50:]}") # Log end of URL
                else:
                     logger.warning(f"Could not extract valid/downloadable URL for {media_type} at index {index}.")

            return media_items

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing API data for media URLs: {e}. Structure might have changed.", exc_info=True)
            # Log structure for debugging
            logger.debug(f"Problematic tweet_data structure snippet: {json.dumps(tweet_data, indent=2)[:1500]}...")
            return None # Indicate failure to parse