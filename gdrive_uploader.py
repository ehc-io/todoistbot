import os
import logging
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

logger = logging.getLogger(__name__)

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_gdrive_service():
    """
    Authenticates with Google Drive API using a pre-generated gdrive_token.json.
    Handles token refreshing automatically for unattended operation.
    """
    creds = None
    # Prioritize token path from environment variable, otherwise use the vault path as default.
    token_path = os.getenv('GOOGLE_TOKEN_PATH', '/mnt/storage/vault/gdrive_token.json')

    # Check for the existence of the token file
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If credentials are not loaded or not valid, handle the situation.
    if not creds or not creds.valid:
        # If the token is expired and has a refresh token, try to refresh it.
        if creds and creds.expired and creds.refresh_token:
            try:
                logger.info("Google Drive token has expired, attempting to refresh...")
                creds.refresh(Request())
                # Persist the refreshed credentials for subsequent runs
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
                logger.info("Token refreshed and saved successfully.")
            except Exception as e:
                logger.error(f"Failed to refresh Google token: {e}", exc_info=True)
                logger.error("The refresh token might be expired or revoked. "
                             "Please re-run the 'generate_gdrive_token.py' script to get a new token.")
                return None
        else:
            # This block is reached if the token file doesn't exist or is invalid without a refresh token.
            logger.error(f"Token file not found or is invalid at '{token_path}'.")
            logger.error("For unattended operation, a valid token file is required.")
            logger.error("Please run the 'generate_gdrive_token.py' script to create it and ensure it's at the correct path.")
            return None

    try:
        service = build('drive', 'v3', credentials=creds)
        logger.debug("Successfully authenticated with Google Drive.")
        return service
    except HttpError as error:
        logger.error(f'An error occurred while building Google Drive service: {error}')
        return None

def upload_to_gdrive(file_path, folder_id):
    """
    Uploads a file to a specific Google Drive folder.

    Args:
        file_path (str): The local path of the file to upload.
        folder_id (str): The ID of the Google Drive folder.

    Returns:
        str: The ID of the uploaded file, or None on failure.
    """
    service = get_gdrive_service()
    if not service:
        logger.error("Could not get Google Drive service. Upload failed.")
        return None

    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id]
    }
    # Ensure mimetype is set correctly for markdown
    media = MediaFileUpload(file_path, mimetype='text/markdown', resumable=True)

    try:
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink' # Request webViewLink to provide a direct link
        ).execute()
        file_id = file.get('id')
        file_link = file.get('webViewLink')
        logger.info(f"✓ Report successfully uploaded to Google Drive. File ID: {file_id}")
        logger.info(f"  Direct link: {file_link}")
        return file_id
    except HttpError as error:
        logger.error(f'An error occurred during Google Drive upload: {error}')
        # Check for common errors
        if error.resp.status == 404:
            logger.error("  Error 404: The specified Google Drive folder ID might be incorrect or you may not have permission to access it.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during upload: {e}")
        return None 