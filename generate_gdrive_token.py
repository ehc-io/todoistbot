import os
import logging
import argparse
import webbrowser
from google_auth_oauthlib.flow import InstalledAppFlow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the scope required for the application.
# This must match the scopes in your main application.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Define the default name for the token file.
TOKEN_FILE = 'token.json'

def generate_token(credentials_path, output_path):
    """
    Runs an interactive flow to generate and save a token file for Google Drive API access.
    This script should be run once, locally, to authorize the application.
    
    Args:
        credentials_path (str): The path to the OAuth 2.0 credentials file.
        output_path (str): The path where the generated token file will be saved.
    """
    # Check if the credentials file exists.
    if not os.path.exists(credentials_path):
        logger.error(f"Credentials file not found at: '{credentials_path}'")
        logger.error("Please download your OAuth 2.0 credentials from the Google Cloud Console")
        logger.error("and provide the correct path using the --credentials argument.")
        return

    logger.info(f"Loading credentials from '{credentials_path}'...")
    
    # Create an InstalledAppFlow instance to handle the OAuth 2.0 flow.
    # This will use the credentials file to identify the application.
    flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)

    # run_local_server will start a local webserver, open the user's browser
    # to the authorization page, and handle the OAuth 2.0 dance.
    # The 'port=0' argument lets it find an available port automatically.
    # A fallback to manual authentication is provided for headless environments.
    try:
        creds = flow.run_local_server(port=0)
    except (webbrowser.Error, OSError) as e:
        logger.info(f"Could not start local web server ({e}), falling back to manual console authentication.")
        
        # Generate the authorization URL and prompt the user to visit it.
        auth_url, _ = flow.authorization_url(prompt='consent')
        
        print("\n--- Manual Google Drive Authentication ---")
        print("Please open this URL in a browser on any machine to authorize the application:")
        print(f"\n{auth_url}\n")
        
        # Ask the user to enter the authorization code they receive.
        code = input("After authorization, copy the code provided and paste it here: ")

        # Exchange the authorization code for a credentials object.
        try:
            flow.fetch_token(code=code)
            creds = flow.credentials
        except Exception as fetch_error:
            logger.error(f"Failed to fetch token using the provided code: {fetch_error}")
            logger.error("Please ensure you are copying the code correctly.")
            return

    # The flow is complete, and we have the credentials.
    # Now, save the credentials to the specified output file.
    try:
        # Ensure the output directory exists before trying to write the file
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_path, 'w') as token:
            token.write(creds.to_json())
        logger.info(f"✓ Authentication successful. Token saved to '{output_path}'.")
        logger.info("You can now use this token file in your application.")
    except (IOError, NameError) as e:
        if isinstance(e, NameError):
             logger.error("Authentication was not completed, so no token file was generated.")
        else:
            logger.error(f"Error saving token file to '{output_path}': {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a Google Drive API token file for unattended use.")
    parser.add_argument(
        '--credentials',
        type=str,
        default='googledrive_credentials.json',
        help="Path to the Google OAuth 2.0 credentials file. Defaults to 'googledrive_credentials.json'."
    )
    parser.add_argument(
        '--output',
        type=str,
        default='gdrive_token.json',
        help="Path to save the generated token file. Defaults to 'gdrive_token.json'."
    )
    args = parser.parse_args()
    
    generate_token(args.credentials, args.output) 