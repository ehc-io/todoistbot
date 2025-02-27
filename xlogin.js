const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

// Configuration variables
const TARGET_URL = "https://x.com/home";
const SESSION_DATA_DIR = "session-data";
const SESSION_DATA_PATH = path.join(SESSION_DATA_DIR, "session.json");

// Timeout configurations (in milliseconds)
const PAGE_LOAD_TIMEOUT = 3000;
const LOGIN_WAIT_TIMEOUT = 10000;
const FORM_INTERACTION_DELAY = 1500;
const SELECTOR_TIMEOUT = 3000;

/**
 * Logger utility to provide consistent timestamp format
 */
class Logger {
  static log(message) {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] ${message}`);
  }
  
  static error(message, err) {
    const timestamp = new Date().toISOString();
    console.error(`[${timestamp}] ERROR: ${message}`, err || '');
  }
}

/**
 * Formats timestamp for filenames by replacing characters not allowed in filenames
 * @returns {string} Formatted timestamp for filenames
 */
function getFormattedTimestamp() {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

/**
 * Main function to handle X.com session automation
 */
async function scrapeXHomepage() {
  Logger.log('Starting X.com session manager...');
  
  // Create session data directory if it doesn't exist
  if (!fs.existsSync(SESSION_DATA_DIR)) {
    fs.mkdirSync(SESSION_DATA_DIR, { recursive: true });
    Logger.log(`Created session data directory: ${SESSION_DATA_DIR}`);
  }
  
  // Get credentials from environment variables
  const username = process.env.X_USERNAME;
  const password = process.env.X_PASSWORD;
  
  if (!username || !password) {
    throw new Error('X_USERNAME and X_PASSWORD environment variables must be set');
  }
  
  // Launch browser
  Logger.log('Launching browser...');
  const browser = await chromium.launch({
    headless: true
  });
  
  try {
    // Create browser context
    let context;
    let hasValidSession = false;
    
    // Check if we have session data
    if (fs.existsSync(SESSION_DATA_PATH)) {
      Logger.log('Found existing session data, attempting to restore session...');
      try {
        // Load session data
        const sessionData = JSON.parse(fs.readFileSync(SESSION_DATA_PATH, 'utf8'));
        context = await browser.newContext({
          storageState: sessionData
        });
        
        // Verify if session is still valid
        const page = await context.newPage();
        await page.goto(TARGET_URL);
        await page.waitForTimeout(PAGE_LOAD_TIMEOUT);
        
        // Verify login success
        const isLoginSuccessful = await page.evaluate(() => {
          return document.title.includes("Home");
        });

        if (isLoginSuccessful) {
          Logger.log('Session is valid, continuing with existing session');
          hasValidSession = true;
          
          // Take screenshot of valid session
          await page.screenshot({ path: `${getFormattedTimestamp()}-valid-session.png` });
        } else {
          Logger.log('Session is no longer valid, will attempt fresh login');
          await page.close();
          await context.close();
          context = null;
        }
      } catch (error) {
        Logger.error('Error while restoring session:', error);
        Logger.log('Will attempt fresh login');
        if (context) await context.close();
        context = null;
      }
    } else {
      Logger.log('No session data found, will perform login');
    }
    
    // If session is not valid, perform login
    if (!hasValidSession) {
      Logger.log('Creating new browser context for login...');
      context = await browser.newContext();
      const page = await context.newPage();
      
      Logger.log('Navigating to X.com homepage...');
      await page.goto(TARGET_URL);
      await page.waitForTimeout(PAGE_LOAD_TIMEOUT);
      
      // Take initial screenshot
      await page.screenshot({ path: `${getFormattedTimestamp()}-before-login.png` });
      
      Logger.log(`Attempting to login with username: ${username}`);
      
      // Fill username
      await page.waitForSelector('input[name="text"]', { state: 'visible', timeout: SELECTOR_TIMEOUT });
      await page.fill('input[name="text"]', username);
      
      // Take screenshot after putting username
      await page.screenshot({ path: `${getFormattedTimestamp()}-after-username.png` });

      // Click Next button
      await page.evaluate(() => {
        const buttons = Array.from(document.querySelectorAll('button[type="button"]'));
        const nextButton = buttons.find(button => button.textContent.includes('Next'));
        if (nextButton) {
          nextButton.click();
        } else {
          throw new Error('Next button not found');
        }
      });
      
      await page.waitForTimeout(PAGE_LOAD_TIMEOUT);

      // Take screenshot after clicking Next
      await page.screenshot({ path: `${getFormattedTimestamp()}-after-next-button.png` });

      // Wait for password field and fill it
      await page.waitForSelector('input[name="password"]', { state: 'visible', timeout: SELECTOR_TIMEOUT });
      await page.fill('input[name="password"]', password);
      await page.waitForTimeout(FORM_INTERACTION_DELAY);

      // Take screenshot after putting password
      await page.screenshot({ path: `${getFormattedTimestamp()}-after-password.png` });

      // Click Login button
      await page.waitForSelector('button[data-testid="LoginForm_Login_Button"]', { state: 'visible', timeout: SELECTOR_TIMEOUT });
      await page.click('button[data-testid="LoginForm_Login_Button"]');
      
      await page.waitForTimeout(LOGIN_WAIT_TIMEOUT);

      // Take screenshot after Login button
      await page.screenshot({ path: `${getFormattedTimestamp()}-after-login-button.png` });

      // Verify login success
      const isLoginSuccessful = await page.evaluate(() => {
        return document.title.includes("Home");
      });

      if (!isLoginSuccessful) {
        throw new Error('Login failed: Could not verify Home page after login attempt');
      }
      
      Logger.log('Login successful');
      
      // Take screenshot after successful login
      await page.screenshot({ path: `${getFormattedTimestamp()}-after-login.png` });
      
      // Save session data
      Logger.log('Saving session data...');
      const storageState = await context.storageState();
      fs.writeFileSync(SESSION_DATA_PATH, JSON.stringify(storageState, null, 2));
      
      hasValidSession = true;
    }
    
    // Instead of loading the page again, just provide success message
    if (hasValidSession) {
      Logger.log('SUCCESS: X.com authentication credentials successfully obtained and stored');
    }
  } catch (error) {
    Logger.error('An error occurred during execution:', error);
    throw error; // Re-throw to ensure the process exits with non-zero code on failure
  } finally {
    Logger.log('Closing browser...');
    await browser.close();
  }
}

// Execute the main function
scrapeXHomepage().catch(err => {
  Logger.error('Fatal error:', err);
  process.exit(1);
});