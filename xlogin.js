const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const target_url = "https://x.com/home";
const sessionDataDir = "session-data";
const sessionDataPath = path.join(sessionDataDir, "session.json");

async function scrapeXHomepage() {
  console.log('Starting X.com automation...');
  
  // Create session data directory if it doesn't exist
  if (!fs.existsSync(sessionDataDir)) {
    fs.mkdirSync(sessionDataDir, { recursive: true });
    console.log(`Created session data directory: ${sessionDataDir}`);
  }
  
  // Get credentials from environment variables
  const username = process.env.X_USERNAME;
  const password = process.env.X_PASSWORD;
  
  if (!username || !password) {
    throw new Error('X_USERNAME and X_PASSWORD environment variables must be set');
  }
  
  // Launch browser
  console.log('Launching browser...');
  const browser = await chromium.launch({
    headless: true
  });
  
  try {
    // Create browser context
    let context;
    let hasValidSession = false;
    
    // Check if we have session data
    if (fs.existsSync(sessionDataPath) && fs.readdirSync(sessionDataDir).length > 0) {
      console.log('Found existing session data, attempting to restore session...');
      try {
        // Load session data
        const sessionData = JSON.parse(fs.readFileSync(sessionDataPath, 'utf8'));
        context = await browser.newContext({
          storageState: sessionData
        });
        
        // Verify if session is still valid
        const page = await context.newPage();
        await page.goto(target_url);
        await page.waitForTimeout(3000);
        
        // Get and log the actual page title
        const pageTitle = await page.title();
        console.log(`After login, the page title is: "${pageTitle}"`);

        // Verify login success using substring matching
        const isLoginSuccessful = await page.evaluate(() => {
          // Check if title contains "Home" instead of exact match
          const title = document.title;
          const isTitleCorrect = title.includes("Home");
          
          // Log what we found for debugging
          if (!isTitleCorrect) {
            console.log(`Unexpected page title: "${title}"`);
          }
          // Return true if verification method succeeds
          return isTitleCorrect;
        });

        if (!isLoginSuccessful) {
          throw new Error('Login failed: Could not verify Title after login attempt');
        }
        
        if (isLoginSuccessful) {
          console.log('Session is valid, continuing with existing session');
          hasValidSession = true;
          
          // Take screenshot of valid session
          const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
          await page.screenshot({ path: `${timestamp}-valid-session.png` });
        } else {
          console.log('Session is no longer valid, will attempt fresh login');
          await page.close();
          await context.close();
          context = null;
        }
      } catch (error) {
        console.error('Error while restoring session:', error);
        console.log('Will attempt fresh login');
        if (context) await context.close();
        context = null;
      }
    } else {
      console.log('No session data found, will perform login');
    }
    
    // If session is not valid, perform login
    if (!hasValidSession) {
      console.log('Creating new browser context for login...');
      context = await browser.newContext();
      const page = await context.newPage();
      
      console.log('Navigating to X.com homepage...');
      await page.goto(target_url);
      await page.waitForTimeout(3000);
      
      // Take initial screenshot
      let Timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      await page.screenshot({ path: `${Timestamp}-before-login.png` });
      
      console.log(`Attempting to login with username: ${username}`);
      
      // Fill username
      await page.waitForSelector('input[name="text"]', { state: 'visible' });
      await page.fill('input[name="text"]', username);
      
      // Take screenshot after putting usename
      Timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      await page.screenshot({ path: `${Timestamp}-after-username.png` });

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
      
      await page.waitForTimeout(3000);

      // Take screenshot after clicking Next
      Timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      await page.screenshot({ path: `${Timestamp}-after-next-button.png` });

      // Wait for password field
      await page.waitForSelector('input[name="password"]', { state: 'visible', timeout: 3000 });
      
      // Fill password
      await page.fill('input[name="password"]', password);

      await page.waitForTimeout(1500);

      // Take screenshot after putting password
      Timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      await page.screenshot({ path: `${Timestamp}-after-password.png` });

      // Click Login button
      await page.waitForSelector('button[data-testid="LoginForm_Login_Button"]', { state: 'visible' });
      await page.click('button[data-testid="LoginForm_Login_Button"]');
      
      await page.waitForTimeout(3000);

      // Take screenshot after Login button
      Timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      await page.screenshot({ path: `${Timestamp}-after-login-button.png` });
      
      // Wait for navigation
      await page.waitForTimeout(10000);

      // Get and log the actual page title
      const pageTitle = await page.title();
      console.log(`After login, the page title is: "${pageTitle}"`);

      // Verify login success using substring matching
      const isLoginSuccessful = await page.evaluate(() => {
        // Check if title contains "Home" instead of exact match
        const title = document.title;
        const isTitleCorrect = title.includes("Home");
        
        // Log what we found for debugging
        if (!isTitleCorrect) {
          console.log(`Unexpected page title: "${title}"`);
        }
        // Return true if verification method succeeds
        return isTitleCorrect;
      });

      if (!isLoginSuccessful) {
        throw new Error('Login failed: Could not verify Title after login attempt');
      }
      
      console.log('Login successful');
      
      // Take screenshot after successful login
      const afterLoginTimestamp = new Date().toISOString().replace(/[:.]/g, '-');
      await page.screenshot({ path: `${afterLoginTimestamp}-after-login.png` });
      
      // Save session data
      console.log('Saving session data...');
      const storageState = await context.storageState();
      fs.writeFileSync(sessionDataPath, JSON.stringify(storageState, null, 2));
      
      hasValidSession = true;
    }
    
    // Continue with a valid session
    if (hasValidSession) {
      const page = await context.newPage();
      await page.goto(target_url);
      await page.waitForTimeout(3000);
      
      console.log('Successfully loaded X.com with valid session');
      
      // Get page title
      const title = await page.title();
      console.log(`\nPage Title: ${title}\n`);
      
      // Get visible text
      const bodyText = await page.evaluate(() => document.body.innerText);
      console.log('\nVisible Text (sample):');
      console.log(bodyText.substring(0, 500) + '...');
      
      // You can add more functionality here
    }
  } catch (error) {
    console.error('An error occurred during execution:', error);
  } finally {
    console.log('Closing browser...');
    await browser.close();
  }
}

scrapeXHomepage();
