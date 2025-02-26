const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const target_url = "https://x.com/home"
async function scrapeXHomepage() {
  console.log('Launching browser...');
  
  const browser = await chromium.launch({
    headless: true
  });

  try {
    console.log('Opening new page...');
    const context = await browser.newContext();
    const page = await context.newPage();
    
    console.log('Navigating to X.com homepage...');
    await page.goto(target_url, { waitUntil: 'networkidle' });
    
    // Take initial screenshot after page loads but before clicking login
    const initialTimestamp = Math.floor(Date.now() / 1000);
    const initialScreenshotPath = `${initialTimestamp}-session.png`;
    console.log(`Taking initial screenshot and saving as: ${initialScreenshotPath}`);
    await page.screenshot({ path: initialScreenshotPath });
    
    // Fill the username field
    console.log('Filling username field with "username ####### ...');
    await page.waitForSelector('input[name="text"]', { state: 'visible' });
    await page.fill('input[name="text"]', 'teste123');
    
    // Wait 3 seconds for any reactions to the input
    console.log('Waiting for 3 seconds...');
    await page.waitForTimeout(3000);
    
    // Click the Next button
    console.log('Clicking the Next button...');
    await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('button[type="button"]'));
      const nextButton = buttons.find(button => button.textContent.includes('Next'));
      if (nextButton) {
        nextButton.click();
      } else {
        console.log('Next button not found');
      }
    });
    // HERE onwards da pau
    // Wait for navigation or UI changes after clicking Next
    console.log('Waiting after clicking Next...');
    await page.waitForTimeout(1000);
    
    // Take a screenshot after clicking Next
    const afterNextTimestamp = Math.floor(Date.now() / 1000);
    const afterNextScreenshotPath = `${afterNextTimestamp}-after-next-button.png`;
    console.log(`Taking screenshot after clicking Next and saving as: ${afterNextScreenshotPath}`);
    await page.screenshot({ path: afterNextScreenshotPath });
  
    // Fill the password field
    console.log('Filling password field...');
    await page.waitForSelector('input[name="password"]', { state: 'visible' });
    await page.fill('input[name="password"]', '1234567890');
    
    // Click the Login button
    console.log('Clicking the Login button...');
    await page.waitForSelector('button[data-testid="LoginForm_Login_Button"]', { state: 'visible' });
    await page.click('button[data-testid="LoginForm_Login_Button"]');
    
    // Wait 5 seconds after login
    console.log('Waiting 5 seconds after login...');
    await page.waitForTimeout(5000);
    
    // Take a screenshot after login
    const afterLoginTimestamp = Math.floor(Date.now() / 1000);
    const afterLoginScreenshotPath = `${afterLoginTimestamp}-after-login.png`;
    console.log(`Taking screenshot after login and saving as: ${afterLoginScreenshotPath}`);
    await page.screenshot({ path: afterLoginScreenshotPath });

    // Continue with the rest of the functionality
    console.log('Extracting page content...');
    
    // Get page title
    const title = await page.title();
    console.log(`\nPage Title: ${title}\n`);
    
    // Get visible text
    const bodyText = await page.evaluate(() => document.body.innerText);
    console.log('\nVisible Text:');
    console.log(bodyText);
    
  } catch (error) {
    console.error('An error occurred during scraping:', error);
  } finally {
    console.log('Closing browser...');
    await browser.close();
  }
}

scrapeXHomepage();
