import asyncio
from playwright.async_api import async_playwright

async def take_youtube_screenshot(video_url, output_filename="youtube_screenshot.png"):
    async with async_playwright() as p:
        # Launch the browser in headless mode
        browser = await p.chromium.launch(headless=True)
        # Create a new browser context
        context = await browser.new_context()
        # Open a new page
        page = await context.new_page()
        
        # Go to the YouTube video URL
        await page.goto(video_url)
        
        # Wait for a few seconds to ensure the page fully loads
        await page.wait_for_timeout(5000)  # 5 seconds

        # Take a screenshot and save to file
        await page.screenshot(path=output_filename, full_page=True)
        print(f"Screenshot saved as {output_filename}")

        # Close the browser
        await browser.close()

def main():
    youtube_video_url = "https://www.youtube.com/watch?v=aQXvtrNI_yw"
    output_file = "my_youtube_screenshot.png"
    
    asyncio.run(take_youtube_screenshot(youtube_video_url, output_file))

if __name__ == "__main__":
    main()
