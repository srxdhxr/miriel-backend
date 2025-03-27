from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from shared.database import get_db
from shared.models import ScrapedDocument
from sqlmodel import select
import json
import re
import hashlib
from datetime import datetime
import logging
import sys
import asyncio
import time
from urllib.robotparser import RobotFileParser
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class WebsiteScraper:
    def __init__(self, domain_id, domain_url):
        self.domain_id = domain_id
        self.domain_url = domain_url
        self.visited_urls = set()
        self.domain = urlparse(domain_url).netloc
        self.scraped_content = {}
        self.queue = deque()
        
        # Load existing hashes from the database
        self.existing_hashes = self.load_existing_hashes()

    def load_existing_hashes(self):
        """Load existing URL hashes from database"""
        hash_dict = {}
        with get_db() as db:
            # Get all documents for this domain
            stmt = select(ScrapedDocument).where(ScrapedDocument.domain_id == self.domain_id)
            documents = db.exec(stmt).all()
            
            # Create URL to hash mapping
            for doc in documents:
                hash_dict[doc.url] = doc.url_hash
                
        logger.info(f"Loaded {len(hash_dict)} existing URL hashes")
        return hash_dict

    def normalize_url(self, url):
        """Normalize URL by removing query parameters and fragments."""
        return re.sub(r'[#?].*$', '', url.rstrip('/'))

    def should_scrape_url(self, url: str, content: str) -> bool:
        """Check if URL should be scraped based on content hash"""
        new_hash = self.generate_url_hash(url, content)
        existing_hash = self.existing_hashes.get(url)
        
        if existing_hash and existing_hash == new_hash:
            logger.info(f"Skipping {url} - content unchanged")
            return False
        return True

    def generate_url_hash(self, url, content):
        """Generate a hash for the URL and its content"""
        content_string = f"{url}{content}"
        return hashlib.md5(content_string.encode()).hexdigest()

    def is_valid_url(self, url):
        """Check if the URL is valid for scraping"""
        if not url:
            return False
        parsed = urlparse(url)
        if any(ext in url.lower() for ext in ['.pdf', '.jpg', '.png', '.gif', '.css', '.js', 'docx', 'doc']):
            return False
        return parsed.netloc == self.domain

    def extract_content(self, html):
        """Extract raw content only"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        unwanted_selectors = [
            'header', 'footer', 'nav', 
            '.menu', '.sidebar', '.navigation',
            '.privacy-policy', '.cookie-notice',
            '.social-media', '.advertisement',
            '.search', '.breadcrumb',
            '.modal', '.popup',
            'script', 'style', 'noscript',
            '[role="navigation"]',
            '[role="banner"]',
            '[role="complementary"]'
        ]
        
        for element in soup.select(','.join(unwanted_selectors)):
            element.decompose()

        # Get main content areas
        main_content = soup.find('main') or soup.find('article') or soup
        
        # Get raw text
        raw_text = ' '.join(main_content.get_text(separator=' ').split())
        
        return raw_text

    def extract_links(self, html, current_url):
        """Extract valid links from the HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        
        for anchor in soup.find_all('a', href=True):
            url = anchor['href']
            absolute_url = urljoin(current_url, url)
            normalized_url = self.normalize_url(absolute_url)
            if self.is_valid_url(normalized_url):
                links.add(normalized_url)
        
        return links

    def can_scrape(self, url):
        """Check if the URL is allowed by robots.txt"""
        try:
            rp = RobotFileParser()
            rp.set_url(urlparse(url)._replace(path='/robots.txt').geturl())
            rp.read()
            return rp.can_fetch('*', url)
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {str(e)}")
            # Default to allowing scraping if robots.txt check fails
            return True

    async def retry_request(self, page, url, retries=3, delay=2):
        """Retry request in case of failure"""
        for attempt in range(retries):
            try:
                # Increase timeout and use domcontentloaded instead of networkidle
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                # Wait a bit for content to load
                await asyncio.sleep(2)
                return True  # Success
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {url}: {str(e)}")
                if attempt == retries - 1:
                    logger.error(f"Failed to scrape {url} after {retries} attempts")
                    return False
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff

    async def scrape_page(self, page, url):
        """Scrape a single page"""
        if url in self.visited_urls:
            return
        
        if not self.can_scrape(url):
            logger.warning(f"Skipping {url} due to robots.txt restrictions")
            return
        
        try:
            logger.info(f"Processing URL: {url}")
            success = await self.retry_request(page, url)
            
            if not success:
                self.visited_urls.add(url)  # Mark as visited to avoid retrying
                return
                
            html = await page.content()
            
            # Extract content first
            raw_content = self.extract_content(html)
            logger.info(f"Content preview: {raw_content[:50]}...")
            
            # Check if content has changed
            if not self.should_scrape_url(url, raw_content):
                self.visited_urls.add(url)
                return
            
            # Generate hash and store content
            content_hash = self.generate_url_hash(url, raw_content)
            self.scraped_content[url] = {
                'url': url,
                'url_hash': content_hash,
                'last_scraped': datetime.now().isoformat(),
                'content_raw': raw_content,
                'domain_id': self.domain_id
            }
            logger.info(f"Stored content for {url}")
            
            # Mark as visited
            self.visited_urls.add(url)
            
            # Process all links without limit
            links = self.extract_links(html, url)
            new_links = [link for link in links if link not in self.visited_urls and link not in self.queue]
            logger.info(f"Found {len(links)} links on {url}, {len(new_links)} are new")
            
            # Add new links to the queue
            for link in links:
                if link not in self.visited_urls and link not in self.queue:
                    self.queue.append(link)
                    
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")

    async def scrape_website(self):
        async with async_playwright() as p:
            # Use chromium instead of firefox for better compatibility
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-site-isolation-trials',
                    '--no-sandbox',
                    '--disable-setuid-sandbox'
                ]
            )
            
            # Set up browser context with more permissive settings
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                ignore_https_errors=True
            )
            
            page = await context.new_page()
            
            # Start with the base URL
            self.queue.append(self.domain_url)
            
            # Track total links found and processed
            total_links_found = 0
            total_links_processed = 0
            
            # Process queue with limited concurrency
            while self.queue:
                # Log queue status
                remaining_links = len(self.queue)
                logger.info(f"Links remaining in queue: {remaining_links}")
                logger.info(f"Total links found so far: {total_links_found}")
                logger.info(f"Total links processed: {total_links_processed}")
                
                # Process up to 5 URLs at a time
                batch = []
                for _ in range(min(5, len(self.queue))):
                    if self.queue:
                        batch.append(self.queue.popleft())
                
                # Create a new page for each URL in the batch
                pages = []
                for _ in batch:
                    pages.append(await context.new_page())
                
                # Create tasks for each URL
                tasks = [self.scrape_page(pages[i], batch[i]) for i in range(len(batch))]
                
                # Run tasks concurrently
                await asyncio.gather(*tasks)
                
                # Update counters
                total_links_processed += len(batch)
                
                # Close the pages
                for p in pages:
                    await p.close()
            
            await browser.close()
            logger.info(f"Scraping completed. New/changed pages: {len(self.scraped_content)}")
            logger.info(f"Total links found: {total_links_found}")
            logger.info(f"Total links processed: {total_links_processed}")
            return self.scraped_content

    def scrape_website_sync(self):
        """Synchronous version of scrape_website"""
        with sync_playwright() as p:
            # Use chromium instead of firefox for better compatibility
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-site-isolation-trials',
                    '--no-sandbox',
                    '--disable-setuid-sandbox'
                ]
            )
            
            # Set up browser context with more permissive settings
            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                ignore_https_errors=True
            )
            
            page = context.new_page()
            
            # Track total links
            total_links_found = 0
            total_links_processed = 0
            
            def scrape_page_sync(url, depth=0, max_depth=10):
                """Synchronous version of scrape_page with depth limit"""
                nonlocal total_links_found, total_links_processed
                
                if url in self.visited_urls or depth > max_depth:
                    return
                
                total_links_processed += 1
                
                try:
                    logger.info(f"Processing URL: {url} (depth: {depth})")
                    logger.info(f"Links processed: {total_links_processed}, Links found: {total_links_found}")
                    
                    # Retry logic for sync version with better error handling
                    success = False
                    retries = 3
                    for attempt in range(retries):
                        try:
                            # Use domcontentloaded instead of networkidle for faster loading
                            page.goto(url, wait_until="domcontentloaded", timeout=60000)
                            # Wait a bit for content to load
                            time.sleep(2)
                            success = True
                            break
                        except Exception as e:
                            logger.warning(f"Attempt {attempt+1} failed for {url}: {str(e)}")
                            if attempt == retries - 1:
                                logger.error(f"Failed to scrape {url} after {retries} attempts")
                                return
                            time.sleep(2 * (2 ** attempt))  # Exponential backoff
                    
                    if not success:
                        self.visited_urls.add(url)  # Mark as visited to avoid retrying
                        return
                        
                    html = page.content()
                    
                    # Extract content
                    raw_content = self.extract_content(html)
                    logger.info(f"Content preview: {raw_content[:50]}...")
                    
                    # Check if content has changed
                    if not self.should_scrape_url(url, raw_content):
                        self.visited_urls.add(url)
                        
                        # Still process links from unchanged pages if not too deep
                        if depth < max_depth:
                            links = self.extract_links(html, url)
                            logger.info(f"Found {len(links)} links on {url} (unchanged content)")
                            for link in links:
                                if link not in self.visited_urls:
                                    scrape_page_sync(link, depth + 1, max_depth)
                        return
                    
                    # Generate hash and store content
                    content_hash = self.generate_url_hash(url, raw_content)
                    self.scraped_content[url] = {
                        'url': url,
                        'url_hash': content_hash,
                        'last_scraped': datetime.now().isoformat(),
                        'content_raw': raw_content,
                        'domain_id': self.domain_id
                    }
                    logger.info(f"Stored content for {url}")
                    
                    # Mark as visited
                    self.visited_urls.add(url)
                    
                    # Process links if not too deep
                    if depth < max_depth:
                        links = self.extract_links(html, url)
                        logger.info(f"Found {len(links)} links on {url}")
                        new_links = [link for link in links if link not in self.visited_urls]
                        total_links_found += len(new_links)
                        logger.info(f"Found {len(links)} links on {url}, {len(new_links)} are new")
                        for link in links:
                            if link not in self.visited_urls:
                                scrape_page_sync(link, depth + 1, max_depth)
                            
                except Exception as e:
                    logger.error(f"Error scraping {url}: {str(e)}")
            
            # Start scraping from the base URL
            logger.info(f"Starting sync scrape from base URL: {self.domain_url}")
            scrape_page_sync(self.domain_url)
            browser.close()
            
            logger.info(f"Sync scraping completed. New/changed pages: {len(self.scraped_content)}")
            logger.info(f"Total links found: {total_links_found}")
            logger.info(f"Total links processed: {total_links_processed}")
            return self.scraped_content
