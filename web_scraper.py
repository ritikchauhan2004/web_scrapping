import os
import sys
import json
import time
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse
import base64
from io import BytesIO

# Third-party imports
try:
    import requests
    from bs4 import BeautifulSoup
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    from dotenv import load_dotenv
    import google.generativeai as genai
    from PIL import Image
    import pytesseract
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Please install: pip install playwright beautifulsoup4 requests python-dotenv google-generativeai pytesseract Pillow lxml")
    sys.exit(1)


# CONFIGURATION SECTION

class Config:
    """Configuration manager - loads settings from .env file"""
    
    def __init__(self):
        load_dotenv()
        
        # Gemini API Configuration
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        self.gemini_model = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')
        
        # Browser Configuration
        self.headless_mode = os.getenv('HEADLESS_MODE', 'true').lower() == 'true'
        self.browser_timeout = int(os.getenv('BROWSER_TIMEOUT', '30000'))
        self.viewport_width = int(os.getenv('VIEWPORT_WIDTH', '1920'))
        self.viewport_height = int(os.getenv('VIEWPORT_HEIGHT', '1080'))
        
        # OCR Configuration
        self.ocr_language = os.getenv('OCR_LANGUAGE', 'eng')
        self.ocr_dpi = int(os.getenv('OCR_DPI', '300'))
        
        # Output Configuration
        self.output_folder = os.getenv('OUTPUT_FOLDER', 'output')
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.retry_delay = int(os.getenv('RETRY_DELAY', '2'))
        
        # URLs file
        self.urls_file = os.getenv('URLS_FILE', 'urls.txt')
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate critical configuration"""
        if not self.gemini_api_key:
            logging.warning("GEMINI_API_KEY not set in .env file")
        
        # Create output folder if it doesn't exist
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)


# LOGGING SETUP

def setup_logging(log_level: str) -> logging.Logger:
    """Configure logging with both file and console output"""
    
    logger = logging.getLogger('WebScraper')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    log_file = f'scraper_{datetime.now().strftime("%Y%m%d")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# PAGE TYPE DETECTOR

class PageTypeDetector:
    """Detects the type of web page and appropriate extraction strategy"""
    
    @staticmethod
    def detect(url: str, html_content: str, logger: logging.Logger) -> str:
        """
        Detect page type
        Returns: 'STATIC', 'DYNAMIC', 'IMAGE_BASED', or 'HYBRID'
        """
        
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Check for heavy JavaScript frameworks
        js_frameworks = [
            'react', 'vue', 'angular', 'next.js', 'nuxt',
            'backbone', 'ember', '__NEXT_DATA__', 'ng-app'
        ]
        has_js_framework = any(
            framework in html_content.lower() 
            for framework in js_frameworks
        )
        
        # Check for dynamic loading indicators
        dynamic_indicators = [
            'data-react', 'data-vue', 'ng-', 'v-',
            '__NUXT__', 'webpack', 'hydrate'
        ]
        has_dynamic_loading = any(
            indicator in html_content 
            for indicator in dynamic_indicators
        )
        
        # Check for canvas/chart elements
        has_canvas = bool(soup.find_all('canvas'))
        has_svg_charts = bool(soup.find_all('svg', class_=re.compile(r'chart|graph')))
        
        # Check for image-heavy content
        images = soup.find_all('img')
        has_many_images = len(images) > 10
        
        # Check for complex tables
        tables = soup.find_all('table')
        has_complex_tables = any(
            table.find_all('table') or  # Nested tables
            len(table.find_all('th', colspan=True)) > 0 or  # Merged cells
            len(table.find_all('tr')) > 50  # Large tables
            for table in tables
        )
        
        # Decision logic
        if (has_canvas or has_svg_charts) and has_many_images:
            return 'IMAGE_BASED'
        elif has_js_framework or has_dynamic_loading:
            return 'DYNAMIC'
        elif has_complex_tables or len(tables) > 0:
            return 'STATIC'
        elif has_many_images:
            return 'IMAGE_BASED'
        else:
            return 'HYBRID'


# STATIC EXTRACTOR (BeautifulSoup)

class StaticExtractor:
    """Extract data from static HTML pages"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def extract(self, url: str) -> Dict[str, Any]:
        """Extract data from static HTML"""
        
        self.logger.info(f"Using Static Extractor for: {url}")
        
        try:
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            data = {
                'url': url,
                'title': self._extract_title(soup),
                'tables': self._extract_tables(soup),
                'text_content': self._extract_text(soup),
                'metadata': self._extract_metadata(soup)
            }
            
            self.logger.info(f"Extracted {len(data['tables'])} tables and text content")
            return data
            
        except Exception as e:
            self.logger.error(f"Static extraction failed: {e}")
            return {'url': url, 'error': str(e)}
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        h1_tag = soup.find('h1')
        return h1_tag.get_text(strip=True) if h1_tag else 'Untitled'
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract all tables from HTML"""
        
        tables_data = []
        tables = soup.find_all('table')
        
        for idx, table in enumerate(tables):
            try:
                # Extract headers
                headers = []
                header_row = table.find('thead')
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                else:
                    first_row = table.find('tr')
                    if first_row:
                        headers = [th.get_text(strip=True) for th in first_row.find_all(['th', 'td'])]
                
                # Extract rows
                rows = []
                tbody = table.find('tbody') or table
                for tr in tbody.find_all('tr')[1 if not table.find('thead') else 0:]:
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells:
                        if headers and len(cells) == len(headers):
                            rows.append(dict(zip(headers, cells)))
                        else:
                            rows.append({'data': cells})
                
                if rows:
                    tables_data.append({
                        'table_id': f'table_{idx + 1}',
                        'headers': headers,
                        'rows': rows,
                        'row_count': len(rows),
                        'column_count': len(headers) if headers else len(rows[0]) if rows else 0
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error extracting table {idx}: {e}")
        
        return tables_data
    
    def _extract_text(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract meaningful text content"""
        
        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Extract headings
        headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3'])]
        
        # Extract paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 50]
        
        return {
            'headings': headings[:10],
            'paragraphs': paragraphs[:5],
            'full_text_length': len(text)
        }
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract metadata from page"""
        
        metadata = {}
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property', '')
            content = meta.get('content', '')
            if name and content:
                metadata[name] = content
        
        return metadata



# DYNAMIC EXTRACTOR (Playwright)

class DynamicExtractor:
    """Extract data from JavaScript-rendered pages"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def extract(self, url: str, screenshot_filename: Optional[str] = None) -> Dict[str, Any]:
        """Extract data using headless browser"""
        
        self.logger.info(f"Using Dynamic Extractor (Playwright) for: {url}")
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.config.headless_mode)
                context = browser.new_context(
                    viewport={
                        'width': self.config.viewport_width,
                        'height': self.config.viewport_height
                    }
                )
                page = context.new_page()
                
                # Navigate to page with retry and fallback strategies
                page_loaded = False
                html_content = ''
                
                # Try multiple loading strategies
                for strategy in ['networkidle', 'load', 'domcontentloaded']:
                    try:
                        self.logger.info(f"Loading page with strategy: {strategy}")
                        page.goto(url, wait_until=strategy, timeout=self.config.browser_timeout)
                        page_loaded = True
                        break
                    except PlaywrightTimeout:
                        self.logger.warning(f"Timeout with {strategy} strategy, trying next...")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error with {strategy} strategy: {e}")
                        continue
                
                # If all strategies fail, try without waiting
                if not page_loaded:
                    try:
                        self.logger.warning("All wait strategies failed, loading without wait...")
                        page.goto(url, timeout=self.config.browser_timeout)
                    except Exception as e:
                        self.logger.error(f"Failed to load page: {e}")
                        # Continue anyway to capture whatever is loaded
                
                # Wait for content to load
                page.wait_for_timeout(5000)
                
                # Click on interactive elements to expand data
                self._click_interactive_elements(page)
                
                # Scroll to load lazy content
                self._scroll_page(page)
                
                # Get rendered HTML
                try:
                    html_content = page.content()
                except Exception as e:
                    self.logger.warning(f"Error getting page content: {e}")
                    html_content = '<html><body></body></html>'
                
                # Take screenshot for image extraction with proper naming - ALWAYS capture
                if not screenshot_filename:
                    screenshot_filename = f"temp_screenshot_{int(time.time())}.png"
                screenshot_path = os.path.join(self.config.output_folder, screenshot_filename)
                
                try:
                    page.screenshot(path=screenshot_path, full_page=True)
                    self.logger.info(f"Screenshot saved: {screenshot_filename}")
                except Exception as e:
                    self.logger.warning(f"Full page screenshot failed, trying viewport screenshot: {e}")
                    try:
                        page.screenshot(path=screenshot_path)
                        self.logger.info(f"Viewport screenshot saved: {screenshot_filename}")
                    except Exception as e2:
                        self.logger.error(f"Screenshot capture failed: {e2}")
                        screenshot_path = None
                
                browser.close()
                
                # Parse rendered HTML
                soup = BeautifulSoup(html_content, 'lxml')
                
                data = {
                    'url': url,
                    'title': self._extract_title(soup),
                    'tables': self._extract_tables(soup),
                    'dynamic_content': self._extract_dynamic_content(soup),
                    'screenshot': screenshot_path,
                    'screenshot_filename': screenshot_filename
                }
                
                self.logger.info(f"Extracted {len(data['tables'])} tables from dynamic content")
                return data
                
        except PlaywrightTimeout:
            self.logger.error(f"Timeout loading page: {url}")
            return {'url': url, 'error': 'Timeout'}
        except Exception as e:
            self.logger.error(f"Dynamic extraction failed: {e}")
            return {'url': url, 'error': str(e)}
    
    def _click_interactive_elements(self, page):
        """Click on interactive elements to expand data"""
        try:
            # Common selectors for expandable elements
            selectors = [
                'button:has-text("Show more")',
                'button:has-text("Load more")',
                'button:has-text("View all")',
                '[role="tab"]',
                '.tab',
                '[data-tab]',
                'button[aria-expanded="false"]',
                '.dropdown',
                '.expand',
                '[role="button"]'
            ]
            
            for selector in selectors:
                try:
                    elements = page.query_selector_all(selector)
                    for element in elements[:5]:  # Click up to 5 of each type
                        try:
                            element.click(timeout=2000)
                            page.wait_for_timeout(1000)
                        except:
                            continue
                except:
                    continue
                    
            self.logger.info("Clicked interactive elements for data expansion")
        except Exception as e:
            self.logger.debug(f"Error clicking interactive elements: {e}")
    
    def _scroll_page(self, page):
        """Scroll page to trigger lazy loading"""
        try:
            for _ in range(5):  # Increased scroll iterations
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1500)
        except:
            pass
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title = soup.find('title')
        return title.get_text(strip=True) if title else 'Untitled'
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract tables from rendered HTML with better structure"""
        tables_data = []
        tables = soup.find_all('table')
        
        for idx, table in enumerate(tables):
            try:
                rows = []
                headers = []
                
                # Try to find headers
                for th in table.find_all('th'):
                    headers.append(th.get_text(strip=True))
                
                # Extract all rows
                for tr in table.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells:
                        rows.append(cells)
                
                if rows:
                    # Clean and structure the data
                    cleaned_rows = self._clean_table_data(headers if headers else rows[0], 
                                                         rows[1:] if headers else rows)
                    
                    tables_data.append({
                        'table_id': f'dynamic_table_{idx + 1}',
                        'headers': headers if headers else rows[0] if rows else [],
                        'rows': rows[1:] if headers else rows,  # Keep raw data
                        'structured_rows': cleaned_rows,  # Add cleaned/structured data
                        'row_count': len(cleaned_rows),
                        'column_count': len(headers) if headers else (len(rows[0]) if rows else 0)
                    })
            except Exception as e:
                self.logger.warning(f"Error extracting dynamic table {idx}: {e}")
        
        return tables_data
    
    def _clean_table_data(self, headers: List[str], rows: List[List[str]]) -> List[Dict[str, Any]]:
        """Clean and structure table data for better LLM consumption"""
        import re
        
        cleaned_data = []
        
        for row in rows:
            # Skip rows that are mostly empty
            non_empty_cells = [cell for cell in row if cell.strip()]
            if len(non_empty_cells) < len(headers) // 2:
                continue
            
            # Create structured row object
            row_dict = {}
            
            for i, header in enumerate(headers):
                if i >= len(row):
                    continue
                    
                cell_value = row[i].strip()
                
                # Parse based on column name
                if 'Name' in header or 'name' in header.lower():
                    # Extract cryptocurrency name and symbol
                    match = re.search(r'^([A-Za-z\s]+?)([A-Z]{2,10})(?:Buy)?$', cell_value)
                    if match:
                        row_dict['name'] = match.group(1).strip()
                        row_dict['symbol'] = match.group(2)
                    else:
                        row_dict['name'] = cell_value
                        
                elif 'Price' in header or 'price' in header.lower():
                    row_dict['price'] = cell_value
                    
                elif 'Market Cap' in header or 'market' in header.lower():
                    match = re.search(r'(\$[\d.]+[KMBT]?)', cell_value)
                    if match:
                        row_dict['market_cap'] = match.group(1)
                    else:
                        row_dict['market_cap'] = cell_value
                        
                elif 'Volume' in header or 'volume' in header.lower():
                    # Extract readable volume
                    match = re.search(r'(\$[\d.]+[KMBT]?)', cell_value)
                    if match:
                        row_dict['volume_24h'] = match.group(1)
                    else:
                        row_dict['volume_24h'] = cell_value
                        
                elif 'Supply' in header or 'supply' in header.lower():
                    match = re.search(r'([\d.]+[KMBT]?)([A-Z]+)?', cell_value)
                    if match:
                        row_dict['circulating_supply'] = match.group(1)
                        if match.group(2):
                            row_dict['supply_symbol'] = match.group(2)
                    else:
                        row_dict['circulating_supply'] = cell_value
                        
                elif '#' in header or 'rank' in header.lower():
                    row_dict['rank'] = cell_value
                    
                elif '%' in header:
                    # Handle percentage columns (1h %, 24h %, 7d %)
                    col_name = header.lower().replace(' ', '_').replace('%', 'percent')
                    row_dict[col_name] = cell_value
                    
                else:
                    # Generic column
                    col_name = header.lower().replace(' ', '_').replace('(', '').replace(')', '')
                    row_dict[col_name] = cell_value
            
            if row_dict:  # Only add if we extracted something
                cleaned_data.append(row_dict)
        
        return cleaned_data
    
    def _extract_dynamic_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract dynamically loaded content"""
        
        content = {
            'lists': [],
            'cards': [],
            'dynamic_elements': []
        }
        
        # Extract common dynamic patterns
        for ul in soup.find_all(['ul', 'ol'], class_=re.compile(r'list|items')):
            items = [li.get_text(strip=True) for li in ul.find_all('li')[:20]]
            if items:
                content['lists'].append(items)
        
        # Extract card-like elements
        for card in soup.find_all(class_=re.compile(r'card|item|entry')):
            card_text = card.get_text(strip=True)
            if len(card_text) > 20 and len(card_text) < 500:
                content['cards'].append(card_text)
        
        return content


# IMAGE/CHART EXTRACTOR (OCR + Vision)

class ImageExtractor:
    """Extract data from images and charts using OCR and Vision AI"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        if config.gemini_api_key:
            genai.configure(api_key=config.gemini_api_key)
            self.vision_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        else:
            self.vision_model = None
    
    def extract(self, url: str, screenshot_path: Optional[str] = None) -> Dict[str, Any]:
        """Extract data from images/charts with comprehensive OCR"""
        
        self.logger.info(f"Using Image Extractor with OCR for: {url}")
        
        data = {
            'url': url,
            'images_and_charts': [],
            'screenshot_ocr': '',
            'vision_analysis': {},
            'total_images_processed': 0
        }
        
        try:
            # Get page content
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Find all images, charts, graphs, SVGs
            images = soup.find_all('img')
            canvases = soup.find_all('canvas')
            svgs = soup.find_all('svg')
            
            self.logger.info(f"Found {len(images)} images, {len(canvases)} canvases, {len(svgs)} SVGs")
            
            # Process all images with OCR
            for idx, img in enumerate(images[:20]):  # Process more images
                img_url = img.get('src', '')
                if not img_url:
                    continue
                
                # Make absolute URL
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                elif img_url.startswith('/'):
                    parsed = urlparse(url)
                    img_url = f"{parsed.scheme}://{parsed.netloc}{img_url}"
                
                try:
                    # Download image
                    img_response = requests.get(img_url, timeout=10)
                    image = Image.open(BytesIO(img_response.content))
                    
                    # Get image info
                    img_alt = img.get('alt', '')
                    img_title = img.get('title', '')
                    img_class = ' '.join(img.get('class', []))
                    
                    # Enhanced OCR extraction with preprocessing
                    ocr_text = pytesseract.image_to_string(
                        image,
                        lang=self.config.ocr_language,
                        config='--psm 6'  # Assume uniform block of text
                    )
                    
                    # Try OCR with data extraction for tables/charts
                    ocr_data = pytesseract.image_to_data(
                        image,
                        lang=self.config.ocr_language,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    image_data = {
                        'image_id': f'image_{idx + 1}',
                        'source_url': img_url,
                        'alt_text': img_alt,
                        'title': img_title,
                        'css_class': img_class,
                        'ocr_text': ocr_text.strip(),
                        'ocr_confidence': sum(ocr_data['conf']) / len(ocr_data['conf']) if ocr_data['conf'] else 0,
                        'image_size': f"{image.width}x{image.height}",
                        'type': self._detect_image_type(img_alt, img_title, img_class, ocr_text)
                    }
                    
                    # Vision AI analysis (if available)
                    if self.vision_model:
                        vision_data = self._analyze_with_vision(image)
                        if vision_data:
                            image_data['vision_analysis'] = vision_data
                    
                    data['images_and_charts'].append(image_data)
                    data['total_images_processed'] += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error processing image {idx}: {e}")
            
            if screenshot_path and os.path.exists(screenshot_path):
                try:
                    self.logger.info("Performing OCR on full page screenshot...")
                    image = Image.open(screenshot_path)
                    
                    # Full page OCR
                    ocr_text = pytesseract.image_to_string(
                        image,
                        lang=self.config.ocr_language
                    )
                    data['screenshot_ocr'] = ocr_text.strip()
                    
                    # Vision AI analysis of screenshot
                    if self.vision_model:
                        data['vision_analysis']['full_page'] = self._analyze_with_vision(image)
                    
                    # Don't delete screenshot - keep it with proper naming
                    self.logger.info(f"Screenshot saved: {screenshot_path}")
                    
                except Exception as e:
                    self.logger.warning(f"Error processing screenshot: {e}")
            
            self.logger.info(f"Extracted {len(data['images_and_charts'])} images/charts with OCR")
            return data
            
        except Exception as e:
            self.logger.error(f"Image extraction failed: {e}")
            return {'url': url, 'error': str(e)}
    
    def _detect_image_type(self, alt_text: str, title: str, css_class: str, ocr_text: str) -> str:
        """Detect if image is a chart, graph, diagram, or regular image"""
        
        indicators = (alt_text + ' ' + title + ' ' + css_class + ' ' + ocr_text).lower()
        
        if any(word in indicators for word in ['chart', 'graph', 'plot', 'diagram']):
            return 'chart/graph'
        elif any(word in indicators for word in ['table', 'data', 'statistics']):
            return 'data_table'
        elif any(word in indicators for word in ['map', 'location']):
            return 'map'
        else:
            return 'general_image'
    
    def _analyze_with_vision(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image using Gemini Vision"""
        
        try:
            # Convert image to bytes
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create prompt
            prompt = """Analyze this image and extract any data, tables, charts, or statistics.
            Provide a structured description of what you see.
            If there are tables or charts, extract the data in a structured format.
            Focus on numerical data, trends, and key information."""
            
            response = self.vision_model.generate_content([prompt, image])
            
            return {
                'description': response.text,
                'extracted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.warning(f"Vision API analysis failed: {e}")
            return {}


# DATA NORMALIZER (LLM)

class DataNormalizer:
    """Normalize and structure extracted data using LLM"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        if config.gemini_api_key:
            genai.configure(api_key=config.gemini_api_key)
            self.model = genai.GenerativeModel(config.gemini_model)
        else:
            self.model = None
            self.logger.warning("Gemini API key not configured - skipping LLM normalization")
    
    def normalize(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw extracted data into structured JSON"""
        
        if not self.model:
            return self._basic_normalization(raw_data)
        
        self.logger.info("Normalizing data with Gemini LLM")
        
        try:
            # Prepare prompt
            prompt = f"""You are a data structuring expert. Analyze the following extracted web data and create a clean, structured JSON output.

Raw Data:
{json.dumps(raw_data, indent=2)}

Instructions:
1. Clean and organize all tables into a consistent format
2. Extract key insights and patterns
3. Identify the main data types (e.g., statistics, time series, rankings)
4. Remove duplicates and inconsistencies
5. Add meaningful metadata
6. Provide a brief interpretation of the data's purpose and use cases

Return ONLY valid JSON in this format:
{{
  "metadata": {{
    "source_url": "...",
    "data_type": "...",
    "extraction_date": "...",
    "quality_score": 0.0-1.0
  }},
  "structured_data": {{
    // Organized data here
  }},
  "analysis": {{
    "summary": "...",
    "key_findings": [],
    "recommended_uses": []
  }}
}}"""
            
            response = self.model.generate_content(prompt)
            
            # Parse JSON from response
            response_text = response.text
            
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            normalized_data = json.loads(response_text)
            
            self.logger.info("Data normalization completed")
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"LLM normalization failed: {e}")
            return self._basic_normalization(raw_data)
    
    def _basic_normalization(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic normalization without LLM"""
        
        return {
            'metadata': {
                'source_url': raw_data.get('url', ''),
                'extraction_timestamp': datetime.now().isoformat(),
                'page_type': 'unknown',
                'data_quality_score': 0.7
            },
            'structured_data': raw_data,
            'analysis': {
                'summary': 'Data extracted without LLM normalization',
                'key_findings': [],
                'recommended_uses': ['Further analysis required']
            }
        }


# MAIN ORCHESTRATOR

class WebScraper:
    """Main orchestrator for web data extraction"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Initialize extractors
        self.static_extractor = StaticExtractor(logger)
        self.dynamic_extractor = DynamicExtractor(config, logger)
        self.image_extractor = ImageExtractor(config, logger)
        self.normalizer = DataNormalizer(config, logger)
    
    def process_url(self, url: str, url_index: int) -> Dict[str, Any]:
        """Process a single URL with appropriate extraction strategy and retry logic"""
        
        self.logger.info(f"Processing URL: {url}")
        
        start_time = time.time()
        
        # Generate filenames based on URL index
        screenshot_filename = f"url{url_index}_screenshot.png"
        json_filename = f"url{url_index}.json"
        
        # Retry logic with exponential backoff
        max_retries = self.config.max_retries
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                # Initial page load for detection
                response = requests.get(url, timeout=30, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                html_content = response.text
                break  # Success, exit retry loop
            except Exception as e:
                retry_count += 1
                last_error = e
                if retry_count <= max_retries:
                    wait_time = self.config.retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                    self.logger.warning(f"Initial request failed (attempt {retry_count}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"All retry attempts failed for initial request: {e}")
                    html_content = '<html><body></body></html>'  # Use empty HTML for detection
        
        try:
            
            # Detect page type
            page_type = PageTypeDetector.detect(url, html_content, self.logger)
            self.logger.info(f"Detected page type: {page_type}")
            
            # Route to appropriate extractor
            raw_data = {}
            screenshot_path = None
            
            if page_type == 'DYNAMIC':
                raw_data = self.dynamic_extractor.extract(url, screenshot_filename)
                screenshot_path = raw_data.get('screenshot')
            elif page_type == 'STATIC':
                raw_data = self.static_extractor.extract(url)
                # Still take screenshot for OCR
                dynamic_data = self.dynamic_extractor.extract(url, screenshot_filename)
                raw_data['screenshot'] = dynamic_data.get('screenshot')
                screenshot_path = dynamic_data.get('screenshot')
            elif page_type == 'IMAGE_BASED':
                # Use dynamic extractor for screenshot, then image extraction
                dynamic_data = self.dynamic_extractor.extract(url, screenshot_filename)
                screenshot_path = dynamic_data.get('screenshot')
                image_data = self.image_extractor.extract(url, screenshot_path)
                raw_data = {**dynamic_data, **image_data}
            else:  # HYBRID
                # Combine multiple strategies
                static_data = self.static_extractor.extract(url)
                dynamic_data = self.dynamic_extractor.extract(url, screenshot_filename)
                screenshot_path = dynamic_data.get('screenshot')
                image_data = self.image_extractor.extract(url, screenshot_path)
                
                raw_data = {
                    'static': static_data,
                    'dynamic': dynamic_data,
                    'images': image_data
                }
            
            # Add processing metadata
            raw_data['processing_metadata'] = {
                'page_type': page_type,
                'extraction_timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(time.time() - start_time, 2)
            }
            
            # Normalize data with LLM
            normalized_data = self.normalizer.normalize(raw_data)
            
            # Add final metadata
            normalized_data['extraction_metadata'] = {
                'url': url,
                'page_type': page_type,
                'extraction_method': self._get_extraction_method(page_type),
                'total_processing_time': round(time.time() - start_time, 2),
                'timestamp': datetime.now().isoformat(),
                'output_files': {
                    'json': json_filename,
                    'screenshot': screenshot_filename if screenshot_path else None
                }
            }
            
            self.logger.info(f"✓ Successfully processed {url} in {normalized_data['extraction_metadata']['total_processing_time']}s")
            
            return normalized_data, json_filename
            
        except Exception as e:
            self.logger.error(f"✗ Failed to process {url}: {e}")
            return {
                'url': url,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, json_filename
    
    def _get_extraction_method(self, page_type: str) -> str:
        """Get extraction method description"""
        methods = {
            'STATIC': 'BeautifulSoup',
            'DYNAMIC': 'Playwright + BeautifulSoup',
            'IMAGE_BASED': 'Playwright + OCR + Vision AI',
            'HYBRID': 'Multi-strategy (Static + Dynamic + Image)'
        }
        return methods.get(page_type, 'Unknown')
    
    def save_output(self, data: Dict[str, Any], filename: str):
        """Save extracted data to JSON file with specified filename"""
        
        output_path = os.path.join(self.config.output_folder, filename)
        
        # Ensure screenshot data is in structured_data for LLM queries
        if 'structured_data' in data and 'screenshot' not in data['structured_data']:
            # Check if screenshot exists in raw_data
            if 'raw_data' in data:
                screenshot = data['raw_data'].get('screenshot')
                screenshot_filename = data['raw_data'].get('screenshot_filename')
                if screenshot:
                    data['structured_data']['screenshot'] = screenshot
                if screenshot_filename:
                    data['structured_data']['screenshot_filename'] = screenshot_filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✓ Saved output to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"✗ Failed to save output: {e}")
            return None
    
    def run(self):
        """Main execution flow"""
        
        self.logger.info("WEB DATA EXTRACTION SYSTEM")
        
        # Load URLs
        urls = self._load_urls()
        
        if not urls:
            self.logger.error("No URLs found to process")
            return
        
        self.logger.info(f"Loaded {len(urls)} URLs from {self.config.urls_file}")
        
        # Process each URL
        results = []
        for idx, url in enumerate(urls, 1):
            self.logger.info(f"\n[{idx}/{len(urls)}] Processing URL: {url}")
            
            # Process URL with index for naming
            result_data = self.process_url(url, idx)
            
            # Handle tuple return (data, filename) or dict (error case)
            if isinstance(result_data, tuple):
                result, json_filename = result_data
                # Save output
                output_path = self.save_output(result, json_filename)
                result['output_file'] = output_path
            else:
                result = result_data
            
            results.append(result)
            
            # Small delay between requests
            if idx < len(urls):
                time.sleep(self.config.retry_delay)
        
        # Generate summary report
        self._generate_report(results)
        
        self.logger.info("EXTRACTION COMPLETED")
    
    def _load_urls(self) -> List[str]:
        """Load URLs from file"""
        
        urls = []
        
        if not os.path.exists(self.config.urls_file):
            self.logger.error(f"URLs file not found: {self.config.urls_file}")
            return urls
        
        try:
            with open(self.config.urls_file, 'r') as f:
                for line in f:
                    url = line.strip()
                    if url and not url.startswith('#'):
                        urls.append(url)
        except Exception as e:
            self.logger.error(f"Error reading URLs file: {e}")
        
        return urls
    
    def _generate_report(self, results: List[Dict[str, Any]]):
        """Generate summary report"""
        
        total = len(results)
        successful = sum(1 for r in results if 'error' not in r)
        failed = total - successful
        
        self.logger.info("EXTRACTION REPORT")
        self.logger.info(f"Total URLs processed: {total}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        
        if successful > 0:
            avg_time = sum(
                r.get('extraction_metadata', {}).get('total_processing_time', 0)
                for r in results if 'error' not in r
            ) / successful
            self.logger.info(f"Average processing time: {avg_time:.2f}s")
        
        # Save report
        report_path = os.path.join(
            self.config.output_folder,
            f"extraction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        try:
            with open(report_path, 'w') as f:
                json.dump({
                    'summary': {
                        'total': total,
                        'successful': successful,
                        'failed': failed,
                        'timestamp': datetime.now().isoformat()
                    },
                    'results': results
                }, f, indent=2)
            
            self.logger.info(f"Report saved to: {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")


# MAIN ENTRY POINT

def main():
    """Main entry point"""
    
    
    # Load configuration
    config = Config()
    
    # Setup logging
    logger = setup_logging(config.log_level)
    
    # Create scraper
    scraper = WebScraper(config, logger)
    
    try:
        # Run extraction
        scraper.run()
        
    except KeyboardInterrupt:
        logger.info("\n\nExtraction interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()