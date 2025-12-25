# Web Data Extraction System

A sophisticated, AI-powered web scraping system that intelligently extracts structured data from any website using adaptive extraction strategies, OCR, and LLM-based data normalization.

## Features

- **Intelligent Page Detection** - Automatically detects page type (Static, Dynamic, Image-based, Hybrid)
- **Multi-Strategy Extraction** - Uses BeautifulSoup, Playwright, OCR, and Vision AI based on content type
- **LLM Data Normalization** - Structures raw data using Google Gemini AI
- **Interactive Chatbot** - Query extracted data using natural language via OpenRouter API
- **Robust Error Handling** - Retry logic with exponential backoff
- **Comprehensive Logging** - Detailed logs for debugging and monitoring
- **Screenshot Capture** - Full-page screenshots for visual reference and OCR

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Algorithmic Approach](#algorithmic-approach)
7. [Output Structure](#output-structure)
8. [Troubleshooting](#troubleshooting)
9. [API Credits](#api-credits)

---

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    Web Scraper System                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐      ┌──────────────────────────┐      │
│  │   Config    │──────│  PageTypeDetector        │      │
│  │  Manager    │      │  (Analyzes HTML/JS)      │      │ 
│  └─────────────┘      └──────────────────────────┘      │ 
│         │                       │                       │
│         ├───────────────────────┴──────────┐            │
│         ▼                                   ▼           │
│  ┌─────────────────┐            ┌──────────────────┐    │
│  │ Static Extractor│            │ Dynamic Extractor│    │
│  │ (BeautifulSoup) │            │  (Playwright)    │    │ 
│  └─────────────────┘            └──────────────────┘    │
│         │                                   │           │
│         └───────────┬───────────────────────┘           │
│                     ▼                                   │
│            ┌──────────────────┐                         │
│            │  Image Extractor │                         │
│            │  (OCR + Vision)  │                         │
│            └──────────────────┘                         │
│                     │                                   │
│                     ▼                                   │
│            ┌──────────────────┐                         │
│            │ Data Normalizer  │                         │
│            │  (Gemini LLM)    │                         │
│            └──────────────────┘                         │
│                     │                                   │
│                     ▼                                   │
│            ┌──────────────────┐                         │
│            │  JSON Output +   │                         │
│            │   Screenshots    │                         │
│            └──────────────────┘                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              LLM Query Chatbot (llm_query.py)           │
├─────────────────────────────────────────────────────────┤
│  • Loads extracted JSON data                            │
│  • Interactive CLI for natural language queries         │
│  • Uses OpenRouter API (Claude 3.5 Sonnet)              │
│  • Context-aware responses based on scraped data        │
└─────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Software

- **Python 3.8+**
- **Tesseract OCR** - For image text extraction
  - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
  - Linux: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`

### API Keys

1. **Google Gemini API** (Free tier available)
   - Get key: https://makersuite.google.com/app/apikey
   - Used for: Data normalization

2. **OpenRouter API** (Pay-as-you-go, $5 minimum)
   - Get key: https://openrouter.ai
   - Used for: Interactive chatbot queries

---

## Installation

### Step 1: Clone Repository

```bash
cd "d:\web scrapping assesment"
```

### Step 2: Create Virtual Environment (Recommended)

Creating a virtual environment isolates project dependencies from your system Python.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Playwright Browsers

```bash
playwright install chromium
```

### Step 5: Configure Environment Variables

Create a `.env` file in the project root:

```env
# Gemini API (Required)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash

# OpenRouter API (Optional - for chatbot)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=nvidia/nemotron-3-nano-30b-a3b:free

```

---

## Configuration

### URLs File (`urls.txt`)

Add one URL per line:

```txt
https://coinmarketcap.com
https://www.worldometers.info/coronavirus/
https://en.wikipedia.org/wiki/List_of_countries_by_population
```

---

## Usage

### 1. Web Scraping

```bash
python web_scraper.py
```

**What happens:**
- Reads URLs from `urls.txt`
- Detects page type for each URL
- Extracts data using appropriate strategy
- Normalizes data with LLM
- Saves to `output/url{N}.json`
- Captures screenshots to `output/url{N}_screenshot.png`
- Generates extraction report

### 2. Interactive Data Querying

```bash
python llm_query.py
```

**Available Commands:**

```bash
You: What are the top 5 cryptocurrencies by market cap?
Assistant: Based on the data from CoinMarketCap...

You: /help          # Show all commands
You: /reload        # Reload data from output folder
You: /reset         # Clear conversation history
You: /data          # Show data summary
You: /quit          # Exit chatbot
```

---

## Algorithmic Approach

### Phase 1: Page Type Detection

```python
Algorithm: Intelligent Page Classification
────────────────────────────────────────
Input: URL, HTML content

1. Parse HTML with BeautifulSoup
2. Analyze indicators:
   ├─ JavaScript frameworks (React, Vue, Angular)
   ├─ Dynamic loading patterns
   ├─ Canvas/SVG charts
   ├─ Image density
   └─ Table complexity

3. Classification Decision Tree:
   ├─ Canvas + Many Images → IMAGE_BASED
   ├─ JS Framework → DYNAMIC
   ├─ Complex Tables → STATIC
   └─ Mixed signals → HYBRID

Output: Page type classification
```

### Phase 2: Adaptive Data Extraction

#### Static Extraction (BeautifulSoup)
```python
Best for: Simple HTML pages, Wikipedia tables

Process:
1. HTTP GET request with User-Agent
2. Parse HTML with lxml parser
3. Extract:
   ├─ Title and headings
   ├─ Tables with headers/rows
   ├─ Text content (paragraphs)
   └─ Metadata (meta tags)

Advantages:
- Fast and lightweight
- Low resource usage
- Works without JavaScript
```

#### Dynamic Extraction (Playwright)
```python
Best for: JavaScript-heavy pages, SPAs

Process:
1. Launch headless Chromium browser
2. Multi-strategy page loading:
   ├─ Try 'networkidle' (wait for network)
   ├─ Fallback to 'load' (DOM loaded)
   └─ Final fallback to 'domcontentloaded'

3. Interaction simulation:
   ├─ Click expandable elements
   ├─ Trigger tab switches
   └─ Scroll to lazy-load content

4. Capture:
   ├─ Rendered HTML
   └─ Full-page screenshot

5. Parse rendered HTML with BeautifulSoup

Advantages:
- Handles dynamic content
- Captures visual state
- Works with lazy loading
```

#### Image/OCR Extraction (Tesseract + Vision AI)
```python
Best for: Charts, graphs, image-based data

Process:
1. Download all images from page
2. Perform OCR with Tesseract:
   ├─ Extract text from images
   ├─ Detect image type (chart/table/map)
   └─ Calculate confidence scores

3. Optional: Gemini Vision API analysis
   ├─ Structured description
   └─ Data extraction from charts

4. Process full-page screenshot with OCR

Advantages:
- Extracts text from images
- Handles charts/graphs
- Vision AI for complex visuals
```

### Phase 3: Data Normalization (LLM)

```python
Algorithm: LLM-Powered Data Structuring
──────────────────────────────────────

Input: Raw extracted data (JSON)

1. Prepare structured prompt:
   ├─ Include all raw data
   ├─ Define output schema
   └─ Add normalization instructions

2. Send to Gemini LLM with instructions:
   ├─ Clean and organize tables
   ├─ Remove duplicates
   ├─ Add metadata
   ├─ Extract key insights
   └─ Identify data patterns

3. Parse LLM response:
   ├─ Extract JSON from markdown
   ├─ Validate structure
   └─ Fallback to basic normalization on error

Output: Structured, normalized JSON
```

### Phase 4: Enhanced Table Data Cleaning

```python
Algorithm: Intelligent Table Parsing
────────────────────────────────────

For each table:
1. Identify headers (th tags or first row)

2. Clean cell data with regex patterns:
   ├─ Name columns: Extract "BitcoinBTC" → name:"Bitcoin", symbol:"BTC"
   ├─ Price columns: Keep currency format
   ├─ Market cap: Extract readable format "$1.76T"
   ├─ Volume: Parse 24h volume
   ├─ Supply: Extract circulating supply with units
   └─ Percentages: Parse change indicators

3. Create structured row objects:
   {
     "rank": "1",
     "name": "Bitcoin",
     "symbol": "BTC",
     "price": "$88,042.24",
     "market_cap": "$1.76T",
     "volume_24h": "$54.76B",
     "circulating_supply": "19.96M"
   }

Advantages:
- Machine-readable structure
- Consistent field naming
- Better for LLM queries
```

### Phase 5: Retry Logic with Exponential Backoff

```python
Algorithm: Resilient Request Handling
─────────────────────────────────────

max_retries = 3
retry_delay = 2 seconds

For each request:
  attempt = 0
  while attempt <= max_retries:
    try:
      Execute request
      Break on success
    except Error:
      attempt += 1
      wait_time = retry_delay * (2 ^ (attempt - 1))
      Sleep(wait_time)
      
  If all attempts fail:
    Return error response
    Continue to next URL

Example delays: 2s → 4s → 8s
```

---

## Output Structure

### JSON Output Files (`output/url{N}.json`)

```json
{
  "metadata": {
    "source_url": "https://example.com",
    "data_type": "cryptocurrency_market_data",
    "extraction_date": "2025-12-25T23:32:00",
    "quality_score": 0.95
  },
  "structured_data": {
    "title": "Page Title",
    "tables": [
      {
        "table_id": "table_1",
        "headers": ["Name", "Price", "Market Cap"],
        "structured_rows": [
          {
            "rank": "1",
            "name": "Bitcoin",
            "symbol": "BTC",
            "price": "$88,042.24",
            "market_cap": "$1.76T"
          }
        ],
        "row_count": 100,
        "column_count": 7
      }
    ],
    "screenshot": "output/url1_screenshot.png"
  },
  "analysis": {
    "summary": "Cryptocurrency market data...",
    "key_findings": [...],
    "recommended_uses": [...]
  },
  "extraction_metadata": {
    "page_type": "DYNAMIC",
    "extraction_method": "Playwright + BeautifulSoup",
    "total_processing_time": 45.67,
    "timestamp": "2025-12-25T23:32:45"
  }
}
```

### Log Files (`scraper_YYYYMMDD.log`)

Detailed execution logs with timestamps, levels, and messages.

### Extraction Report (`output/extraction_report_YYYYMMDD_HHMMSS.json`)

Summary of all processed URLs with success/failure statistics.

---

## Troubleshooting

### Common Issues

#### 1. Playwright Installation Error
```bash
# Solution: Reinstall Playwright browsers
playwright install chromium --force
```

#### 2. Tesseract Not Found
```bash
# Windows: Add to PATH or set in code
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

#### 3. Gemini API Error (404 Model Not Found)
```env
# Update .env to use correct model name
GEMINI_MODEL=gemini-1.5-flash-latest
```

#### 4. Page Load Timeout
```env
# Increase timeout in .env
BROWSER_TIMEOUT=90000
```

#### 5. Memory Issues with Large Pages
```python
# Reduce concurrent processing or increase system RAM
# Consider processing URLs in batches
```

---

## API Credits

### Google Gemini API
- **Free Tier:** 60 requests/minute
- **Cost:** Free for moderate usage
- **Get Key:** https://makersuite.google.com/app/apikey

### OpenRouter API
- **Pricing:** Pay-as-you-go (starts at $5 credit)
- **Models:** Claude 3.5 Sonnet (~$3 per 1M tokens)
- **Get Key:** https://openrouter.ai
- **Note:** Required for chatbot functionality

---

## Project Files

```
d:\web scrapping assesment\
├── web_scraper.py          # Main scraper (850 lines)
├── llm_query.py            # Interactive chatbot (270 lines)
├── requirements.txt        # Python dependencies
├── .env                    # Configuration (API keys)
├── urls.txt                # Input URLs
├── README.md               # This file
├── output/                 # Extraction results
│   ├── url1.json          # Structured data
│   ├── url1_screenshot.png # Page screenshot
│   └── extraction_report_*.json
└── scraper_YYYYMMDD.log   # Execution logs
```

---

## Best Practices

1. **Rate Limiting:** Add delays between requests to avoid server blocking
2. **User-Agent:** Always use realistic User-Agent headers
3. **Error Handling:** Check logs for debugging failed extractions
4. **Data Validation:** Verify JSON output structure before using
5. **API Quotas:** Monitor Gemini/OpenRouter usage to avoid rate limits
6. **Screenshot Storage:** Clean old screenshots periodically to save disk space

---

## Performance Metrics

| Metric | Average Value |
|--------|---------------|
| Static page processing | 5-10 seconds |
| Dynamic page processing | 30-60 seconds |
| Image-heavy page | 45-90 seconds |
| LLM normalization | 2-5 seconds |
| Memory usage | 200-500 MB |

---

## Future Enhancements

- [ ] Parallel URL processing
- [ ] Database storage (SQLite/PostgreSQL)
- [ ] Web UI dashboard
- [ ] Real-time monitoring
- [ ] Advanced chart recognition
- [ ] Multi-language OCR
- [ ] Scheduled scraping


---

## Support

For issues or questions:
1. Check logs in `scraper_YYYYMMDD.log`
2. Review [Troubleshooting](#troubleshooting) section
3. Verify API keys in `.env` file
4. Ensure all dependencies are installed

---

**Last Updated:** December 25, 2025
