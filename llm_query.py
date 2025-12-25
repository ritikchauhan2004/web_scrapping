import os
import sys
import json
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

try:
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Please install: pip install openai python-dotenv")
    sys.exit(1)


class DataLoader:
    """Load and manage extracted JSON data"""
    
    def __init__(self, output_folder: str = "output"):
        self.output_folder = output_folder
        self.data_cache = {}
        self.load_all_data()
    
    def load_all_data(self):
        """Load all JSON files from output folder"""
        if not os.path.exists(self.output_folder):
            print(f"Warning: Output folder '{self.output_folder}' not found")
            return
        
        json_files = [f for f in os.listdir(self.output_folder) if f.endswith('.json')]
        
        for filename in json_files:
            filepath = os.path.join(self.output_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.data_cache[filename] = data
                    print(f"✓ Loaded: {filename}")
            except Exception as e:
                print(f"✗ Failed to load {filename}: {e}")
        
        print(f"\n Total files loaded: {len(self.data_cache)}")
    
    def get_all_data(self) -> Dict[str, Any]:
        """Return all loaded data"""
        return self.data_cache
    
    def get_data_summary(self) -> str:
        """Generate a summary of available data for context"""
        if not self.data_cache:
            return "No data available."
        
        summary_parts = []
        for filename, data in self.data_cache.items():
            if 'metadata' in data:
                url = data['metadata'].get('source_url', 'Unknown')
                timestamp = data['metadata'].get('extraction_timestamp', 'Unknown')
                summary_parts.append(f"- {filename}: Data from {url} (extracted: {timestamp})")
            
            if 'structured_data' in data:
                sd = data['structured_data']
                title = sd.get('title', 'No title')
                table_count = len(sd.get('tables', []))
                summary_parts.append(f"  Title: {title}")
                summary_parts.append(f"  Tables: {table_count}")
        
        return "\n".join(summary_parts)
    
    def format_data_for_context(self) -> str:
        """Format all data for LLM context in a concise way"""
        context_parts = ["=== AVAILABLE SCRAPED DATA ===\n"]
        
        for filename, data in self.data_cache.items():
            context_parts.append(f"\n--- File: {filename} ---")
            
            # Metadata
            if 'metadata' in data:
                meta = data['metadata']
                context_parts.append(f"Source: {meta.get('source_url', 'Unknown')}")
                context_parts.append(f"Extracted: {meta.get('extraction_timestamp', 'Unknown')}")
                context_parts.append(f"Page Type: {meta.get('page_type', 'Unknown')}")
            
            # Structured Data
            if 'structured_data' in data:
                sd = data['structured_data']
                
                # Title
                if 'title' in sd:
                    context_parts.append(f"\nTitle: {sd['title']}")
                
                # Tables
                if 'tables' in sd and sd['tables']:
                    context_parts.append(f"\nTables ({len(sd['tables'])}):")
                    for idx, table in enumerate(sd['tables'], 1):
                        table_id = table.get('table_id', f'table_{idx}')
                        headers = table.get('headers', [])
                        
                        structured_rows = table.get('structured_rows', [])
                        rows = table.get('rows', [])
                        
                        context_parts.append(f"  Table {idx} ({table_id}):")
                        context_parts.append(f"    Columns: {', '.join(headers)}")  # All headers
                        context_parts.append(f"    Total Rows: {len(structured_rows) if structured_rows else len(rows)}")
                        
                        # Include structured data if available
                        if structured_rows:
                            context_parts.append(f"    STRUCTURED DATA ({len(structured_rows)} items):")
                            for row_idx, row in enumerate(structured_rows, 1):
                                # Format each row as key-value pairs
                                row_str = ', '.join([f"{k}: {v}" for k, v in row.items() if v])
                                context_parts.append(f"      [{row_idx}] {row_str}")
                        else:
                            # Fallback to raw rows
                            context_parts.append(f"    RAW DATA ({len(rows)} rows):")
                            for row_idx, row in enumerate(rows, 1):
                                row_str = ' | '.join([str(cell) for cell in row])
                                context_parts.append(f"      Row {row_idx}: {row_str}")
                
                # Dynamic content
                if 'dynamic_content' in sd:
                    dc = sd['dynamic_content']
                    if 'text_content' in dc:
                        text_preview = dc['text_content'][:200]
                        context_parts.append(f"\nText Content Preview: {text_preview}...")
                
                # Screenshot info
                if 'screenshot' in sd:
                    context_parts.append(f"\nScreenshot: {sd['screenshot']}")
            
            # Analysis
            if 'analysis' in data:
                analysis = data['analysis']
                if 'summary' in analysis:
                    context_parts.append(f"\nSummary: {analysis['summary']}")
                if 'key_findings' in analysis and analysis['key_findings']:
                    context_parts.append(f"Key Findings: {', '.join(analysis['key_findings'][:3])}")
        
        context_parts.append("\n\n=== END OF SCRAPED DATA ===")
        return "\n".join(context_parts)


class LLMQueryBot:
    """LLM-powered chatbot for querying scraped data using OpenRouter"""
    
    def __init__(self, openrouter_api_key: str, model: str = "anthropic/claude-3.5-sonnet", site_url: str = "", site_name: str = "Web Scraper LLM Query"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        self.model = model
        self.site_url = site_url
        self.site_name = site_name
        self.conversation_history = []
        self.data_loader = DataLoader()
        self.data_context = self.data_loader.format_data_for_context()
        
        # System prompt
        self.system_prompt = """You are an intelligent data analyst assistant. You have access to web-scraped data from various sources.
Your job is to answer user questions based ONLY on the provided scraped data. Be concise, accurate, and helpful.

When answering:
1. Only use information from the provided data context
2. If data is not available, clearly state that
3. Provide specific numbers, names, and facts when available
4. Format tables and lists clearly
5. Reference the source URL when relevant

Data Context:
""" + self.data_context
    
    def query(self, user_question: str) -> str:
        """Send query to LLM and get response via OpenRouter"""
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_question
        })
        
        # Prepare messages for API call
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history
        
        try:
            extra_headers = {}
            if self.site_url:
                extra_headers["HTTP-Referer"] = self.site_url
            if self.site_name:
                extra_headers["X-Title"] = self.site_name
            
            chat_completion = self.client.chat.completions.create(
                extra_headers=extra_headers if extra_headers else None,
                model=self.model,
                messages=messages,
                temperature=0.3, 
                max_tokens=1000, 
                top_p=0.9
            )
            
            # Extract response
            assistant_response = chat_completion.choices[0].message.content
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })
            
            return assistant_response
            
        except Exception as e:
            return f"Error querying LLM: {e}"
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def reload_data(self):
        """Reload data from output folder"""
        print("\n🔄 Reloading data...")
        self.data_loader.load_all_data()
        self.data_context = self.data_loader.format_data_for_context()
        self.system_prompt = """You are an intelligent data analyst assistant. You have access to web-scraped data from various sources.
Your job is to answer user questions based ONLY on the provided scraped data. Be concise, accurate, and helpful.

When answering:
1. Only use information from the provided data context
2. If data is not available, clearly state that
3. Provide specific numbers, names, and facts when available
4. Format tables and lists clearly
5. Reference the source URL when relevant

Data Context:
""" + self.data_context
        print("✓ Data reloaded successfully!\n")

def print_help():
    """Print available commands"""
    print("""
Available Commands:
  /help     - Show this help message
  /reload   - Reload data from output folder
  /reset    - Clear conversation history
  /data     - Show data summary
  /quit     - Exit the chatbot
  
Just type your question to query the scraped data!
    """)


def main():
    """Main chatbot interface"""
    
    # Load environment variables
    load_dotenv()
    
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    openrouter_model = os.getenv('OPENROUTER_MODEL', 'anthropic/claude-3.5-sonnet')
    site_url = os.getenv('SITE_URL', '')
    site_name = os.getenv('SITE_NAME', 'Web Scraper LLM Query')
    
    if not openrouter_api_key or openrouter_api_key == 'your_openrouter_api_key_here':
        print("Error: OPENROUTER_API_KEY not set in .env file")
        print("\nTo get an API key:")
        print("1. Visit: https://openrouter.ai")
        print("2. Sign up for an account")
        print("3. Add credits to your account")
        print("4. Create an API key")
        print("5. Add it to your .env file as OPENROUTER_API_KEY=your_key_here")
        sys.exit(1)
    
    
    # Initialize chatbot
    try:
        bot = LLMQueryBot(openrouter_api_key, openrouter_model, site_url, site_name)
        print(f"✓ Connected to OpenRouter API (Model: {openrouter_model})\n")
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        sys.exit(1)
    
    # Show data summary
    print(" Data Summary:")
    print(bot.data_loader.get_data_summary())
    print("\n" + "="*60)
    print("Type your question or /help for commands")
    print("="*60 + "\n")
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()
                
                if command == '/quit' or command == '/exit':
                    print("\n Goodbye!")
                    break
                elif command == '/help':
                    print_help()
                elif command == '/reload':
                    bot.reload_data()
                elif command == '/reset':
                    bot.reset_conversation()
                    print("✓ Conversation history cleared!\n")
                elif command == '/data':
                    print("\n Data Summary:")
                    print(bot.data_loader.get_data_summary())
                    print()
                else:
                    print(f"Unknown command: {user_input}")
                    print("Type /help for available commands\n")
                continue
            
            # Query the bot
            print("\n Assistant: ", end="", flush=True)
            response = bot.query(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\n Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}\n")


if __name__ == "__main__":
    main()
