import pandas as pd
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import re

def parse_uploaded_file(file):
    """
    Parse uploaded file and extract article data.
    
    Args:
        file: Streamlit uploaded file object
        
    Returns:
        list: List of article dictionaries
    """
    try:
        file_type = file.name.split('.')[-1].lower()
        
        if file_type == 'csv':
            return parse_csv_file(file)
        elif file_type == 'json':
            return parse_json_file(file)
        elif file_type in ['xml', 'rss']:
            return parse_xml_rss_file(file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        print(f"Error parsing file: {e}")
        return []

def parse_csv_file(file):
    """Parse CSV file."""
    try:
        df = pd.read_csv(file)
        articles = []
        
        for _, row in df.iterrows():
            article = {
                'title': str(row.get('title', '')),
                'summary': str(row.get('summary', '')),
                'published': str(row.get('published', '')),
                'url': str(row.get('url', '')),
                'source': str(row.get('source', 'Unknown'))
            }
            articles.append(article)
        
        return articles
        
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return []

def parse_json_file(file):
    """Parse JSON file."""
    try:
        content = file.read()
        data = json.loads(content)
        
        # Handle both list and single object
        if isinstance(data, dict):
            data = [data]
        
        articles = []
        for item in data:
            article = {
                'title': str(item.get('title', '')),
                'summary': str(item.get('summary', '')),
                'published': str(item.get('published', '')),
                'url': str(item.get('url', '')),
                'source': str(item.get('source', 'Unknown'))
            }
            articles.append(article)
        
        return articles
        
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return []

def parse_xml_rss_file(file):
    """Parse XML/RSS file."""
    try:
        content = file.read()
        
        # Try parsing as RSS first
        try:
            import feedparser
            feed = feedparser.parse(content)
            if feed.entries:
                return parse_rss_entries(feed.entries)
        except:
            pass
        
        # Try parsing as XML
        try:
            root = ET.fromstring(content)
            return parse_xml_elements(root)
        except:
            pass
        
        return []
        
    except Exception as e:
        print(f"Error parsing XML/RSS: {e}")
        return []

def parse_rss_entries(entries):
    """Parse RSS feed entries."""
    articles = []
    
    for entry in entries:
        try:
            title = entry.get('title', '')
            summary = entry.get('summary', '')
            published = entry.get('published', '')
            link = entry.get('link', '')
            
            # Extract source - try multiple possible locations
            source = 'Unknown'
            if hasattr(entry, 'source') and entry.source:
                if hasattr(entry.source, 'text'):
                    source = entry.source.text
                elif isinstance(entry.source, str):
                    source = entry.source
            elif 'source' in entry:
                if isinstance(entry.source, dict) and 'text' in entry.source:
                    source = entry.source['text']
                elif isinstance(entry.source, str):
                    source = entry.source
            
            # Fallback: extract from link domain
            if source == 'Unknown' and link:
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(link)
                    source = parsed_url.netloc
                except:
                    pass
            
            # Clean HTML tags from summary
            summary = re.sub(r'<[^>]+>', '', summary)
            
            article = {
                'title': title,
                'summary': summary,
                'published': published,
                'url': link,
                'source': source
            }
            articles.append(article)
            
        except Exception as e:
            print(f"Error processing RSS entry: {e}")
            continue
    
    return articles

def parse_xml_elements(root):
    """Parse generic XML elements."""
    articles = []
    
    # Look for common XML structures
    items = root.findall('.//item') or root.findall('.//entry')
    
    for item in items:
        try:
            title = ''
            summary = ''
            published = ''
            link = ''
            source = 'Unknown'
            
            # Extract title
            title_elem = item.find('title')
            if title_elem is not None:
                title = title_elem.text or ''
            
            # Extract summary/description (try summary tag first, then description)
            summary_elem = item.find('summary') or item.find('description')
            if summary_elem is not None:
                summary = summary_elem.text or ''
                summary = re.sub(r'<[^>]+>', '', summary)
            
            # Extract published date
            date_elem = item.find('pubDate') or item.find('published')
            if date_elem is not None:
                published = date_elem.text or ''
            
            # Extract link
            link_elem = item.find('link')
            if link_elem is not None:
                link = link_elem.text or ''
            
            # Extract source
            source_elem = item.find('source')
            if source_elem is not None:
                source = source_elem.text or 'Unknown'
            
            article = {
                'title': title,
                'summary': summary,
                'published': published,
                'url': link,
                'source': source
            }
            articles.append(article)
            
        except Exception as e:
            print(f"Error processing XML item: {e}")
            continue
    
    return articles

def build_sentiment_summary(df: pd.DataFrame) -> dict:
    """
    Build sentiment summary statistics from DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis results
        
    Returns:
        dict: Summary statistics
    """
    try:
        total_count = len(df)
        
        if total_count == 0:
            return {
                'total_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'positive_pct': 0.0,
                'negative_pct': 0.0,
                'neutral_pct': 0.0
            }
        
        # Count sentiments (using TextBlob sentiment as primary)
        sentiment_counts = df['textblob_sentiment'].value_counts()
        
        positive_count = sentiment_counts.get('Positive', 0)
        negative_count = sentiment_counts.get('Negative', 0)
        neutral_count = sentiment_counts.get('Neutral', 0)
        
        # Calculate percentages
        positive_pct = (positive_count / total_count) * 100
        negative_pct = (negative_count / total_count) * 100
        neutral_pct = (neutral_count / total_count) * 100
        
        return {
            'total_count': total_count,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_pct': positive_pct,
            'negative_pct': negative_pct,
            'neutral_pct': neutral_pct
        }
        
    except Exception as e:
        print(f"Error building sentiment summary: {e}")
        return {
            'total_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'positive_pct': 0.0,
            'negative_pct': 0.0,
            'neutral_pct': 0.0
        }

def make_clickable_link(title: str, url: str) -> str:
    """
    Create a clickable link in markdown format.
    
    Args:
        title (str): Link text
        url (str): Link URL
        
    Returns:
        str: Markdown formatted link
    """
    try:
        if url and url.strip():
            # Clean and validate URL
            url = url.strip()
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Create markdown link that opens in new tab
            return f"[{title}]({url})"
        else:
            return title
            
    except Exception as e:
        print(f"Error creating clickable link: {e}")
        return title

def clean_text(text: str) -> str:
    """
    Clean text by removing HTML tags and extra whitespace.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    try:
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return text

def validate_article_data(article: dict) -> bool:
    """
    Validate article data structure.
    
    Args:
        article (dict): Article data dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check required fields
        required_fields = ['title']
        
        for field in required_fields:
            if field not in article or not article[field]:
                return False
        
        return True
        
    except Exception as e:
        print(f"Error validating article data: {e}")
        return False

def format_date(date_str: str) -> datetime:
    """
    Format date string to datetime object.
    
    Args:
        date_str (str): Date string
        
    Returns:
        datetime: Formatted datetime object
    """
    try:
        # Try common date formats
        formats = [
            '%a, %d %b %Y %H:%M:%S %Z',  # RSS format
            '%Y-%m-%d %H:%M:%S',         # Standard format
            '%Y-%m-%d',                  # Date only
            '%m/%d/%Y',                  # US format
            '%d/%m/%Y',                  # EU format
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # If no format works, try pandas
        return pd.to_datetime(date_str)
        
    except Exception as e:
        print(f"Error formatting date: {e}")
        return datetime.now()
