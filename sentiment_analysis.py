import pandas as pd
import requests
import feedparser
import xml.etree.ElementTree as ET
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import re
import os

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

def analyze_feed(keyword: str, session: requests.Session) -> pd.DataFrame:
    """
    Analyze sentiment from Google News RSS feed for a given keyword.
    
    Args:
        keyword (str): The keyword to search for
        session (requests.Session): HTTP session for making requests
        
    Returns:
        pd.DataFrame: DataFrame with sentiment analysis results
    """
    try:
        # Construct Google News RSS URL
        base_url = "https://news.google.com/rss/search"
        params = {
            'q': keyword,
            'hl': 'en',
            'gl': 'US',
            'ceid': 'US:en'
        }
        
        # Set up session with headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Make request to Google News RSS
        response = session.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse RSS feed
        feed = feedparser.parse(response.content)
        
        if not feed.entries:
            return pd.DataFrame()
        
        # Extract article data
        articles_data = []
        
        for entry in feed.entries:
            try:
                # Extract basic information
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
                
                # Parse published date
                try:
                    if published:
                        pub_date = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %Z')
                    else:
                        pub_date = datetime.now()
                except:
                    pub_date = datetime.now()
                
                # Combine title and summary for comprehensive sentiment analysis
                # Use both title and summary for better accuracy
                text_content = f"{title}. {summary}" if summary else title
                
                # Perform sentiment analysis
                sentiment_results = perform_sentiment_analysis(text_content)
                
                # Create article record
                article = {
                    'title': title,
                    'summary': summary,
                    'date': pub_date,
                    'url': link,
                    'source': source,
                    'textblob_sentiment': sentiment_results['textblob_sentiment'],
                    'textblob_polarity': sentiment_results['textblob_polarity'],
                    'textblob_subjectivity': sentiment_results['textblob_subjectivity'],
                    'vader_sentiment': sentiment_results['vader_sentiment'],
                    'vader_compound': sentiment_results['vader_compound'],
                    'vader_positive': sentiment_results['vader_positive'],
                    'vader_negative': sentiment_results['vader_negative'],
                    'vader_neutral': sentiment_results['vader_neutral'],
                    'action_urgency': sentiment_results['action_urgency']
                }
                
                articles_data.append(article)
                
            except Exception as e:
                print(f"Error processing article: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(articles_data)
        
        # Sort by date (newest first)
        if not df.empty:
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Error analyzing feed: {e}")
        return pd.DataFrame()

def analyze_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Analyze sentiment from uploaded file (CSV, JSON, XML, RSS).
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        pd.DataFrame: DataFrame with sentiment analysis results
    """
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'csv':
            return analyze_csv_file(uploaded_file)
        elif file_type == 'json':
            return analyze_json_file(uploaded_file)
        elif file_type in ['xml', 'rss']:
            return analyze_xml_rss_file(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        print(f"Error analyzing uploaded file: {e}")
        return pd.DataFrame()

def analyze_csv_file(uploaded_file) -> pd.DataFrame:
    """Analyze sentiment from CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check required columns
        required_columns = ['title']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV must contain 'title' column")
        
        # Process each row
        results = []
        for _, row in df.iterrows():
            try:
                title = str(row.get('title', ''))
                summary = str(row.get('summary', ''))
                published = row.get('published', '')
                url = row.get('url', '')
                source = row.get('source', 'Unknown')
                
                # Parse date
                try:
                    if published:
                        pub_date = pd.to_datetime(published)
                    else:
                        pub_date = datetime.now()
                except:
                    pub_date = datetime.now()
                
                # Combine text for analysis
                text_content = f"{title}. {summary}"
                
                # Perform sentiment analysis
                sentiment_results = perform_sentiment_analysis(text_content)
                
                # Create record
                result = {
                    'title': title,
                    'summary': summary,
                    'date': pub_date,
                    'url': url,
                    'source': source,
                    **sentiment_results
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"Error analyzing CSV file: {e}")
        return pd.DataFrame()

def analyze_json_file(uploaded_file) -> pd.DataFrame:
    """Analyze sentiment from JSON file."""
    try:
        import json
        
        content = uploaded_file.read()
        data = json.loads(content)
        
        # Handle both list and single object
        if isinstance(data, dict):
            data = [data]
        
        results = []
        for item in data:
            try:
                title = str(item.get('title', ''))
                summary = str(item.get('summary', ''))
                published = item.get('published', '')
                url = item.get('url', '')
                source = item.get('source', 'Unknown')
                
                # Parse date
                try:
                    if published:
                        pub_date = pd.to_datetime(published)
                    else:
                        pub_date = datetime.now()
                except:
                    pub_date = datetime.now()
                
                # Combine text for analysis
                text_content = f"{title}. {summary}"
                
                # Perform sentiment analysis
                sentiment_results = perform_sentiment_analysis(text_content)
                
                # Create record
                result = {
                    'title': title,
                    'summary': summary,
                    'date': pub_date,
                    'url': url,
                    'source': source,
                    **sentiment_results
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing JSON item: {e}")
                continue
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"Error analyzing JSON file: {e}")
        return pd.DataFrame()

def analyze_xml_rss_file(uploaded_file) -> pd.DataFrame:
    """Analyze sentiment from XML/RSS file."""
    try:
        content = uploaded_file.read()
        
        # Try parsing as RSS first
        try:
            feed = feedparser.parse(content)
            if feed.entries:
                return analyze_rss_entries(feed.entries)
        except:
            pass
        
        # Try parsing as XML
        try:
            root = ET.fromstring(content)
            return analyze_xml_elements(root)
        except:
            pass
        
        raise ValueError("Could not parse XML/RSS content")
        
    except Exception as e:
        print(f"Error analyzing XML/RSS file: {e}")
        return pd.DataFrame()

def analyze_rss_entries(entries) -> pd.DataFrame:
    """Analyze RSS feed entries."""
    results = []
    
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
            
            # Parse published date
            try:
                if published:
                    pub_date = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %Z')
                else:
                    pub_date = datetime.now()
            except:
                pub_date = datetime.now()
            
            # Combine title and summary for comprehensive sentiment analysis
            # Use both title and summary for better accuracy
            text_content = f"{title}. {summary}" if summary else title
            
            # Perform sentiment analysis
            sentiment_results = perform_sentiment_analysis(text_content)
            
            # Create record
            result = {
                'title': title,
                'summary': summary,
                'date': pub_date,
                'url': link,
                'source': source,
                **sentiment_results
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing RSS entry: {e}")
            continue
    
    return pd.DataFrame(results)

def analyze_xml_elements(root) -> pd.DataFrame:
    """Analyze generic XML elements."""
    results = []
    
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
            
            # Parse date
            try:
                if published:
                    pub_date = pd.to_datetime(published)
                else:
                    pub_date = datetime.now()
            except:
                pub_date = datetime.now()
            
            # Combine text for analysis
            text_content = f"{title}. {summary}"
            
            # Perform sentiment analysis
            sentiment_results = perform_sentiment_analysis(text_content)
            
            # Create record
            result = {
                'title': title,
                'summary': summary,
                'date': pub_date,
                'url': link,
                'source': source,
                **sentiment_results
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing XML item: {e}")
            continue
    
    return pd.DataFrame(results)

def perform_sentiment_analysis(text: str) -> dict:
    """
    Perform sentiment analysis using TextBlob and VADER.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Sentiment analysis results
    """
    try:
        # TextBlob analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Classify TextBlob sentiment
        if textblob_polarity > 0.1:
            textblob_sentiment = 'Positive'
        elif textblob_polarity < -0.1:
            textblob_sentiment = 'Negative'
        else:
            textblob_sentiment = 'Neutral'
        
        # VADER analysis
        vader_scores = vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']
        vader_positive = vader_scores['pos']
        vader_negative = vader_scores['neg']
        vader_neutral = vader_scores['neu']
        
        # Classify VADER sentiment
        if vader_compound >= 0.05:
            vader_sentiment = 'Positive'
        elif vader_compound <= -0.05:
            vader_sentiment = 'Negative'
        else:
            vader_sentiment = 'Neutral'
        
        # Determine action urgency
        if textblob_sentiment == 'Negative' and vader_sentiment == 'Negative':
            action_urgency = 'High'
        elif textblob_sentiment == 'Positive' and vader_sentiment == 'Positive':
            action_urgency = 'Low'
        else:
            action_urgency = 'Medium'
        
        return {
            'textblob_sentiment': textblob_sentiment,
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'vader_sentiment': vader_sentiment,
            'vader_compound': vader_compound,
            'vader_positive': vader_positive,
            'vader_negative': vader_negative,
            'vader_neutral': vader_neutral,
            'action_urgency': action_urgency
        }
        
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return {
            'textblob_sentiment': 'Neutral',
            'textblob_polarity': 0.0,
            'textblob_subjectivity': 0.0,
            'vader_sentiment': 'Neutral',
            'vader_compound': 0.0,
            'vader_positive': 0.0,
            'vader_negative': 0.0,
            'vader_neutral': 1.0,
            'action_urgency': 'Medium'
        }
