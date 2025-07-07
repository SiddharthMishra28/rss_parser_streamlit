import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
import os
from sentiment_analysis import analyze_feed, analyze_uploaded_file
from utils import build_sentiment_summary, make_clickable_link, parse_uploaded_file

# Configure page
st.set_page_config(
    page_title="Investment Banking Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("📊 Investment Banking Sentiment Analysis Dashboard")
st.markdown("---")

# Sidebar for input controls
st.sidebar.header("Analysis Controls")

# Input method selection
analysis_type = st.sidebar.radio(
    "Choose Analysis Type:",
    ["Live RSS Feed", "Upload File"],
    help="Select whether to analyze live RSS feeds or uploaded files"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'current_keyword' not in st.session_state:
    st.session_state.current_keyword = 'ubs'

# Analysis controls based on type
if analysis_type == "Live RSS Feed":
    st.sidebar.subheader("RSS Feed Analysis")
    
    # Keyword input
    keyword = st.sidebar.text_input(
        "Enter keyword for news search:",
        value=st.session_state.current_keyword,
        help="Enter a keyword to search for relevant news articles"
    )
    
    # Analyze button
    if st.sidebar.button("🔍 Analyze Sentiment", type="primary"):
        if keyword.strip():
            with st.spinner(f"Fetching and analyzing news for '{keyword}'..."):
                try:
                    # Create requests session
                    session = requests.Session()
                    
                    # Analyze the feed
                    results_df = analyze_feed(keyword, session)
                    
                    if results_df is not None and not results_df.empty:
                        st.session_state.analysis_results = results_df
                        st.session_state.current_keyword = keyword
                        st.success(f"✅ Analysis complete! Found {len(results_df)} articles.")
                    else:
                        st.error("❌ No articles found for the given keyword.")
                        st.session_state.analysis_results = None
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.session_state.analysis_results = None
        else:
            st.sidebar.error("Please enter a keyword.")

else:  # Upload File
    st.sidebar.subheader("File Upload Analysis")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload your file:",
        type=['csv', 'json', 'xml', 'rss'],
        help="Support formats: CSV, JSON, XML, RSS"
    )
    
    # Analyze button
    if st.sidebar.button("📁 Analyze Uploaded File", type="primary"):
        if uploaded_file is not None:
            with st.spinner("Processing uploaded file..."):
                try:
                    # Analyze the uploaded file
                    results_df = analyze_uploaded_file(uploaded_file)
                    
                    if results_df is not None and not results_df.empty:
                        st.session_state.analysis_results = results_df
                        st.success(f"✅ File analysis complete! Found {len(results_df)} articles.")
                    else:
                        st.error("❌ No valid articles found in the uploaded file.")
                        st.session_state.analysis_results = None
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    st.session_state.analysis_results = None
        else:
            st.sidebar.error("Please upload a file first.")

# Display results if available
if st.session_state.analysis_results is not None:
    df = st.session_state.analysis_results
    
    # Build sentiment summary
    summary = build_sentiment_summary(df)
    
    # Display summary cards
    st.subheader("📈 Sentiment Overview")
    
    # Create columns for sentiment cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🟢 Positive",
            value=summary['positive_count'],
            delta=f"{summary['positive_pct']:.1f}%"
        )
    
    with col2:
        st.metric(
            label="🔴 Negative",
            value=summary['negative_count'],
            delta=f"{summary['negative_pct']:.1f}%"
        )
    
    with col3:
        st.metric(
            label="⚪ Neutral",
            value=summary['neutral_count'],
            delta=f"{summary['neutral_pct']:.1f}%"
        )
    
    with col4:
        st.metric(
            label="📰 Total Articles",
            value=summary['total_count']
        )
    
    st.markdown("---")
    
    # Create two columns for charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("🍩 Sentiment Distribution")
        
        # Create donut chart
        fig_donut = px.pie(
            values=[summary['positive_count'], summary['negative_count'], summary['neutral_count']],
            names=['Positive', 'Negative', 'Neutral'],
            color_discrete_map={
                'Positive': '#00ff00',
                'Negative': '#ff0000',
                'Neutral': '#808080'
            },
            hole=0.4
        )
        
        fig_donut.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig_donut.update_layout(
            showlegend=True,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with chart_col2:
        st.subheader("📊 Sentiment Trend Over Time")
        
        # Prepare trend data
        if 'date' in df.columns:
            # Convert date to datetime if it's not already
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Group by date and sentiment
            trend_data = df.groupby([df['date'].dt.date, 'textblob_sentiment']).size().reset_index(name='count')
            
            # Create line chart
            fig_line = px.line(
                trend_data,
                x='date',
                y='count',
                color='textblob_sentiment',
                color_discrete_map={
                    'Positive': '#00ff00',
                    'Negative': '#ff0000',
                    'Neutral': '#808080'
                },
                title="Sentiment Count Over Time"
            )
            
            fig_line.update_layout(
                xaxis_title="Date",
                yaxis_title="Article Count",
                height=400,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("📅 Date information not available for trend analysis.")
    
    st.markdown("---")
    
    # Display data table
    st.subheader("📋 Article Details")
    
    # Prepare display DataFrame
    display_df = df.copy()
    
    # Make article titles clickable if URL is available
    if 'url' in display_df.columns:
        display_df['title'] = display_df.apply(
            lambda row: make_clickable_link(row['title'], row['url']), 
            axis=1
        )
    
    # Select columns to display
    columns_to_show = ['title', 'source', 'date', 'textblob_sentiment', 'vader_sentiment', 'action_urgency']
    available_columns = [col for col in columns_to_show if col in display_df.columns]
    
    # Display the table
    st.dataframe(
        display_df[available_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            'title': st.column_config.TextColumn(
                'Article Title',
                width='large'
            ),
            'source': st.column_config.TextColumn(
                'Source',
                width='medium'
            ),
            'date': st.column_config.DatetimeColumn(
                'Published Date',
                width='medium'
            ),
            'textblob_sentiment': st.column_config.TextColumn(
                'TextBlob Sentiment',
                width='small'
            ),
            'vader_sentiment': st.column_config.TextColumn(
                'VADER Sentiment',
                width='small'
            ),
            'action_urgency': st.column_config.TextColumn(
                'Action Urgency',
                width='small'
            )
        }
    )
    
    st.markdown("---")
    
    # Chat interface placeholder
    st.subheader("💬 Chat Interface")
    st.info("🚧 Chat functionality will be available soon for advanced Q&A about the sentiment analysis results.")
    
    # Chat input placeholder
    chat_input = st.text_input(
        "Ask a question about the sentiment analysis:",
        placeholder="e.g., What are the main themes in negative articles?",
        help="Future enhancement: This will enable RAG-based Q&A about the analyzed articles."
    )
    
    if chat_input:
        st.info("💡 Chat functionality coming soon! Your question will be processed using RAG-based analysis.")

else:
    # Welcome message when no analysis has been performed
    st.info(
        """
        👋 **Welcome to the Investment Banking Sentiment Analysis Dashboard!**
        
        **Getting Started:**
        1. **Live RSS Feed Analysis**: Enter a keyword (e.g., 'ubs', 'goldman sachs') to analyze current news
        2. **File Upload Analysis**: Upload CSV, JSON, XML, or RSS files for offline analysis
        3. Click the appropriate analysis button to start
        
        **Features:**
        - 📊 Interactive sentiment distribution charts
        - 📈 Trend analysis over time
        - 📋 Detailed article breakdown with clickable links
        - 💬 Chat interface for advanced Q&A (coming soon)
        
        **Supported File Formats:**
        - **CSV**: Columns should include 'title', 'summary', 'published', 'url'
        - **JSON**: Objects with similar structure
        - **XML/RSS**: Standard RSS feed format
        """
    )
    
    # Sample data format information
    with st.expander("📋 Sample Data Format"):
        st.code("""
        CSV Format:
        title,summary,published,url,source
        "Article Title","Article summary...","2025-07-07","https://example.com","Source Name"
        
        JSON Format:
        [
            {
                "title": "Article Title",
                "summary": "Article summary...",
                "published": "2025-07-07",
                "url": "https://example.com",
                "source": "Source Name"
            }
        ]
        """, language='text')

# Footer
st.markdown("---")
st.caption("🏦 Investment Banking Sentiment Analysis Dashboard | Built with Streamlit, TextBlob, and VADER")
