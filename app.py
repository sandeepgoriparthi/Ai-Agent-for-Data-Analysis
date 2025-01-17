import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import ollama  # Direct Ollama client
from langchain_community.llms import Ollama as LangchainOllama  # Langchain Ollama integration
from langchain.prompts import PromptTemplate
from io import StringIO

# Configure Streamlit page
st.set_page_config(page_title="AI Data Analysis Assistant", layout="wide")

# Initialize session state for data and analysis
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# Ollama LLM Integration (Phi4)
def get_ollama_response(prompt, context=None):
    try:
        # Use Langchain's Ollama integration with Phi4
        llm = LangchainOllama(model="phi")
        return llm(prompt)
    except Exception as e:
        st.error(f"Error generating AI response: {e}")
        return None

# Direct Ollama Chat Function with Phi4
def get_direct_ollama_response(prompt):
    try:
        response = ollama.chat(model='phi', messages=[{
            'role': 'user',
            'content': prompt
        }])
        return response['message']['content']
    except Exception as e:
        st.error(f"Direct Ollama error: {e}")
        return None

# Data Analysis Functions
def perform_basic_analysis(df):
    """Perform basic statistical analysis on the dataframe"""
    analysis = {
        "Dataset Shape": df.shape,
        "Columns": list(df.columns),
        "Data Types": df.dtypes.to_dict(),
        "Summary Statistics": df.describe().to_dict()
    }
    return analysis

def generate_data_insights(df):
    """Generate AI-powered insights about the dataset"""
    # Prepare a comprehensive prompt for data insights
    prompt = f"""
    You are an expert data scientist. Analyze the following dataset:
    Columns: {list(df.columns)}
    Number of Rows: {len(df)}
    Data Types: {df.dtypes.to_dict()}
    
    Provide a detailed analysis including:
    1. Key observations about the dataset
    2. Potential patterns or trends
    3. Recommended visualization types
    4. Any potential data quality issues
    5. Suggestions for further analysis

    Be concise, professional, and provide actionable insights.
    """
    
    return get_ollama_response(prompt)

# Visualization Functions
def create_visualization(df, chart_type, x_column, y_column=None):
    """Create different types of visualizations"""
    try:
        if chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
        elif chart_type == "Line Plot":
            fig = px.line(df, x=x_column, y=y_column, title=f"{x_column} over {y_column}")
        elif chart_type == "Bar Plot":
            fig = px.bar(df, x=x_column, y=y_column, title=f"{y_column} by {x_column}")
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_column, title=f"Distribution of {x_column}")
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_column, y=y_column, title=f"{y_column} Distribution by {x_column}")
        else:
            st.error("Unsupported chart type")
            return None
        
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

# Streamlit App Layout
def main():
    st.title("🤖 AI-Powered Data Analysis Assistant")
    
    # Sidebar for navigation
    st.sidebar.header("Data Analysis Toolkit")
    
    # File Upload Section
    st.sidebar.subheader("1. Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the file
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_data = df
        
        # Tabs for different analysis modes
        tab1, tab2, tab3 = st.tabs(["Basic Analysis", "AI Insights", "Visualizations"])
        
        with tab1:
            st.header("Basic Dataset Analysis")
            analysis = perform_basic_analysis(df)
            
            # Display analysis results
            st.json(analysis)
        
        with tab2:
            st.header("AI-Generated Insights")
            with st.spinner("Generating AI insights..."):
                insights = generate_data_insights(df)
                if insights:
                    st.write(insights)
        
        with tab3:
            st.header("Data Visualization")
            
            # Visualization options
            chart_type = st.selectbox("Select Chart Type", 
                ["Scatter Plot", "Line Plot", "Bar Plot", "Histogram", "Box Plot"])
            
            # Dynamic column selection
            x_column = st.selectbox("Select X-axis Column", df.columns)
            
            # Only show y-column for charts that need it
            if chart_type in ["Scatter Plot", "Line Plot", "Bar Plot", "Box Plot"]:
                y_column = st.selectbox("Select Y-axis Column", 
                    [col for col in df.columns if col != x_column])
                
                # Create and display visualization
                fig = create_visualization(df, chart_type, x_column, y_column)
            else:
                # For Histogram
                fig = create_visualization(df, chart_type, x_column)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Please upload a CSV file to begin analysis.")
    
    # AI Chat Interface
    st.sidebar.subheader("2. AI Assistant Chat")
    user_prompt = st.sidebar.text_input("Ask about your data:")
    if user_prompt:
        with st.spinner("Generating response..."):
            response = get_ollama_response(user_prompt)
            if response:
                st.sidebar.write(response)

# Run the app
if __name__ == "__main__":
    main()
