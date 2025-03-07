import streamlit as st
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Import your analysis classes (make sure these modules are in your working directory)
from PDFAnalyzer import PDFAnalyzer
from LLMThemeComparator import LLMThemeComparator

# Configuration parameters
UPLOAD_FOLDER = "out"
ANALYSIS_MODEL = "gpt-4"  # or "gpt-3.5-turbo" based on your needs
API_KEY = "your-api-key"  # Replace with your actual API key

def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

def save_uploaded_file(file_content, company, year, doc_type, file_name):
    file_path = Path(UPLOAD_FOLDER) / company / year / doc_type / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path

def process_pdf(file_path, company, year):
    analyzer = PDFAnalyzer(API_KEY)
    analysis = analyzer.analyze_pdf(str(file_path))
    # Generate a combined analysis (includes both verbatim summary and insights)
    return analyzer.generate_complete_analysis(analysis)

def load_or_process_document(file_content, company, year, doc_type):
    file_hash = get_file_hash(file_content)
    file_name = f"{file_hash}.pdf"
    analysis_path = Path(UPLOAD_FOLDER) / company / year / "analysis" / f"{file_hash}_analysis.json"
    
    # If analysis exists, load it
    if analysis_path.exists():
        with open(analysis_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Otherwise, save the file and process it
    file_path = save_uploaded_file(file_content, company, year, doc_type, file_name)
    analysis = process_pdf(file_path, company, year)
    
    # Save analysis for future reuse
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f)
    
    return analysis

def aggregate_company_data(company_data, year):
    aggregated = {
        "company_name": company_data.get("company_name", "Unknown Company"),
        "themes_data": defaultdict(list),
        "financial_metrics": defaultdict(dict),
        "time_period": year
    }
    
    # Aggregate verbatim quotes by theme and financial metrics from all documents
    for doc in company_data["documents"]:
        if 'aggregated_verbatim_quotes' in doc['analysis']:
            for theme, quotes in doc['analysis']['aggregated_verbatim_quotes'].items():
                aggregated["themes_data"][theme].extend(quotes)
        if 'financial_metrics' in doc['analysis']:
            for metric, value in doc['analysis']['financial_metrics'].items():
                aggregated["financial_metrics"][metric] = value
                
    return aggregated

def main():
    st.title("Company Analysis and Theme Comparison")
    st.write("Upload PDF documents (e.g. earnings call transcripts, factsheets, or financials) for both Accenture and your other company. "
             "The system will perform analysis (or load prior results), aggregate the data, and compare the companies based on key themes.")
    
    # Year selection (used for folder structure)
    selected_year = st.selectbox(
        "Select Year", 
        [str(y) for y in range(2018, datetime.now().year+1)],
        index=3
    )
    
    # Initialize session state for processed document data
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {
            'Accenture': {'documents': []},
            'Other Company': {'documents': []}
        }
    
    # Two-column layout for uploads
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accenture Documents")
        with st.expander("Upload Transcript PDFs"):
            accenture_transcripts = st.file_uploader("Accenture Transcripts", type=["pdf"], 
                                                     accept_multiple_files=True, key="accenture_transcripts")
        with st.expander("Upload Factsheet PDFs"):
            accenture_factsheets = st.file_uploader("Accenture Factsheets", type=["pdf"], 
                                                     accept_multiple_files=True, key="accenture_factsheets")
        with st.expander("Upload Financial PDFs"):
            accenture_financials = st.file_uploader("Accenture Financials", type=["pdf"], 
                                                     accept_multiple_files=True, key="accenture_financials")
    with col2:
        st.subheader("Other Company Documents")
        with st.expander("Upload Transcript PDFs"):
            other_transcripts = st.file_uploader("Other Transcripts", type=["pdf"], 
                                                 accept_multiple_files=True, key="other_transcripts")
        with st.expander("Upload Factsheet PDFs"):
            other_factsheets = st.file_uploader("Other Factsheets", type=["pdf"], 
                                                 accept_multiple_files=True, key="other_factsheets")
        with st.expander("Upload Financial PDFs"):
            other_financials = st.file_uploader("Other Financials", type=["pdf"], 
                                                 accept_multiple_files=True, key="other_financials")
    
    if st.button("Process & Compare Documents"):
        companies = {
            'Accenture': {
                'Transcripts': accenture_transcripts,
                'Factsheets': accenture_factsheets,
                'Financials': accenture_financials
            },
            'Other Company': {
                'Transcripts': other_transcripts,
                'Factsheets': other_factsheets,
                'Financials': other_financials
            }
        }
        
        # Process each uploaded file (or load cached analysis)
        for company, doc_types in companies.items():
            st.session_state.processed_data[company]['documents'] = []
            for doc_type, files in doc_types.items():
                if files:
                    for file in files:
                        with st.spinner(f"Processing {company} {doc_type}..."):
                            file_content = file.getvalue()
                            analysis = load_or_process_document(file_content, company, selected_year, doc_type)
                            st.session_state.processed_data[company]['documents'].append({
                                'type': doc_type,
                                'analysis': analysis
                            })
        
        # Aggregate data for each company from all processed documents
        accenture_data = aggregate_company_data(st.session_state.processed_data['Accenture'], selected_year)
        other_data = aggregate_company_data(st.session_state.processed_data['Other Company'], selected_year)
        
        # Initialize the LLM-based theme comparator
        comparator = LLMThemeComparator(API_KEY, ANALYSIS_MODEL)
        
        # Run theme comparison (this includes similarity scoring and markdown-based analysis)
        theme_comparison = comparator.compare_companies(accenture_data, other_data)
        
        st.header("Theme-based Comparison")
        for theme, comparison in theme_comparison['comparisons'].items():
            with st.expander(f"{theme} Comparison"):
                st.markdown(comparison)
        
        st.subheader("Financial Metrics Comparison")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("### Accenture Financial Metrics")
            for metric, value in accenture_data['financial_metrics'].items():
                st.write(f"**{metric}**: {value}")
        with col_f2:
            st.markdown("### Other Company Financial Metrics")
            for metric, value in other_data['financial_metrics'].items():
                st.write(f"**{metric}**: {value}")
        
        st.subheader("Strategic Insights")
        insights = comparator.generate_financial_comparison(
            accenture_data['financial_metrics'],
            other_data['financial_metrics']
        )
        st.markdown(insights)
        
        # Create visualizations and display them in the UI
        st.header("Visualizations")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_output_dir = f"theme_comparisons_{timestamp}"
        os.makedirs(vis_output_dir, exist_ok=True)
        
        sorted_df = comparator.create_visualizations(
            theme_comparison['summary_df'],
            accenture_data["company_name"],
            other_data["company_name"],
            vis_output_dir
        )
        
        sim_img_path = os.path.join(vis_output_dir, "theme_similarity_scores.png")
        mentions_img_path = os.path.join(vis_output_dir, "theme_mentions_comparison.png")
        if os.path.exists(sim_img_path):
            st.image(sim_img_path, caption="Theme Similarity Scores", use_column_width=True)
        if os.path.exists(mentions_img_path):
            st.image(mentions_img_path, caption="Theme Mentions Comparison", use_column_width=True)
        
        st.success("Processing and comparison complete.")

if __name__ == "__main__":
    main()
