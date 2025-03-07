import streamlit as st
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from PDFAnalyzer import PDFAnalyzer
from LLMThemeComparator import LLMThemeComparator

# Configuration
UPLOAD_FOLDER = "out"
CACHE_DIR = "cache"
ANALYSIS_MODEL = "gpt-4"
API_KEY = "API_KEY"  # Load from environment variable

# Check for API key
if not API_KEY:
    st.error("API key missing. Set the 'API_KEY' environment variable.")
    st.stop()

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

def get_file_hash(file_content):
    """Generate MD5 hash of the file content."""
    return hashlib.md5(file_content).hexdigest()

def save_uploaded_file(file_content, company, year, doc_type, file_name):
    """Save uploaded file to a structured directory."""
    file_path = Path(UPLOAD_FOLDER) / company / year / doc_type / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path

def process_pdf(file_path, company, year):
    """Process a PDF using PDFAnalyzer."""
    try:
        analyzer = PDFAnalyzer(API_KEY)
        analysis = analyzer.analyze_pdf(str(file_path))
        return analyzer.generate_complete_analysis(analysis)
    except Exception as e:
        st.error(f"Failed to process {file_path.name}: {str(e)}")
        return None

def load_or_process_document(file_content, company, year, doc_type):
    """Load existing analysis from cache or process the document and cache the result."""
    file_hash = get_file_hash(file_content)
    analysis_path = os.path.join(CACHE_DIR, f"{file_hash}_analysis.json")
    
    # Check if analysis already exists in cache
    if os.path.exists(analysis_path):
        print(f"Cached result available for {file_hash}")
        with open(analysis_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Process the file if no cached result is found
    print(f"No cached result found for {file_hash}. Processing...")
    file_path = save_uploaded_file(file_content, company, year, doc_type, f"{file_hash}.pdf")
    analysis = process_pdf(file_path, company, year)
    
    if analysis:
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f)
        print(f"Analysis saved to cache for {file_hash}")
    
    return analysis

def aggregate_company_data(company_data, year):
    """Aggregate themes and financial metrics for a company."""
    aggregated = {
        "company_name": company_data["company_name"],
        "themes_data": defaultdict(list),
        "financial_metrics": defaultdict(dict),
        "time_period": year
    }
    
    for doc in company_data["documents"]:
        if doc["analysis"] and "aggregated_verbatim_quotes" in doc["analysis"]:
            for theme, quotes in doc["analysis"]["aggregated_verbatim_quotes"].items():
                aggregated["themes_data"][theme].extend(quotes)
        if doc["analysis"] and "financial_metrics" in doc["analysis"]:
            for metric, value in doc["analysis"]["financial_metrics"].items():
                aggregated["financial_metrics"][metric] = value
    
    return aggregated

# Main Application
def main():
    st.title("Company Analysis Comparator")

    # Year Selection
    selected_year = st.selectbox(
        "Select Year",
        [str(y) for y in range(2018, datetime.now().year + 1)],
        index=3
    )

    # Session State Initialization
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = {
            "Accenture": {"company_name": "Accenture", "documents": []},
            "Other Company": {"company_name": "Other Company", "documents": []}
        }

    # Upload Interface
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📂 Other Company Documents")
        with st.expander("📎 Upload Transcript PDFs"):
            other_transcripts = st.file_uploader(
                "Other Transcripts", type=["pdf"], accept_multiple_files=True, key="other_transcripts"
            )
        with st.expander("📎 Upload Factsheet PDFs"):
            other_factsheets = st.file_uploader(
                "Other Factsheets", type=["pdf"], accept_multiple_files=True, key="other_factsheets"
            )
        with st.expander("📎 Upload Financial PDFs"):
            other_financials = st.file_uploader(
                "Other Financials", type=["pdf"], accept_multiple_files=True, key="other_financials"
            )

    with col2:
        st.subheader("📂 Accenture Documents")
        with st.expander("📎 Upload Transcript PDFs"):
            accenture_transcripts = st.file_uploader(
                "Accenture Transcripts", type=["pdf"], accept_multiple_files=True, key="accenture_transcripts"
            )
        with st.expander("📎 Upload Factsheet PDFs"):
            accenture_factsheets = st.file_uploader(
                "Accenture Factsheets", type=["pdf"], accept_multiple_files=True, key="accenture_factsheets"
            )
        with st.expander("📎 Upload Financial PDFs"):
            accenture_financials = st.file_uploader(
                "Accenture Financials", type=["pdf"], accept_multiple_files=True, key="accenture_financials"
            )

    # Process and Compare
    if st.button("🔍 Process & Compare Documents"):
        companies = {
            "Accenture": {
                "Transcripts": accenture_transcripts,
                "Factsheets": accenture_factsheets,
                "Financials": accenture_financials
            },
            "Other Company": {
                "Transcripts": other_transcripts,
                "Factsheets": other_factsheets,
                "Financials": other_financials
            }
        }

        for company, doc_types in companies.items():
            st.session_state.processed_data[company]["documents"] = []
            total_files = sum(len(files) for files in doc_types.values() if files)
            progress_bar = st.progress(0)
            processed_files = 0

            for doc_type, files in doc_types.items():
                if files:
                    for file in files:
                        with st.spinner(f"Processing {company} {doc_type}..."):
                            file_content = file.getvalue()
                            analysis = load_or_process_document(file_content, company, selected_year, doc_type)
                            if analysis:
                                st.session_state.processed_data[company]["documents"].append({
                                    "type": doc_type,
                                    "analysis": analysis
                                })
                        processed_files += 1
                        progress_bar.progress(processed_files / total_files)

        # Aggregate and Compare
        accenture_data = aggregate_company_data(st.session_state.processed_data["Accenture"], selected_year)
        other_data = aggregate_company_data(st.session_state.processed_data["Other Company"], selected_year)

        try:
            comparator = LLMThemeComparator(API_KEY, ANALYSIS_MODEL)
            theme_comparison = comparator.compare_companies(accenture_data, other_data)

            # Display Results
            st.header("📊 Comparison Results")

            st.subheader("🔍 Theme-based Analysis")
            for theme, comparison in theme_comparison["comparisons"].items():
                with st.expander(f"{theme} Comparison"):
                    st.markdown(comparison)

            st.subheader("💰 Financial Metrics Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Accenture")
                for metric, value in accenture_data["financial_metrics"].items():
                    st.write(f"**{metric}**: {value}")
            with col2:
                st.markdown("### Other Company")
                for metric, value in other_data["financial_metrics"].items():
                    st.write(f"**{metric}**: {value}")

            st.subheader("📈 Strategic Insights")
            insights = comparator.compare_companies(
                accenture_data["financial_metrics"],
                other_data["financial_metrics"]
            )
            st.markdown(insights)
        except Exception as e:
            st.error(f"Comparison failed: {str(e)}")

if __name__ == "__main__":
    main()