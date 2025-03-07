import streamlit as st
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
import re
import time
import sys
import io

# ------------------ Tactical Analysis Imports ------------------
from PDFAnalyzer import PDFAnalyzer
from LLMThemeComparator import LLMThemeComparator

# ------------------ Configuration ------------------
UPLOAD_FOLDER = "out"
TAC_ANALYSIS_MODEL = "gpt-4"  # For tactical analysis
API_KEY = "your-api-key"       # Replace with your actual API key

# ------------------ Custom UI Styles ------------------
st.markdown(
    """
    <style>
    /* Reduce side margins */
    .reportview-container .main .block-container{
        padding-left: 40px;
        padding-right: 40px;
        max-width: 100% !important;
    }
    /* Smaller heading sizes */
    h1, h2, h3, h4, h5, h6 {
        font-size: 1.2rem !important;
    }
    .st-emotion-cache-mtjnbi {
    width: 100%;
    padding: 6rem 5rem 10rem;
    max-width: 2000px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Custom Streamlit Logger ------------------
class StreamlitLogger(io.StringIO):
    """
    A custom logger that writes print messages into a Streamlit placeholder.
    The logger updates the placeholder with the most recent message.
    """
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.buffer = ""

    def write(self, s):
        # Append incoming string to our buffer
        self.buffer += s
        # If a newline is found, update the placeholder with the latest complete line.
        if "\n" in self.buffer:
            lines = self.buffer.splitlines()
            # Display the last line only to keep the UI uncluttered.
            self.placeholder.text(lines[-1])
        else:
            self.placeholder.text(self.buffer)
        return len(s)

    def flush(self):
        pass

# ------------------ Utility Functions for Tactical Analysis ------------------

def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

def save_uploaded_file(file_content, company, year, quarter, doc_type, file_name):
    file_path = Path(UPLOAD_FOLDER) / company / year / quarter / doc_type / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path

def process_pdf(file_path, company, year, quarter):
    analyzer = PDFAnalyzer(API_KEY)
    # Create a placeholder for progress messages.
    progress_placeholder = st.empty()
    original_stdout = sys.stdout
    sys.stdout = StreamlitLogger(progress_placeholder)
    try:
        # These methods print their progress messages and will now show in the UI.
        analysis = analyzer.analyze_pdf(str(file_path))
        complete_analysis = analyzer.generate_complete_analysis(analysis)
        # Add quarter info into the metadata
        if "metadata" not in complete_analysis:
            complete_analysis["metadata"] = {}
        complete_analysis["metadata"]["quarter"] = quarter
    finally:
        sys.stdout = original_stdout
        progress_placeholder.empty()  # Clear progress messages once done.
    return complete_analysis

def load_or_process_document(file_content, company, year, quarter, doc_type, original_filename):
    file_hash = get_file_hash(file_content)
    base_name, _ = os.path.splitext(original_filename)
    final_analysis_filename = f"{base_name}_{file_hash}_analysis.json"
    analysis_path = Path(UPLOAD_FOLDER) / company / year / quarter / "analysis" / final_analysis_filename

    # Check if analysis already exists
    if analysis_path.exists():
        with open(analysis_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Save the PDF file using a similar naming convention
    pdf_filename = f"{base_name}_{file_hash}.pdf"
    file_path = save_uploaded_file(file_content, company, year, quarter, doc_type, pdf_filename)

    # Run heavy analysis with progress redirected to UI
    complete_analysis = process_pdf(file_path, company, year, quarter)

    # Save the analysis for future reuse
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(complete_analysis, f, indent=2)
    return complete_analysis

def aggregate_company_data(company_data, year, quarter, default_company_name):
    """
    Aggregates data from all processed documents that were uploaded for the specified quarter.
    """
    aggregated = {
        "company_name": default_company_name,
        "themes_data": {},
        "financial_metrics": {},
        "time_period": year,
        "quarter": quarter
    }
    for doc in company_data.get("documents", []):
        analysis = doc.get("analysis", {})
        # Only aggregate if the document's quarter matches the selected quarter
        if analysis.get("metadata", {}).get("quarter", "").upper() != quarter.upper():
            continue

        if "metadata" in analysis and analysis["metadata"].get("company_name"):
            aggregated["company_name"] = analysis["metadata"]["company_name"]
        if "aggregated_verbatim_quotes" in analysis:
            for theme, quotes in analysis["aggregated_verbatim_quotes"].items():
                aggregated["themes_data"].setdefault(theme, []).extend(quotes)
        elif "verbatim_summary" in analysis:
            summary_text = analysis["verbatim_summary"]
            sections = re.split(r"\n##\s*", summary_text)
            for section in sections[1:]:
                parts = section.split("\n", 1)
                if len(parts) == 2:
                    theme = parts[0].strip()
                    content = parts[1].strip()
                    aggregated["themes_data"].setdefault(theme, []).append({
                        "quote": content,
                        "page": "N/A"
                    })
        if "financial_metrics" in analysis:
            aggregated["financial_metrics"].update(analysis["financial_metrics"])
    return aggregated

# ------------------ Main Application ------------------

def main():
    st.title("Company Tactical Comparison (Quarter-by-Quarter)")
    st.write("Upload PDF documents for each company. Files are segregated by year and quarter. "
             "Analysis will only run if both companies have data for the same quarter.")

    selected_year = st.selectbox(
        "Select Year",
        [str(y) for y in range(2018, datetime.now().year + 1)],
        index=3
    )
    selected_quarter = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])

    # Initialize session state for tactical data
    if "tactical_data" not in st.session_state:
        st.session_state.tactical_data = {
            "Accenture": {"documents": []},
            "Other Company": {"documents": []}
        }

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

    if st.button("Process & Compare Tactical Documents"):
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
            st.session_state.tactical_data[company]["documents"] = []
            for doc_type, files in doc_types.items():
                if files:
                    for file in files:
                        # Show a spinner with the current file and document type being processed.
                        with st.spinner(f"Processing {company} {doc_type} for {selected_quarter}..."):
                            file_content = file.getvalue()
                            analysis = load_or_process_document(
                                file_content, company, selected_year, selected_quarter, doc_type, file.name
                            )
                            st.session_state.tactical_data[company]["documents"].append({
                                "type": doc_type,
                                "analysis": analysis
                            })

        # Ensure both companies have at least one document for the selected quarter.
        if (len(st.session_state.tactical_data["Accenture"]["documents"]) == 0 or
            len(st.session_state.tactical_data["Other Company"]["documents"]) == 0):
            st.error("Both companies must have data for the selected quarter for analysis to proceed.")
            return

        accenture_data = aggregate_company_data(
            st.session_state.tactical_data["Accenture"],
            selected_year,
            selected_quarter,
            "Accenture"
        )
        other_data = aggregate_company_data(
            st.session_state.tactical_data["Other Company"],
            selected_year,
            selected_quarter,
            "Other Company"
        )

        # st.write("Aggregated Accenture Data:", accenture_data)
        # st.write("Aggregated Other Company Data:", other_data)

        # Redirect comparator logs to the UI as well.
        comparator = LLMThemeComparator(API_KEY, TAC_ANALYSIS_MODEL)
        comparator_placeholder = st.empty()
        original_stdout = sys.stdout
        sys.stdout = StreamlitLogger(comparator_placeholder)
        try:
            comparison_results = comparator.compare_companies(accenture_data, other_data)
        finally:
            sys.stdout = original_stdout
            comparator_placeholder.empty()

        st.header("Tactical Theme-based Comparison")
        for theme, comparison in comparison_results["comparisons"].items():
            with st.expander(f"{theme} Comparison"):
                st.markdown(comparison)

        st.subheader("Financial Metrics Comparison")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("### Accenture Financial Metrics")
            if accenture_data["financial_metrics"]:
                for metric, value in accenture_data["financial_metrics"].items():
                    st.write(f"**{metric}**: {value}")
            else:
                st.write("No financial metrics data available.")
        with col_f2:
            st.markdown("### Other Company Financial Metrics")
            if other_data["financial_metrics"]:
                for metric, value in other_data["financial_metrics"].items():
                    st.write(f"**{metric}**: {value}")
            else:
                st.write("No financial metrics data available.")

        # st.subheader("Strategic Insights (Tactical Data)")
        # insights = comparator.generate_financial_comparison(
        #     accenture_data["financial_metrics"],
        #     other_data["financial_metrics"]
        # )
        # st.markdown(insights)

        # st.header("Tactical Visualizations")
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # vis_output_dir = f"theme_comparisons_{timestamp}"
        # os.makedirs(vis_output_dir, exist_ok=True)
        # sorted_df = comparator.create_visualizations(
        #     comparison_results["summary_df"],
        #     accenture_data["company_name"],
        #     other_data["company_name"],
        #     vis_output_dir
        # )
        # sim_img_path = os.path.join(vis_output_dir, "theme_similarity_scores.png")
        # mentions_img_path = os.path.join(vis_output_dir, "theme_mentions_comparison.png")
        # if os.path.exists(sim_img_path):
        #     st.image(sim_img_path, caption="Theme Similarity Scores", use_column_width=True)
        # if os.path.exists(mentions_img_path):
        #     st.image(mentions_img_path, caption="Theme Mentions Comparison", use_column_width=True)
        # st.success("Tactical processing and comparison complete.")

if __name__ == "__main__":
    main()
