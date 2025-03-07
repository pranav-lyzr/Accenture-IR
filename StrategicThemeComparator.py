
# Import required libraries
import json
import os
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict
import numpy as np
from IPython.display import Markdown, display, HTML, Image
import shutil
# from openai import OpenAI

class StrategicThemeComparator:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        """
        Initialize the strategic theme comparator with LLM enhancement

        Parameters:
        -----------
        api_key : str
            OpenAI API key
        model : str
            Model to use ("gpt-3.5-turbo" or "gpt-4o-mini")
        """
        # Predefined themes list
        self.themes = [
            "Macro environment", "Pricing", "Margins", "Bookings/Large Deals",
            "Discretionary/Small Deals", "People", "Cloud", "Security", "Gen AI",
            "M&A", "Investments", "Partnerships", "Technology Budget",
            "Product/IP/Assets", "Talent/Training", "Clients", "Awards/Recognition"
        ]

        # Initialize OpenAI client
        try:
            # self.client = OpenAI(api_key=api_key)
            self.model = model
            print(f"Strategic Theme Comparator initialized with {model}")
        except Exception as e:
            raise Exception(f"Error initializing OpenAI client: {e}")
        
        self.api_url = "https://agent-dev.test.studio.lyzr.ai/v3/inference/chat/"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": "sk-default-PPcvzcCe4cJRRP8JkEXnT51woYJUXzMZ"
        }
        self.user_id = "pranav@lyzr.ai"

    def _call_lyzr_api(self, agent_id: str, session_id: str, message: str) -> str:
        """Helper function to call Lyzr API"""
        payload = {
            "user_id": self.user_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "message": message
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "No response received.")
        except requests.exceptions.RequestException as e:
            return f"Error calling Lyzr API: {str(e)}"


    def load_transcript_data(self, file_path):
        """
        Load transcript data using LLM-based extraction

        Args:
            file_path (str): Path to the file containing transcript data

        Returns:
            dict: A dictionary containing company name, filename, and themes data
        """
        try:
            # Open and read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read().strip()

            # Check if file is empty
            if not file_content:
                raise ValueError(f"File is empty: {file_path}")

            # Try JSON parsing first
            try:
                data = json.loads(file_content)
                return self._process_json_data(data, file_path)
            except json.JSONDecodeError:
                # If not JSON, use LLM for Markdown extraction
                return self._extract_markdown_with_llm(file_content, file_path)

        except (IOError, Exception) as e:
            # Handle file reading or parsing errors
            print(f"Error reading file {file_path}: {e}")
            return None

    def _extract_markdown_with_llm(self, file_content, file_path, model="gpt-3.5-turbo"):
        """
        Extract structured data from Markdown using LLM
        """
        # Prompt for LLM-based extraction
        prompt = f"""Extract structured data from this earnings call transcript markdown document.

Input Document:
{file_content[:10000]}  # Limit initial context to prevent token overflow

Please extract the following structured information:

1. Company Name: Identify the primary company discussed in the document
2. Themes: Extract all distinct themes mentioned
3. Theme Details: For each theme, capture:
   - Number of mentions
   - Key verbatim quotes (with page numbers)
   - Any strategic insights or metrics

Output Format (STRICT JSON):
{{
    "company_name": "Company Name",
    "filename": "original_filename.md",
    "themes_data": {{
        "Theme Name 1": [
            {{
                "quote": "Exact verbatim quote",
                "page": 1,
                "strategic_context": "Brief strategic insight"
            }},
            ...
        ],
        "Theme Name 2": [
            ...
        ]
    }}
}}

Guidelines:
- Be precise in extracting quotes
- Preserve exact wording
- Include page numbers
- Only include themes with at least one substantive quote
- Capture strategic context where possible
"""

        # Retry mechanism for API call
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # response = self.client.chat.completions.create(
                #     model=model,
                #     messages=[
                #         {
                #             "role": "system",
                #             "content": "You are an expert financial analyst extracting structured data from earnings call transcripts."
                #         },
                #         {
                #             "role": "user",
                #             "content": prompt
                #         }
                #     ],
                #     response_format={"type": "json_object"},
                #     temperature=0.3,
                #     max_tokens=4000
                # )

                # # Parse the response
                # result = json.loads(response.choices[0].message.content)

                json_str = self._call_lyzr_api(
                    agent_id="67c86b15be1fc2af4eb4027e",
                    session_id="67c86b15be1fc2af4eb4027e",
                    message="You are an expert financial analyst extracting structured data from earnings call transcripts." + prompt
                )
                
                result = json_str.replace('```json', '').replace('```', '')
                json.loads(result)
            
                # Validate the extracted data
                if not result.get("company_name"):
                    # Fallback to filename-based company name
                    result["company_name"] = os.path.basename(file_path).split('_')[0]

                # Ensure filename is set
                result["filename"] = os.path.basename(file_path)

                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"LLM extraction attempt {attempt + 1} failed: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Fallback to manual extraction if all attempts fail
                    return self._fallback_markdown_extraction(file_content, file_path)

    def _fallback_markdown_extraction(self, file_content, file_path):
        """
        Fallback method for extracting data from Markdown if LLM fails
        """
        # Basic regex-based extraction
        # Try to extract company name from first heading
        company_match = re.search(r'#\s*([^-\n]+)', file_content)
        company_name = company_match.group(1).strip() if company_match else os.path.basename(file_path).split('_')[0]

        # Extract themes and quotes using regex
        themes_data = {}
        theme_pattern = re.compile(r'# (.*?)\n\n## Summary\nThis theme appears in (\d+) verbatim quotes throughout the.*?\n\n(.*?)(?=\n# |$)', re.DOTALL)

        for match in theme_pattern.finditer(file_content):
            theme = match.group(1).strip()
            quotes_content = match.group(3).strip()

            # Extract verbatim quotes
            quote_pattern = re.compile(r'- "([^"]+)" \(Page (\d+)\)')
            quotes = [
                {
                    "quote": quote.group(1),
                    "page": int(quote.group(2)),
                    "strategic_context": ""  # No strategic context in fallback method
                }
                for quote in quote_pattern.finditer(quotes_content)
            ]

            # Only add themes with quotes
            if quotes:
                themes_data[theme] = quotes

        return {
            "company_name": company_name,
            "filename": os.path.basename(file_path),
            "themes_data": themes_data
        }

    def _process_json_data(self, data, file_path):
        """
        Process data from a JSON file
        """
        # Try to extract company name from metadata
        company_name = data.get("metadata", {}).get("company_name")

        # If not found in metadata, extract from filename
        if not company_name:
            filename = os.path.basename(file_path).lower()

            # Dictionary of company name mappings
            company_mappings = {
                "accenture": "Accenture",
                "wipro": "Wipro Limited",
                "tcs": "Tata Consultancy Services",
                "tata": "Tata Consultancy Services",
                "infosys": "Infosys",
                "cognizant": "Cognizant",
                "hcl": "HCL Technologies"
            }

            # Find matching company name
            for keyword, company in company_mappings.items():
                if keyword in filename:
                    company_name = company
                    break

            # If no specific company detected, use generic name
            if not company_name:
                company_name = "Company " + filename.split('_')[0]

        # Extract aggregated quotes by theme
        themes_data = {
            theme: [
                {
                    "quote": quote,
                    "page": 1,  # Default page number if not specified
                    "strategic_context": ""
                } for quote in quotes
            ]
            for theme, quotes in data.get("aggregated_verbatim_quotes", {}).items()
            if quotes  # Only include themes with non-empty quotes
        }

        return {
            "company_name": company_name,
            "filename": os.path.basename(file_path),
            "themes_data": themes_data
        }

    def calculate_llm_similarity(self, theme, company1_name, company2_name, company1_quotes, company2_quotes):
        """
        Use LLM to calculate a similarity score with strategic reasoning
        """
        if not company1_quotes or not company2_quotes:
            return 0.0, "One company has no quotes on this theme"

        # Prepare quotes for prompt (all quotes to preserve context)
        c1_quotes_str = "\n".join([f"- \"{q['quote']}\" (Page {q['page']})" for q in company1_quotes])
        c2_quotes_str = "\n".join([f"- \"{q['quote']}\" (Page {q['page']})" for q in company2_quotes])

        prompt = f"""As a strategy analyst with 15+ years experience in technology consulting, analyze the similarity between how {company1_name} and {company2_name} approach the "{theme}" theme in their earnings calls.

                {company1_name} quotes:
                {c1_quotes_str}

                {company2_name} quotes:
                {c2_quotes_str}

                Calculate a strategic similarity score between 0.0 and 1.0 where:
                - 0.0 means fundamentally different strategic approaches, metrics, priorities, or market positions
                - 0.5 means similar high-level strategy but different implementation tactics, metrics, or emphases
                - 1.0 means identical strategic positioning, priorities, metrics, and execution approach

                Consider:
                1. Strategic focus areas and priorities
                2. Investment levels and resource allocation
                3. Market positioning and competitive differentiation
                4. Metrics used to measure success
                5. Future outlook and growth strategy

                Format your response as JSON:
                {{
                    "similarity_score": X.X,
                    "strategic_reasoning": "Detailed strategic assessment of similarities and differences (3-4 sentences)"
                }}
                """

        # Call API with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # response = self.client.chat.completions.create(
                #     model=self.model,
                #     messages=[
                #         {"role": "system", "content": "You are a senior strategy analyst with 15+ years experience in technology consulting, specialized in comparing enterprise technology companies."},
                #         {"role": "user", "content": prompt}
                #     ],
                #     temperature=0.3,
                #     response_format={"type": "json_object"},
                #     max_tokens=500
                # )

                # result = json.loads(response.choices[0].message.content)

                json_str = self._call_lyzr_api(
                    agent_id="67c86b15be1fc2af4eb4027e",
                    session_id="67c86b15be1fc2af4eb4027e",
                    message="You are an expert financial analyst extracting structured data from earnings call transcripts." + prompt
                )
                
                result = json_str.replace('```json', '').replace('```', '')
                json.loads(result)

                score = float(result.get("similarity_score", 0.5))
                reasoning = result.get("strategic_reasoning", "No strategic assessment provided")

                return score, reasoning
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"API error, retrying Stratigic 358 ({attempt+1}/{max_retries}): {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"API error after {max_retries} attempts: {e}")
                    # Return a placeholder score with error message
                    return 0.5, f"Error calculating similarity: {str(e)}"

    def generate_strategic_comparison(self, theme, company1_name, company2_name, company1_quotes, company2_quotes, similarity_score, reasoning):
        """
        Generate a comprehensive strategic comparison using LLM

        Args:
            theme (str): Strategic theme being analyzed
            company1_name (str): Name of first company
            company2_name (str): Name of second company
            company1_quotes (list): Quotes from first company
            company2_quotes (list): Quotes from second company
            similarity_score (float): Calculated similarity score
            reasoning (str): Strategic reasoning for similarity

        Returns:
            str: Markdown-formatted strategic comparison
        """
        # Preserve all verbatim quotes with page references
        c1_quotes_str = "\n".join([f"- \"{q['quote']}\" (Page {q['page']})" for q in company1_quotes])
        c2_quotes_str = "\n".join([f"- \"{q['quote']}\" (Page {q['page']})" for q in company2_quotes])

        if not c1_quotes_str:
            c1_quotes_str = "No quotes available from this company on this theme."

        if not c2_quotes_str:
            c2_quotes_str = "No quotes available from this company on this theme."

        prompt = f"""As a strategy analyst with 15+ years of experience in the technology consulting sector, create a comprehensive strategic comparison of how {company1_name} and {company2_name} approach the "{theme}" theme.

            {company1_name} verbatim quotes:
            {c1_quotes_str}

            {company2_name} verbatim quotes:
            {c2_quotes_str}

            Similarity Score: {similarity_score:.2f}
            Strategic Assessment: {reasoning}

            Create a detailed strategic analysis that ALWAYS maintains and references the exact verbatim quotes with their page numbers. Your analysis must include:

            1. Strategic positioning of each company on this theme
            2. Key differentiators and competitive advantages
            3. Investment priorities and resource allocation differences
            4. Implications for market share and customer acquisition
            5. Long-term strategic implications for each company

            Format your analysis in markdown with these sections:
            # {theme} Strategic Comparison: {company1_name} vs {company2_name}

            ## Strategic Similarity Score: {similarity_score:.2f}
            [Include your strategic assessment]

            ## {company1_name}'s Strategic Positioning
            [Analyze their approach with specific verbatim quote references and page numbers]

            ## {company2_name}'s Strategic Positioning
            [Analyze their approach with specific verbatim quote references and page numbers]

            ## Strategic Differentiators
            [Identify key strategic differences with references to specific metrics, statements, or positioning from the verbatim quotes]

            ## Competitive Implications
            [Analyze what these differences mean for competitive positioning, market share, and long-term success]

            IMPORTANT REQUIREMENTS:
            1. ALWAYS use verbatim quotes with page numbers as evidence for your analysis
            2. Don't summarize or paraphrase quotes - use them exactly as provided
            3. Ground ALL strategic insights in specific verbatim statements
            4. Focus on strategic implications rather than tactical details
            5. If data is limited for either company, explicitly acknowledge this limitation
            """

        # Call API with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # response = self.client.chat.completions.create(
                #     model=self.model,
                #     messages=[
                #         {
                #             "role": "system",
                #             "content": "You are a senior strategy analyst with 15+ years of experience in technology consulting, specialized in analyzing enterprise technology companies. You always ground your analysis in specific verbatim quotes with their source references."
                #         },
                #         {
                #             "role": "user",
                #             "content": prompt
                #         }
                #     ],
                #     temperature=0.4,
                #     max_tokens=2000
                # )

                # analysis = response.choices[0].message.content
                json_str = self._call_lyzr_api(
                    agent_id="67c86b15be1fc2af4eb4027e",
                    session_id="67c86b15be1fc2af4eb4027e",
                    message="You are an expert financial analyst extracting structured data from earnings call transcripts." + prompt
                )
                
                analysis = json_str.replace('```json', '').replace('```', '')
                json.loads(analysis)

                return analysis

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"API error, retrying StrategicTHEME ({attempt+1}/{max_retries}): {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Fallback to manual comparison if LLM fails
                    return self._generate_fallback_comparison(
                        theme, company1_name, company2_name,
                        company1_quotes, company2_quotes,
                        similarity_score, reasoning
                    )

    def _generate_fallback_comparison(self, theme, company1_name, company2_name,
                                      company1_quotes, company2_quotes,
                                      similarity_score, reasoning):
        """
        Generate a basic comparison without LLM when API fails

        Args:
            theme (str): Strategic theme being analyzed
            company1_name (str): Name of first company
            company2_name (str): Name of second company
            company1_quotes (list): Quotes from first company
            company2_quotes (list): Quotes from second company
            similarity_score (float): Calculated similarity score
            reasoning (str): Strategic reasoning for similarity

        Returns:
            str: Markdown-formatted basic comparison
        """
        comparison_text = f"# {theme} Strategic Comparison: {company1_name} vs {company2_name}\n\n"
        comparison_text += f"## Strategic Similarity Score: {similarity_score:.2f}\n"
        comparison_text += f"{reasoning}\n\n"

        # Add company 1 section with all verbatim quotes
        comparison_text += f"## {company1_name}'s Strategic Positioning\n"
        if company1_quotes:
            for quote in company1_quotes:
                comparison_text += f"- \"{quote['quote']}\" (Page {quote['page']})\n"
        else:
            comparison_text += f"No quotes found from {company1_name} on this theme.\n"

        comparison_text += "\n"

        # Add company 2 section with all verbatim quotes
        comparison_text += f"## {company2_name}'s Strategic Positioning\n"
        if company2_quotes:
            for quote in company2_quotes:
                comparison_text += f"- \"{quote['quote']}\" (Page {quote['page']})\n"
        else:
            comparison_text += f"No quotes found from {company2_name} on this theme.\n"

        comparison_text += "\n"

        # Add strategic differentiators section
        comparison_text += "## Strategic Differentiators\n"
        if not company1_quotes and not company2_quotes:
            comparison_text += "Neither company discussed this theme sufficiently for strategic analysis.\n"
        elif not company1_quotes:
            comparison_text += f"Only {company2_name} has articulated a strategic position on this theme.\n"
        elif not company2_quotes:
            comparison_text += f"Only {company1_name} has articulated a strategic position on this theme.\n"
        else:
            comparison_text += "Strategic differentiators require further analysis based on the verbatim quotes above.\n"

        comparison_text += "\n## Competitive Implications\n"
        comparison_text += "Further analysis required to determine competitive implications based on the verbatim evidence.\n"

        return comparison_text

    def compare_companies(self, company1_data, company2_data):
        """
        Compare companies based on their transcript data with strategic focus

        Args:
            company1_data (dict): Data for first company
            company2_data (dict): Data for second company

        Returns:
            dict: Comparison results including comparisons and summary
        """
        company1_name = company1_data["company_name"]
        company2_name = company2_data["company_name"]

        print(f"Strategically comparing {company1_name} with {company2_name}")

        # Find all unique themes
        all_themes = set(company1_data["themes_data"].keys()).union(
            set(company2_data["themes_data"].keys())
        )

        comparisons = {}
        summary_data = []

        # Analyze each theme
        for theme in all_themes:
            print(f"Strategically analyzing theme: {theme}")
            company1_quotes = company1_data["themes_data"].get(theme, [])
            company2_quotes = company2_data["themes_data"].get(theme, [])

            # Calculate similarity score using LLM
            similarity_score, reasoning = self.calculate_llm_similarity(
                theme, company1_name, company2_name,
                company1_quotes, company2_quotes
            )

            # Generate strategic comparison
            comparison = self.generate_strategic_comparison(
                theme, company1_name, company2_name,
                company1_quotes, company2_quotes,
                similarity_score, reasoning
            )

            comparisons[theme] = comparison

            # Add to summary data
            summary_data.append({
                "Theme": theme,
                "Strategic Similarity": similarity_score,
                f"{company1_name} Mentions": len(company1_quotes),
                f"{company2_name} Mentions": len(company2_quotes),
                "Strategic Assessment": reasoning
            })

        # Create summary dataframe
        summary_df = pd.DataFrame(summary_data)

        return {
            "comparisons": comparisons,
            "summary_df": summary_df
        }


def create_visualizations(summary_df, company1_name, company2_name, output_dir="."):
    """
    Create visualizations comparing the companies

    Args:
        summary_df (pandas.DataFrame): Summary of strategic comparisons
        company1_name (str): Name of first company
        company2_name (str): Name of second company
        output_dir (str, optional): Directory to save visualizations

    Returns:
        pandas.DataFrame: Sorted summary dataframe
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Sort by similarity score
    sorted_df = summary_df.sort_values("Strategic Similarity", ascending=False)

    # 1. Create similarity score visualization
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(
        x="Strategic Similarity",
        y="Theme",
        data=sorted_df,
        palette="viridis"
    )
    plt.title(f"Strategic Theme Similarity: {company1_name} vs {company2_name}", fontsize=16)
    plt.xlabel("Strategic Similarity Score (0 = Different, 1 = Similar)", fontsize=12)
    plt.ylabel("Theme", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strategic_similarity_scores.png"), dpi=300)
    plt.close()

    # 2. Create mentions comparison visualization
    # Create data for grouped bar chart
    mentions_data = []
    for _, row in sorted_df.iterrows():
        mentions_data.append({
            "Theme": row["Theme"],
            "Company": company1_name,
            "Mentions": row[f"{company1_name} Mentions"]
        })
        mentions_data.append({
            "Theme": row["Theme"],
            "Company": company2_name,
            "Mentions": row[f"{company2_name} Mentions"]
        })

    mentions_df = pd.DataFrame(mentions_data)

    # Sort by themes in similarity order
    theme_order = sorted_df["Theme"].tolist()
    mentions_df["Theme"] = pd.Categorical(mentions_df["Theme"], categories=theme_order, ordered=True)

    # Create the grouped bar chart
    plt.figure(figsize=(14, 10))
    sns.barplot(
        x="Mentions",
        y="Theme",
        hue="Company",
        data=mentions_df.sort_values("Theme"),
        palette=["#1f77b4", "#ff7f0e"]
    )

    plt.title(f"Theme Strategic Focus: {company1_name} vs {company2_name}", fontsize=16)
    plt.xlabel("Number of Strategic Mentions", fontsize=12)
    plt.ylabel("Theme", fontsize=12)
    plt.legend(title="Company")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "theme_strategic_focus.png"), dpi=300)
    plt.close()

    return sorted_df

def run_strategic_comparison(source_file, target_file, api_key, model="gpt-3.5-turbo",
                             source_company=None, target_company=None):
    """
    Run the strategic comparison between two companies

    Args:
        source_file (str): Path to the source company file
        target_file (str): Path to the target company file
        api_key (str): OpenAI API key
        model (str, optional): LLM model to use
        source_company (str, optional): Override source company name
        target_company (str, optional): Override target company name

    Returns:
        dict: Comparison results with output directories and analysis
    """
    # Initialize comparator
    comparator = StrategicThemeComparator(api_key=api_key, model=model)

    # Load transcript data
    print("Loading source company data...")
    source_data = comparator.load_transcript_data(source_file)

    print("Loading target company data...")
    target_data = comparator.load_transcript_data(target_file)

    # Override company names if provided
    if source_company:
        source_data["company_name"] = source_company

    if target_company:
        target_data["company_name"] = target_company

    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"{source_data['company_name'].replace(' ', '_')}_vs_{target_data['company_name'].replace(' ', '_')}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # print(f"Comparing {source_data['company_name']} (source) with {target_data['company_name']} (target)...")

    # Run comparison
    results = comparator.compare_companies(source_data, target_data)

    # Create visualizations
    print("Creating strategic visualizations...")
    sorted_df = create_visualizations(
        results["summary_df"],
        source_data["company_name"],
        target_data["company_name"],
        output_dir
    )

    # Save summary to CSV
    summary_file = os.path.join(output_dir, "strategic_comparison_summary.csv")
    sorted_df.to_csv(summary_file, index=False)
    print(f"Saved strategic summary to: {summary_file}")

    for theme, comparison in results["comparisons"].items():
        theme_file = os.path.join(output_dir, f"{theme.replace('/', '_')}_strategic_comparison.md")
        with open(theme_file, 'w', encoding='utf-8') as f:
            f.write(comparison)
        print(f"Saved strategic comparison for {theme} to: {theme_file}")

    # Create combined file with all strategic comparisons
    combined_file = os.path.join(output_dir, "all_strategic_comparisons.md")
    with open(combined_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write(f"# Strategic Theme Comparisons: {source_data['company_name']} vs {target_data['company_name']}\n\n")
        f.write(f"## Comparison Overview\n")
        f.write(f"- **Source Company**: {source_data['company_name']}\n")
        f.write(f"- **Target Company**: {target_data['company_name']}\n")
        f.write(f"- **Comparison Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Total Themes Analyzed**: {len(results['comparisons'])}\n\n")

        # Add summary table
        f.write("## Strategic Similarity Summary\n\n")
        f.write("| Theme | Similarity Score | Strategic Assessment |\n")
        f.write("|-------|-----------------|----------------------|\n")
        for _, row in sorted_df.iterrows():
            f.write(f"| {row['Theme']} | {row['Strategic Similarity']:.2f} | {row['Strategic Assessment']} |\n")

        f.write("\n---\n\n")

        # Add detailed comparisons for each theme
        for theme in sorted_df["Theme"]:
            if theme in results["comparisons"]:
                f.write(results["comparisons"][theme])
                f.write("\n\n---\n\n")

    print(f"Saved all strategic comparisons to: {combined_file}")

    # Create README with instructions and overview
    readme_file = os.path.join(output_dir, "README.md")
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(f"# Strategic Comparison: {source_data['company_name']} vs {target_data['company_name']}\n\n")
        f.write("## Overview\n")
        f.write("This directory contains a comprehensive strategic comparison between two companies based on their earnings call transcripts.\n\n")
        f.write("## Files in this Directory\n")
        f.write("- `strategic_comparison_summary.csv`: Quantitative summary of strategic similarities\n")
        f.write("- `strategic_similarity_scores.png`: Visualization of theme similarity scores\n")
        f.write("- `theme_strategic_focus.png`: Comparison of theme mentions\n")
        f.write("- `all_strategic_comparisons.md`: Detailed markdown report of all theme comparisons\n")
        f.write("- Individual theme comparison markdown files\n\n")
        f.write("## Methodology\n")
        f.write("- Themes were extracted from earnings call transcripts\n")
        f.write("- Strategic similarity calculated using AI-powered analysis\n")
        f.write("- Comparisons ground insights in verbatim quotes\n")

    print(f"Saved README to: {readme_file}")

    return {
        "output_dir": output_dir,
        "results": results,
        "source_data": source_data,
        "target_data": target_data,
        "summary_file": summary_file,
        "combined_file": combined_file,
        "readme_file": readme_file
    }
