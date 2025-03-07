#Step #1 - Normal Text based Retrieval
import os
import time
import json
import requests
# from openai import OpenAI
import fitz  # PyMuPDF
from typing import Dict, Tuple, List
from datetime import datetime
from collections import defaultdict

class PDFAnalyzer:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        # self.client = OpenAI(api_key=api_key)
        self.model = model

        # Define categories with specific metrics to look for
        self.categories = {
            "Macro environment": ["market conditions", "economic factors", "industry trends"],
            "Pricing": ["pricing strategies", "rate changes", "price pressure"],
            "Margins": ["operating margin", "gross margin", "profit margin"],
            "Bookings/Large Deals": ["TCV", "deal size", "contract value"],
            "Discretionary/Small Deals": ["small deal performance", "discretionary spending"],
            "People": ["headcount", "attrition", "hiring"],
            "Cloud": ["cloud revenue", "cloud adoption", "cloud services"],
            "Security": ["security initiatives", "cybersecurity", "data protection"],
            "Gen AI": ["AI investments", "AI adoption", "AI capabilities"],
            "M&A": ["acquisitions", "mergers", "deal values"],
            "Investments": ["capital allocation", "investment focus", "spending"],
            "Partnerships": ["strategic alliances", "ecosystem partners"],
            "Technology Budget": ["tech spending", "IT budget", "digital investments"],
            "Product/IP/Assets": ["product portfolio", "IP development", "assets"],
            "Talent/Training": ["skill development", "training programs", "talent metrics"],
            "Clients": ["client wins", "customer base", "client satisfaction"],
            "Awards/Recognition": ["industry recognition", "awards", "rankings"]
        }

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


    def retry_with_backoff(self, func, *args, max_retries=5, initial_delay=1):
        """Execute a function with exponential backoff retry logic."""
        retries = 0
        delay = initial_delay

        while retries < max_retries:
            try:
                return func(*args)
            except Exception as e:
                retries += 1
                if retries == max_retries:
                    raise e

                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait_time = delay * (2 ** (retries - 1))  # Exponential backoff
                    print(f"\nRate limit hit. Waiting {wait_time} seconds before retry {retries}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e

    def generate_content_with_retry(self, prompt: str) -> str:
        """Generate content with retry logic using OpenAI API."""
        def _generate():
            # response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=[
            #         {"role": "system", "content": "You are a financial analyst expert at analyzing earnings call transcripts. Your main task is to extract EXACT VERBATIM quotes with page numbers. Include the company name and focus on extracting specific statements with exact numbers, metrics, and factual information."},
            #         {"role": "user", "content": prompt}
            #     ],
            #     temperature=0.2,  # Lower temperature for more precise extraction
            #     max_tokens=1500
            # )

            response = self._call_lyzr_api(
                agent_id="67c86b15be1fc2af4eb4027e",
                session_id="67c86b15be1fc2af4eb4027e",
                message="You are a financial analyst expert at analyzing earnings call transcripts. Your main task is to extract EXACT VERBATIM quotes with page numbers. Include the company name and focus on extracting specific statements with exact numbers, metrics, and factual information." + prompt
            )
            return response

        return self.retry_with_backoff(_generate)

    def analyze_page(self, text: str, page_num: int) -> Dict:
        """Analyze a single page of text with focus on extracting verbatim quotes."""
        try:
            analysis_prompt = f"""Analyze this page (Page {page_num}) of an earnings call transcript.

            PRIMARY TASK: Extract EXACT VERBATIM quotes for each relevant category. These must be word-for-word extracts from the text, not summaries or interpretations.

            Format the response as JSON:
            {{
                "page_number": {page_num},
                "company_name": "extracted from text if available",
                "categories": {{
                    "category_name": {{
                        "verbatim_quotes": [
                            {{
                                "quote": "EXACT word-for-word quote from the text",
                                "page": {page_num}
                            }}
                        ]
                    }}
                }}
            }}

            Categories to analyze:
            {json.dumps(self.categories, indent=2)}

            Text to analyze:
            {text}

            Rules:
            1. Only include categories that are actually mentioned on this page
            2. EXTRACT EXACT QUOTES - do not paraphrase or summarize
            3. Include the page number with each quote
            4. Focus on quotes containing specific numbers, percentages, and financial metrics
            5. Prioritize statements made by executives (CEO, CFO, etc.)
            6. Extract company name if mentioned on this page"""

            # Get analysis with retry logic
            response_text = self.generate_content_with_retry(analysis_prompt)

            try:
                # Extract JSON from response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    # Clean up formatting
                    json_str = json_str.replace('```json', '').replace('```', '')
                    return json.loads(json_str)

                return {
                    "page_number": page_num,
                    "categories": {},
                    "raw_text": response_text
                }

            except json.JSONDecodeError as e:
                print(f"JSON parsing error on page {page_num}: {str(e)}")
                print("Raw response:", response_text[:200])
                return {
                    "page_number": page_num,
                    "error": str(e),
                    "raw_text": response_text
                }

        except Exception as e:
            print(f"Error analyzing page {page_num}: {str(e)}")
            return {
                "page_number": page_num,
                "error": str(e)
            }

    def analyze_pdf(self, pdf_path: str) -> Dict:
        """Analyze PDF page by page with focus on extracting verbatim quotes."""
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            print(f"\nProcessing PDF: {pdf_path}")
            print(f"Total pages: {len(doc)}")

            # Generate base filename for results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"verbatim_analysis_{timestamp}"

            # Create directory for intermediate results
            os.makedirs("analysis_results", exist_ok=True)

            # Store results for each page
            page_results = []

            # Aggregated quotes by category
            all_verbatim_quotes = defaultdict(list)
            company_name = ""

            # Process each page
            for page_num in range(len(doc)):
                print(f"\nProcessing page {page_num + 1}/{len(doc)}...")

                # Get page text
                page = doc[page_num]
                text = page.get_text()

                # Clean up text - preserve paragraphs but remove extra whitespace
                text = '\n'.join([' '.join(line.split()) for line in text.split('\n') if line.strip()])

                # Analyze page
                page_analysis = self.analyze_page(text, page_num + 1)
                page_results.append(page_analysis)

                # Extract company name if available and not already set
                if not company_name and "company_name" in page_analysis and page_analysis["company_name"]:
                    company_name = page_analysis["company_name"]

                # Aggregate verbatim quotes from this page
                if "categories" in page_analysis:
                    for category, data in page_analysis["categories"].items():
                        if "verbatim_quotes" in data:
                            all_verbatim_quotes[category].extend(data["verbatim_quotes"])

                # Save intermediate results after each page
                intermediate_results = {
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "filename": pdf_path,
                        "company_name": company_name,
                        "total_pages": len(doc),
                        "processed_pages": len(page_results),
                        "model": self.model,
                        "analysis_version": "2.0"
                    },
                    "page_analysis": page_results,
                    "aggregated_verbatim_quotes": dict(all_verbatim_quotes)
                }

                intermediate_file = os.path.join("analysis_results", f"{base_filename}_intermediate.json")
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump(intermediate_results, f, indent=2, ensure_ascii=False)

            # Prepare final results
            final_results = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "filename": pdf_path,
                    "company_name": company_name,
                    "total_pages": len(doc),
                    "processed_pages": len(page_results),
                    "model": self.model,
                    "analysis_version": "2.0"
                },
                "page_analysis": page_results,
                "aggregated_verbatim_quotes": dict(all_verbatim_quotes)
            }

            # Save final results
            output_file = os.path.join("analysis_results", f"{base_filename}_final.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)

            print(f"\nVerbatim analysis saved to: {output_file}")
            return final_results

        except Exception as e:
            print(f"Error analyzing PDF: {str(e)}")
            return {"error": str(e)}

    def generate_verbatim_summary(self, results: Dict, save_to_file: bool = True) -> str:
        """Generate a summary based on verbatim quotes with page references."""
        try:
            company_name = results.get("metadata", {}).get("company_name", "")

            summary_prompt = f"""Create a comprehensive summary of this earnings call analysis based EXCLUSIVELY on the verbatim quotes.
            Each point must be backed by an exact quote with its page number.

            Company Name: {company_name}

            Verbatim quotes by category:
            {json.dumps(results['aggregated_verbatim_quotes'], indent=2)}

            Format the summary with:
            # {company_name} Earnings Call Analysis

            ## Macro environment
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

            ## Small deals, Discretionary, S&C
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

            ## Bookings/ Large Deals
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

            ## Pricing
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

            ## Margins
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)
            
             ## People
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Americas
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## EMEA
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## APAC
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## CMT
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## FS
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Products
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Resources
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## H&PS
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Managed Services/ Operations
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Industry X
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Song
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Cloud
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Data and AI / GenAI
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)


            Rules:
            1. Use ONLY the provided verbatim quotes - do not add any interpretations
            2. Include page references for ALL quotes
            3. Focus on the most significant quotes with specific numbers and metrics
            4. Organize by topic area, not by original category
            5. If possible, identify the time period covered by the earnings call (e.g., Q3 2024)"""

            # response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=[
            #         {"role": "system", "content": "You are a financial analyst expert at summarizing earnings calls using exact verbatim quotes with page references."},
            #         {"role": "user", "content": summary_prompt}
            #     ],
            #     temperature=0.2,
            #     max_tokens=1500
            # )

            summary_text = self._call_lyzr_api(
                agent_id="67c86b15be1fc2af4eb4027e",
                session_id="67c86b15be1fc2af4eb4027e",
                message=summary_prompt
            )
            # summary_text = response.choices[0].message.content

            if save_to_file:
                # Generate timestamp for the filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_filename = os.path.join("analysis_results", f"verbatim_summary_{timestamp}.md")

                # Create the directory if it doesn't exist
                os.makedirs("analysis_results", exist_ok=True)

                # Save the summary to file
                with open(summary_filename, 'w', encoding='utf-8') as f:
                    f.write(summary_text)
                print(f"\nVerbatim summary saved to: {summary_filename}")

            return summary_text

        except Exception as e:
            error_msg = f"Error generating verbatim summary: {str(e)}"
            if save_to_file:
                error_filename = os.path.join("analysis_results", "verbatim_summary_error.txt")
                with open(error_filename, 'w', encoding='utf-8') as f:
                    f.write(error_msg)
                print(f"\nError log saved to: {error_filename}")
            return error_msg

    def generate_insight_report(self, results: Dict, save_to_file: bool = True) -> str:
        """Generate insights based on verbatim quotes while maintaining the original context."""
        try:
            company_name = results.get("metadata", {}).get("company_name", "")

            insight_prompt = f"""Create an insight report based ONLY on the verbatim quotes from this earnings call.

            Company Name: {company_name}

            Verbatim quotes by category:
            {json.dumps(results['aggregated_verbatim_quotes'], indent=2)}

            Format the report as follows:
            # {company_name} Earnings Call Insights

            ## Macro environment
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

            ## Small deals, Discretionary, S&C
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

            ## Bookings/ Large Deals
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

            ## Pricing
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

            ## Margins
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)
            
             ## People
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Americas
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## EMEA
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## APAC
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## CMT
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## FS
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Products
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Resources
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## H&PS
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Managed Services/ Operations
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Industry X
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Song
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Cloud
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

             ## Data and AI / GenAI
            - [EXACT QUOTE] (Page X)
            - [EXACT QUOTE] (Page X)

            Rules:
            1. CITE the exact verbatim quotes with page numbers for EVERY insight
            2. Do not make claims that cannot be directly supported by the provided quotes
            3. Focus on specific numbers, metrics, and factual statements
            4. For each section, present the quote first, then provide the insight
            5. Use markdown formatting for readability"""

            # response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=[
            #         {"role": "system", "content": "You are a financial analyst expert at deriving insights from earnings call transcripts while maintaining fidelity to the exact verbatim quotes."},
            #         {"role": "user", "content": insight_prompt}
            #     ],
            #     temperature=0.3,
            #     max_tokens=1500
            # )
            # insight_text = response.choices[0].message.content

            insight_text = self._call_lyzr_api(
                agent_id="67c86b15be1fc2af4eb4027e",
                session_id="67c86b15be1fc2af4eb4027e",
                message=insight_prompt
            )

            if save_to_file:
                # Generate timestamp for the filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                insight_filename = os.path.join("analysis_results", f"verbatim_insights_{timestamp}.md")

                # Create the directory if it doesn't exist
                os.makedirs("analysis_results", exist_ok=True)

                # Save the insights to file
                with open(insight_filename, 'w', encoding='utf-8') as f:
                    f.write(insight_text)
                print(f"\nVerbatim insights saved to: {insight_filename}")

            return insight_text

        except Exception as e:
            error_msg = f"Error generating insights: {str(e)}"
            if save_to_file:
                error_filename = os.path.join("analysis_results", "verbatim_insights_error.txt")
                with open(error_filename, 'w', encoding='utf-8') as f:
                    f.write(error_msg)
                print(f"\nError log saved to: {error_filename}")
            return error_msg

    def generate_complete_analysis(self, results: Dict, save_to_file: bool = True) -> Dict:
        """Generate both verbatim summary and insights in a single method."""
        try:
            # Generate verbatim summary
            verbatim_summary = self.generate_verbatim_summary(results, False)

            # Generate insights
            insights = self.generate_insight_report(results, False)

            # Combine results
            complete_analysis = {
                "verbatim_summary": verbatim_summary,
                "insights": insights
            }

            if save_to_file:
                # Generate timestamp for the filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                company_name = results.get("metadata", {}).get("company_name", "company")
                company_name = company_name.replace(" ", "_").lower()

                # Create filename
                combined_filename = os.path.join("analysis_results", f"{company_name}_analysis_{timestamp}.md")

                # Create the directory if it doesn't exist
                os.makedirs("analysis_results", exist_ok=True)

                # Save the combined analysis to file
                with open(combined_filename, 'w', encoding='utf-8') as f:
                    f.write("# EARNINGS CALL VERBATIM ANALYSIS\n\n")
                    f.write("## VERBATIM SUMMARY\n")
                    f.write(verbatim_summary)
                    f.write("\n\n")
                    f.write("## INSIGHTS BASED ON VERBATIM QUOTES\n")
                    f.write(insights)

                print(f"\nComplete analysis saved to: {combined_filename}")

            return complete_analysis

        except Exception as e:
            error_msg = f"Error generating complete analysis: {str(e)}"
            if save_to_file:
                error_filename = os.path.join("analysis_results", "complete_analysis_error.txt")
                with open(error_filename, 'w', encoding='utf-8') as f:
                    f.write(error_msg)
                print(f"\nError log saved to: {error_filename}")
            return {"error": error_msg}

    # Legacy method compatibility wrappers
    def generate_summary(self, results: Dict, save_to_file: bool = True) -> str:
        """Legacy method that maps to generate_verbatim_summary."""
        return self.generate_verbatim_summary(results, save_to_file)

    def generate_summary_and_reflection(self, results: Dict, save_to_file: bool = True) -> Tuple[str, str]:
        """Legacy method that provides summary and reflection based on verbatim quotes."""
        summary = self.generate_verbatim_summary(results, False)
        insights = self.generate_insight_report(results, False)

        if save_to_file:
            # Generate timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            company_name = results.get("metadata", {}).get("company_name", "company")
            company_name = company_name.replace(" ", "_").lower()

            # Create filename for combined report
            combined_filename = os.path.join("analysis_results", f"{company_name}_summary_reflection_{timestamp}.md")

            # Create the directory if it doesn't exist
            os.makedirs("analysis_results", exist_ok=True)

            # Save the combined analysis to file
            with open(combined_filename, 'w', encoding='utf-8') as f:
                f.write("# EARNINGS CALL ANALYSIS\n")
                f.write("=" * 50 + "\n\n")
                f.write("## FACTUAL SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(summary)
                f.write("\n\n")
                f.write("## STRATEGIC REFLECTION\n")
                f.write("-" * 20 + "\n")
                f.write(insights)

            print(f"\nSummary and reflection saved to: {combined_filename}")

        return summary, insights

    def generate_summaryv1(self, results: Dict, save_to_file: bool = True) -> str:
        """Legacy method for backward compatibility."""
        return self.generate_verbatim_summary(results, save_to_file)