# app.py
import streamlit as st
import os
import pandas as pd
import warnings
import google.generativeai as genai
from dotenv import load_dotenv

from models import plot_trust_gauge, plot_claim_distribution, plot_confidence_per_claim
from validation import initialize_clients, validate_report

# Suppress warning messages
warnings.filterwarnings("ignore")

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Truth Layer: AI Research Validator")

# --- Load Environment Variables (as fallback) ---
load_dotenv()
default_gemini_api_key = os.getenv("GEMINI_API_KEY", "")
default_firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", "")

# --- Sample Data ---
sample_report = """
**A Very Short History of Abraham Lincoln**

Abraham Lincoln was born on February 12, 1809, in a log cabin in Kentucky. He was largely self-educated and worked as a lawyer before entering politics in Illinois. His election as the 16th President of the United States in 1860 led to the secession of Southern states and the start of the Civil War[^1][^3].

As president, Lincoln focused on preserving the Union. In 1863, he issued the Emancipation Proclamation, declaring freedom for enslaved people in Confederate-held territories and reframing the war around the abolition of slavery[^3][^5]. That same year, he delivered the Gettysburg Address, a defining speech in American history.

Lincoln was re-elected in 1864 and oversaw the Union's victory in April 1865. Just days later, on April 14, he was assassinated by John Wilkes Booth. His leadership during the Civil War and his moral vision for the nation cemented his legacy as one of America's greatest presidents[^3][^4][^6].

---

**Sources**

[^1]: https://en.wikipedia.org/wiki/Abraham_Lincoln  
[^3]: https://millercenter.org/president/lincoln/life-in-brief  
[^4]: https://en.wikipedia.org/wiki/Abraham_Lincoln  
[^5]: https://www.battlefields.org/learn/biographies/abraham-lincoln  
[^6]: https://kids.nationalgeographic.com/history/article/abraham-lincoln  
"""


def main():
    st.title("🔎 Truth Layer: AI Research Validator")
    st.markdown("""
    This app validates claims made in a research report against its cited sources.
    It uses AI to:
    1.  **Extract** claims and linked source URLs from the text.
    2.  **Fetch** the content of each source URL using Firecrawl.
    3.  **Validate** each claim against its source content using a Gemini model.
    4.  **Generate** a Trust Score and detailed report with visualizations.

    **Enter your API keys and research report text below and click 'Validate Report'.**
    """)

    # Create a section for API key inputs with some explanation
    st.subheader("🔑 API Keys")

    # API key input section with two columns
    col1, col2 = st.columns(2)

    with col1:
        # Use session state for API keys to persist during the session
        if "gemini_api_key" not in st.session_state:
            st.session_state.gemini_api_key = default_gemini_api_key

        gemini_api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.gemini_api_key,
            type="password",
            help="Enter your Gemini API key. Get one from https://ai.dev/",
            key="gemini_key_input",
        )
        st.session_state.gemini_api_key = gemini_api_key

    with col2:
        if "firecrawl_api_key" not in st.session_state:
            st.session_state.firecrawl_api_key = default_firecrawl_api_key

        firecrawl_api_key = st.text_input(
            "Firecrawl API Key",
            value=st.session_state.firecrawl_api_key,
            type="password",
            help="Enter your Firecrawl API key. Get one from https://firecrawl.dev/",
            key="firecrawl_key_input",
        )
        st.session_state.firecrawl_api_key = firecrawl_api_key

    # Initialize clients
    api_keys_valid, extractor_client, validator_client, firecrawl_app = (
        initialize_clients(gemini_api_key, firecrawl_api_key)
    )

    # Input for report text
    st.subheader("📝 Research Report")
    report_text_input = st.text_area(
        "Enter research report text", sample_report, height=300
    )

    if st.button("Validate Report", type="primary", disabled=not api_keys_valid):
        if not report_text_input.strip():
            st.warning("Please enter some report text to validate.")
        else:
            with st.spinner(
                "Running validation pipeline... This may take a few minutes depending on the number of sources."
            ):
                # Configure google.generativeai with the current API key
                genai.configure(api_key=gemini_api_key)

                st.subheader("📊 Validation Process Log")
                log_container = st.container(height=200)  # Container for logs
                # Redirect st.write to the container
                _write = st.write
                st.write = log_container.write

                validation_output = validate_report(
                    report_text_input, extractor_client, validator_client, firecrawl_app
                )

                # Restore original st.write
                st.write = _write

            st.subheader("✅ Validation Complete!")

            if validation_output:
                trust_report, validation_results, extracted = validation_output

                st.header("📈 Trust Report Summary")

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.plotly_chart(
                        plot_trust_gauge(trust_report.trust_score),
                        use_container_width=True,
                    )
                    st.metric("Claims Analyzed", trust_report.claim_count)
                    st.metric(
                        "Source Contradictions Found",
                        "Yes" if trust_report.has_contradictions else "No",
                    )

                with col2:
                    st.plotly_chart(
                        plot_claim_distribution(trust_report), use_container_width=True
                    )

                st.plotly_chart(
                    plot_confidence_per_claim(validation_results),
                    use_container_width=True,
                )

                st.header("📋 Detailed Claim Results")

                # Create DataFrame for display
                claim_id_to_source_urls = {
                    claim.id: claim.source_urls for claim in extracted.claims
                }
                claim_id_to_source_ids = {
                    claim.id: claim.source_ids for claim in extracted.claims
                }

                # Source fetch status check
                source_fetch_status = {}
                for source in extracted.sources:
                    if hasattr(source, "content"):
                        if source.content and not source.content.startswith("Error"):
                            source_fetch_status[source.id] = "Fetched"
                        else:
                            source_fetch_status[source.id] = "Fetch Error/Empty"
                    else:
                        source_fetch_status[source.id] = "Not Fetched (Error?)"

                results_data = []
                for i, r in enumerate(validation_results):
                    source_ids_for_claim = claim_id_to_source_ids.get(r.claim_id, [])
                    source_urls_for_claim = claim_id_to_source_urls.get(r.claim_id, [])
                    num_sources = len(source_ids_for_claim)
                    # Check how many sources used in validation had actual content
                    num_valid_sources = sum(
                        1
                        for src_id in source_ids_for_claim
                        if source_fetch_status.get(src_id) == "Fetched"
                    )

                    results_data.append(
                        {
                            "No.": i + 1,
                            "Claim": r.statement,
                            "Status": r.status,
                            "Confidence": f"{r.confidence:.2f}",
                            "Sources Cited": num_sources,
                            "Valid Sources Used": num_valid_sources,
                            "Internal Contradiction": "Yes"
                            if r.has_contradictions
                            else "No",
                        }
                    )
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)

                st.header("🔍 Claim-by-Claim Analysis")

                for i, result in enumerate(validation_results):
                    with st.expander(
                        f"Claim {i + 1}: {result.statement[:80]}... ({result.status})"
                    ):
                        st.markdown("**Claim Statement:**")
                        st.info(result.statement)
                        st.markdown("**Verification Question:**")
                        st.info(result.verification_question)
                        st.markdown(f"**Validation Status:** `{result.status}`")
                        st.markdown(f"**Confidence Score:** `{result.confidence:.2f}`")
                        st.markdown(
                            f"**Internal Source Contradictions:** `{'Yes' if result.has_contradictions else 'No'}`"
                        )
                        st.markdown("**AI Reasoning:**")
                        st.success(result.reasoning)

                        st.markdown("**Sources Used for this Claim:**")
                        source_ids_for_claim = claim_id_to_source_ids.get(
                            result.claim_id, []
                        )
                        source_urls_for_claim = claim_id_to_source_urls.get(
                            result.claim_id, []
                        )

                        if not source_ids_for_claim:
                            st.warning(
                                "No sources were linked to this claim during extraction."
                            )
                        else:
                            for src_id, src_url in zip(
                                source_ids_for_claim, source_urls_for_claim
                            ):
                                fetch_status = source_fetch_status.get(
                                    src_id, "Unknown"
                                )
                                st.markdown(
                                    f"- `{src_id}`: [{src_url}]({src_url}) - **Fetch Status: {fetch_status}**"
                                )
            else:
                st.error("Validation pipeline failed to produce results.")

    # Add information about how the app works with multiple users
    st.subheader("ℹ️ About User Sessions")
    st.info("""
    **How this app works with multiple users:**

    1. **Each user has an isolated session** - Your API keys and report data are private to your session
    2. **API usage is billed to your accounts** - API calls use your provided keys, so any costs are charged to your accounts
    3. **Shared server resources** - The app runs on shared server resources, but data between users is not shared
    4. **Session data** - Your API keys remain in your session unless you clear cache or cookies

    For more information about Streamlit's session state model, visit [Streamlit documentation](https://docs.streamlit.io/library/advanced-features/session-state).
    """)

    # --- Footer or additional info ---
    st.markdown("---")
    st.caption("Powered by Gemini, Firecrawl, and Streamlit.")


if __name__ == "__main__":
    main()
