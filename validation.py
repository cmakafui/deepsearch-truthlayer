# validation.py
import time
import re
import uuid
from typing import Optional, List, Tuple
from datetime import datetime
import streamlit as st
import instructor
import google.generativeai as genai
from firecrawl import FirecrawlApp

from models import Source, Claim, ValidationResult, ExtractedReport, TrustReport


# --- Initialize API clients function ---
def initialize_clients(gemini_key, firecrawl_key):
    """Initialize API clients with provided keys"""
    api_keys_valid = False
    extractor_client = None
    validator_client = None
    firecrawl_app = None

    if gemini_key and firecrawl_key:
        try:
            extractor_client = instructor.from_gemini(
                client=genai.GenerativeModel(
                    model_name="models/gemini-2.5-pro-exp-03-25",
                ),
                mode=instructor.Mode.GEMINI_JSON,
            )
            validator_client = instructor.from_gemini(
                client=genai.GenerativeModel(
                    model_name="models/gemini-2.0-flash",
                ),
                mode=instructor.Mode.GEMINI_JSON,
            )
            firecrawl_app = FirecrawlApp(api_key=firecrawl_key)
            api_keys_valid = True
        except Exception as e:
            st.error(f"Error initializing clients: {e}. Please check your API keys.")
    else:
        st.warning(
            "Please provide both Gemini API Key and Firecrawl API Key to use this app."
        )

    return api_keys_valid, extractor_client, validator_client, firecrawl_app


# --- Core Logic Functions ---
def extract_claims_and_sources(
    report_text: str, extractor_client
) -> Optional[ExtractedReport]:
    """Extract claims and their source URLs from a report"""
    st.write("1. Extracting claims and sources...")

    report_id = str(uuid.uuid4())[:8]
    try:
        result = extractor_client.chat.completions.create(
            response_model=ExtractedReport,
            messages=[
                {
                    "role": "system",
                    "content": """Extract verifiable claims and their associated sources from this research report.

                 For each claim:
                 1. Extract the exact statement from the text
                 2. Create a verification question that precisely addresses what needs to be verified
                 3. Identify which URLs from the sources are referenced by this claim

                 Extract all sources (URLs) referenced in the document.

                 Only include factual claims that can be objectively verified.
                 Focus on claims with explicit source references.""",
                },
                {"role": "user", "content": report_text},
            ],
        )

        # Assign IDs
        for i, source in enumerate(result.sources):
            source.id = f"source-{report_id}-{i}"
        for i, claim in enumerate(result.claims):
            claim.id = f"claim-{report_id}-{i}"

        # Link claims to source IDs
        url_to_source_id = {source.url: source.id for source in result.sources}
        for claim in result.claims:
            claim.source_ids = [
                url_to_source_id.get(url)
                for url in claim.source_urls
                if url in url_to_source_id
            ]
            claim.source_ids = [
                source_id for source_id in claim.source_ids if source_id is not None
            ]

        st.write(
            f"   Extracted {len(result.claims)} claims and {len(result.sources)} sources."
        )
        return result
    except Exception as e:
        st.error(f"   Error extracting claims: {e}")
        return None


def fetch_sources(sources: List[Source], firecrawl_app) -> List[Source]:
    """Fetch content for all sources using Firecrawl"""
    st.write("2. Fetching source content...")
    if not firecrawl_app:
        st.error("Firecrawl client not initialized. Cannot fetch source content.")
        return sources

    progress_bar = st.progress(0)
    fetched_sources = []
    for i, source in enumerate(sources):
        st.write(f"   Fetching source {i + 1}/{len(sources)}: {source.url}")
        try:
            # Use 'formats' parameter instead of 'pageOptions'
            result = firecrawl_app.scrape_url(
                source.url, params={"formats": ["markdown"]}
            )

            # Simplified response handling based on notebook example
            if result and "markdown" in result:
                # Basic cleaning: remove excessive newlines/whitespace
                cleaned_content = re.sub(r"\s{3,}", "\n\n", result["markdown"]).strip()
                source.content = cleaned_content
                st.write(f"      Success (Length: {len(source.content)})")
            else:
                source.content = f"Error fetching content from {source.url}. Firecrawl returned empty or invalid data."
                st.warning(
                    f"      Failed to fetch or no content found for {source.url}"
                )
        except Exception as e:
            source.content = f"Error fetching content from {source.url}: {str(e)}"
            st.error(f"      Error fetching {source.url}: {e}")
        fetched_sources.append(source)
        progress_bar.progress((i + 1) / len(sources))
        time.sleep(0.1)  # Small delay to allow UI update

    st.write("   Finished fetching source content.")
    return fetched_sources


def validate_claim(
    claim: Claim, sources: List[Source], validator_client
) -> Optional[ValidationResult]:
    """Validate a claim against its sources in a single LLM call"""

    claim_sources = [
        s
        for s in sources
        if s.id in claim.source_ids and s.content and not s.content.startswith("Error")
    ]

    if not claim_sources:
        st.write(
            f"   Claim '{claim.statement[:50]}...': No valid sources found/fetched."
        )
        return ValidationResult(
            claim_id=claim.id,
            statement=claim.statement,
            verification_question=claim.verification_question,
            status="UNVERIFIABLE",
            confidence=0.0,
            reasoning="No valid source content available for this claim after fetching.",
            has_contradictions=False,
        )

    sources_text_list = []
    for s in claim_sources:
        content = s.content if s.content else "[NO CONTENT]"
        sources_text_list.append(f"SOURCE [{s.id}] {s.url}:\n{content}")

    sources_text = "\n\n".join(sources_text_list)

    try:
        result = validator_client.chat.completions.create(
            response_model=ValidationResult,
            messages=[
                {
                    "role": "system",
                    "content": """You are a meticulous fact-checker. Assess whether the claim is supported by the provided source excerpts ONLY. Do not use external knowledge.

                 Possible statuses:
                 - SUPPORTED: The claim is directly and fully supported by the text in one or more sources.
                 - PARTIALLY_SUPPORTED: The claim is partially supported, or supported with significant caveats or missing details mentioned in the claim.
                 - CONTRADICTED: The sources contain information that directly contradicts the claim.
                 - UNVERIFIABLE: There is not enough information in the provided source excerpts to verify or contradict the claim.

                 Provide a confidence score (0.0 to 1.0) reflecting your certainty based *only* on the provided text.
                 Explain your reasoning clearly, citing specific source IDs (e.g., [source-xyz-1]) where possible.
                 If different sources present conflicting information relevant to the claim, note this in the reasoning and set 'has_contradictions' to true.""",
                },
                {
                    "role": "user",
                    "content": f"""Please validate the following claim based *only* on the provided source excerpts:

                 CLAIM: {claim.statement}

                 VERIFICATION QUESTION: {claim.verification_question}

                 SOURCE EXCERPTS:
                 --- START OF SOURCES ---
                 {sources_text}
                 --- END OF SOURCES ---

                 Analyze whether this claim is supported, partially supported, contradicted, or unverifiable based *solely* on these excerpts. Provide reasoning and confidence.""",
                },
            ],
            max_retries=1,  # Reduce retries for faster feedback in UI
        )

        # Fill in claim details which are not part of the LLM response model definition
        result.claim_id = claim.id
        result.statement = claim.statement
        result.verification_question = claim.verification_question
        st.write(
            f"   Claim '{claim.statement[:50]}...': Result - {result.status} (Confidence: {result.confidence:.2f})"
        )
        return result
    except Exception as e:
        st.error(f"   Error validating claim '{claim.statement[:50]}...': {e}")
        # Return an UNVERIFIABLE result on error
        return ValidationResult(
            claim_id=claim.id,
            statement=claim.statement,
            verification_question=claim.verification_question,
            status="UNVERIFIABLE",
            confidence=0.0,
            reasoning=f"Validation failed due to an error: {e}",
            has_contradictions=False,
        )


def validate_all_claims(
    claims: List[Claim], sources: List[Source], validator_client
) -> List[ValidationResult]:
    """Validate all claims against their sources"""
    st.write("3. Validating claims against sources...")
    results = []
    progress_bar = st.progress(0)
    for i, claim in enumerate(claims):
        result = validate_claim(claim, sources, validator_client)
        if result:
            results.append(result)
        progress_bar.progress((i + 1) / len(claims))
        time.sleep(0.1)  # Small delay

    st.write(f"   Validated {len(results)} claims.")
    return results


def calculate_trust_score(validation_results: List[ValidationResult]) -> float:
    """Calculate the overall trust score based on validation results"""
    if not validation_results:
        return 0.0

    weights = {
        "SUPPORTED": 1.0,
        "PARTIALLY_SUPPORTED": 0.5,
        "CONTRADICTED": 0.0,
        "UNVERIFIABLE": 0.2,  # Give a small weight to unverifiable claims
    }

    # Calculate based on status counts, less sensitive to LLM confidence scores
    score_sum = sum(weights.get(result.status, 0.0) for result in validation_results)
    max_score = len(validation_results)  # Using simple count-based score

    if max_score == 0:
        return 0.0

    # Calculate average and scale to 0-100
    trust_score = (score_sum / max_score) * 100  # Status-based

    # Apply a penalty for contradictions found within sources for a *single* claim
    contradictions_within_claim = sum(
        1 for r in validation_results if r.has_contradictions
    )
    if contradictions_within_claim > 0:
        contradiction_penalty = min(20, contradictions_within_claim * 5)  # Cap penalty
        trust_score = max(0, trust_score - contradiction_penalty)
        st.write(
            f"   Applying penalty of {contradiction_penalty:.1f} due to {contradictions_within_claim} claim(s) with internal source contradictions."
        )

    return round(trust_score, 1)


def generate_trust_report(validation_results: List[ValidationResult]) -> TrustReport:
    """Generate a trust report from validation results"""
    st.write("4. Generating final report...")
    status_counts = {
        "SUPPORTED": 0,
        "PARTIALLY_SUPPORTED": 0,
        "CONTRADICTED": 0,
        "UNVERIFIABLE": 0,
    }

    for result in validation_results:
        status_counts[result.status] += 1

    has_contradictions = any(r.has_contradictions for r in validation_results)
    trust_score = calculate_trust_score(validation_results)

    report = TrustReport(
        id=str(uuid.uuid4())[:8],
        timestamp=datetime.now().isoformat(),
        trust_score=trust_score,
        claim_count=len(validation_results),
        results=status_counts,
        has_contradictions=has_contradictions,
    )
    st.write("   Report generation complete.")
    return report


# --- End-to-End Pipeline ---
def validate_report(
    report_text: str, extractor_client, validator_client, firecrawl_app
) -> Optional[Tuple[TrustReport, List[ValidationResult], ExtractedReport]]:
    """Run the complete validation pipeline on a research report"""
    extracted = extract_claims_and_sources(report_text, extractor_client)
    if not extracted or not extracted.claims:
        st.warning(
            "Could not extract any claims or sources. Please check the report format and content."
        )
        return None

    if not extracted.sources:
        st.warning("No sources were extracted. Claims will be marked as UNVERIFIABLE.")
        # Proceed without fetching, claims will be unverifiable
        sources_with_content = []
    else:
        sources_with_content = fetch_sources(extracted.sources, firecrawl_app)

    validation_results = validate_all_claims(
        extracted.claims, sources_with_content, validator_client
    )
    if not validation_results:
        st.error("Claim validation failed.")
        return None

    trust_report = generate_trust_report(validation_results)

    return trust_report, validation_results, extracted
