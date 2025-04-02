# models.py
from typing import Optional, Literal, List, Dict
from pydantic import BaseModel, Field
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# --- Pydantic Models ---
class Source(BaseModel):
    """A source referenced in a research report"""

    id: str = Field(description="Unique identifier for the source")
    url: str = Field(description="URL of the source")
    title: Optional[str] = Field(None, description="Title of the source")
    content: Optional[str] = Field(None, description="The source content")


class Claim(BaseModel):
    """A claim made in a research report that needs verification"""

    id: str = Field(description="Unique identifier for the claim")
    statement: str = Field(description="The claim as stated in the report")
    verification_question: str = Field(
        description="The claim rewritten as a verification question"
    )
    source_urls: List[str] = Field(description="URLs of sources this claim references")
    source_ids: List[str] = Field(
        default_factory=list, description="Source IDs (populated in post-processing)"
    )


class ValidationResult(BaseModel):
    """The validation result for a claim"""

    claim_id: str = Field(description="ID of the claim")
    statement: str = Field(description="The original claim statement")
    verification_question: str = Field(description="The verification question")
    status: Literal[
        "SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"
    ] = Field(description="Validation status")
    confidence: float = Field(description="Confidence score (0.0-1.0)")
    reasoning: str = Field(description="Explanation of the validation")
    has_contradictions: bool = Field(
        False, description="Whether sources contradict each other"
    )


class ExtractedReport(BaseModel):
    """The extracted claims and sources from a report"""

    claims: List[Claim] = Field(description="List of extracted claims")
    sources: List[Source] = Field(description="List of extracted sources")


class TrustReport(BaseModel):
    """Final trust report for a validated document"""

    id: str = Field(description="Report ID")
    timestamp: str = Field(description="When the validation was performed")
    trust_score: float = Field(description="Overall trust score (0-100)")
    claim_count: int = Field(description="Total number of claims")
    results: Dict[str, int] = Field(description="Counts by validation status")
    has_contradictions: bool = Field(
        False, description="Whether any sources contradict each other"
    )


# --- Visualization Functions ---
def plot_trust_gauge(trust_score: float) -> go.Figure:
    """Create a gauge visualization of the trust score using Plotly"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=trust_score,
            title={"text": "Overall Trust Score", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {
                    "color": "rgba(0,0,0,0)"
                },  # Transparent bar, color shown by steps
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 50], "color": "#FF6347"},  # Tomato Red
                    {"range": [50, 80], "color": "#FFD700"},  # Gold Yellow
                    {"range": [80, 100], "color": "#90EE90"},
                ],  # Light Green
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": trust_score,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_claim_distribution(trust_report: TrustReport) -> go.Figure:
    """Create a pie chart of claim validation statuses using Plotly"""
    labels = list(trust_report.results.keys())
    values = list(trust_report.results.values())
    colors = {
        "SUPPORTED": "#90EE90",  # Light Green
        "PARTIALLY_SUPPORTED": "#FFD700",  # Gold Yellow
        "CONTRADICTED": "#FF6347",  # Tomato Red
        "UNVERIFIABLE": "#D3D3D3",  # Light Grey
    }
    marker_colors = [colors[status] for status in labels]

    fig = px.pie(
        names=labels,
        values=values,
        title="Claim Validation Status Distribution",
        color_discrete_sequence=marker_colors,
        hole=0.3,  # Make it a donut chart
    )
    fig.update_traces(textposition="inside", textinfo="percent+label+value")
    fig.update_layout(
        legend_title_text="Status", height=400, margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


def plot_confidence_per_claim(validation_results: List[ValidationResult]) -> go.Figure:
    """Create a bar chart of confidence scores for each claim using Plotly"""
    if not validation_results:
        return go.Figure().update_layout(title="No claims validated.")

    claims = [
        f"Claim {i + 1}<br><sub>({r.statement[:30]}...)</sub>"
        for i, r in enumerate(validation_results)
    ]
    confidences = [r.confidence for r in validation_results]
    statuses = [r.status for r in validation_results]
    colors = {
        "SUPPORTED": "#90EE90",
        "PARTIALLY_SUPPORTED": "#FFD700",
        "CONTRADICTED": "#FF6347",
        "UNVERIFIABLE": "#D3D3D3",
    }
    bar_colors = [colors[status] for status in statuses]

    df = pd.DataFrame(
        {
            "Claim": claims,
            "Confidence": confidences,
            "Status": statuses,
            "Color": bar_colors,
        }
    )

    fig = px.bar(
        df,
        x="Claim",
        y="Confidence",
        color="Status",  # Color bars by status
        color_discrete_map=colors,  # Ensure consistent colors
        title="Confidence Score per Claim",
        text="Confidence",  # Display confidence value on bar
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        yaxis_range=[0, 1.1],
        xaxis_title="Claim",
        yaxis_title="Confidence Score",
        legend_title_text="Status",
        height=400 + (len(claims) * 10),  # Dynamically adjust height slightly
        margin=dict(l=10, r=10, t=50, b=120),  # Increase bottom margin for labels
    )
    return fig
