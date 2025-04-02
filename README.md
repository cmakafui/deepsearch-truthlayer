# Truth Layer: DeepResearch Validator

**A verification system that automatically checks whether claims in an AI-generated report are actually supported by the sources the report cites.**

## 🌟 Motivation

AI tools like ChatGPT and Gemini can instantly generate detailed reports on complex topics by pulling from countless online sources. While these tools are impressive at synthesizing information, there's a crucial gap: **how do we verify the accuracy of AI-generated reports?**

Truth Layer addresses this problem by providing an automated system to validate claims against their cited sources, giving users confidence in AI-generated research without hours of manual fact-checking.

## 🔍 How It Works

Truth Layer follows a four-step verification process:

1. **Extraction**: Uses Google's Gemini to identify specific factual claims, reframe them as verification questions and connect them to the exact URLs cited as evidence
2. **Source Retrieval**: Fetches the actual content from those URLs using batch scraping
3. **Verification**: For each claim, evaluates whether it is SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, or UNVERIFIABLE based solely on the provided source material
4. **Trust Calculation**: Weighs each validation result to calculate an overall Trust Score

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/cmakafui/deepsearch-truthlayer.git
cd truth-layer

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🔑 API Keys

You'll need API keys for:

- [Google Gemini API](https://ai.dev/) - For claim extraction and validation
- [Firecrawl API](https://firecrawl.dev/) - For fetching source content

Create a `.env` file in the project root with your API keys:

```
GEMINI_API_KEY=your_gemini_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
```

## 💻 Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Navigate to the URL provided in the terminal (typically http://localhost:8501) to access the Truth Layer web interface.

### Using the Application:

1. Provide your API keys (or they'll be loaded from the `.env` file)
2. Enter or paste your research report text
3. Click "Validate Report"
4. Review the results:
   - Overall Trust Score
   - Claim validation status distribution
   - Confidence scores per claim
   - Detailed claim-by-claim analysis

## 🏗️ Project Structure

```
truth-layer/
│
├── notebooks         # Jupyter notebooks for exploratory analysis
├── app.py            # Main Streamlit application
├── models.py         # Data models and visualization functions
├── validation.py     # Core validation logic
├── .env              # API keys (not tracked in git)
├── requirements.txt  # Project dependencies
└── README.md         # This file
```

## 🧠 Architecture

Truth Layer uses a pipeline architecture:

1. **Claim & Source Extraction**:

   - Leverages Gemini through the Instructor library for structured JSON outputs
   - Extracts claims and reframes them as verification questions
   - Maps claims to their cited sources

2. **Source Content Retrieval**:

   - Uses Firecrawl for efficient web scraping
   - Handles rate limiting and error management
   - Processes multiple sources in parallel

3. **Claim Validation**:

   - Uses a specialized Gemini model for fact-checking
   - Evaluates claims solely based on provided source material
   - Assigns confidence scores and detects contradictions

4. **Trust Score Calculation & Visualization**:
   - Weighs validation results based on status
   - Applies penalties for contradictions
   - Generates interactive visualizations with Plotly

## 🚧 Challenges and Limitations

- **Access Limitations**: Many valuable sources hide behind paywalls or require logins
- **Technical Constraints**: Challenges with rate limits and varying page structures
- **Nuance Interpretation**: Deciding when a claim is only 'partially supported' requires careful prompt engineering

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Carl Kugblenu**

For more information, read the full blog post: [Verifying DeepResearch: Building the Truth Layer](https://cmakafui.substack.com/p/verifying-deepresearch-building-the)
