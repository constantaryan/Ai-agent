# AI Job Hunting Assistant

An intelligent job hunting assistant that helps you find and analyze job opportunities based on your preferences. This tool uses AI to search job sites like Naukri, Indeed, and Monster to find relevant job listings and provide personalized recommendations.

## Features

- **Job Search**: Find job opportunities matching your job title, location, experience, and skills
- **Job Analysis**: Get detailed analysis of job opportunities including skills match, company insights, and application tips
- **Industry Trends**: Analyze industry trends including salary ranges, growth rates, and in-demand skills
- **Career Recommendations**: Receive personalized recommendations for the best job opportunities

## Requirements

- Python 3.8+
- Firecrawl API key (for web scraping)
- OpenAI API key (for AI analysis)

## Installation

1. Clone the repository
2. Install the required packages:
3. Set up your API keys:
   - Create a `.env` file in the root directory
   - Add your API keys:
     ```
     FIRECRAWL_API_KEY=your_firecrawl_api_key
     OPENAI_API_KEY=your_openai_api_key
     OPENAI_MODEL_ID= o3-mini  or gpt-4o-mini
     ```

## Usage

Run the Streamlit app:

```
streamlit run agents/jobs/job-hunt-agent.py
```

Then:

1. Enter your job search criteria:
   - Job Title (e.g., "Software Engineer")
   - Location (e.g., "Bangalore" or "Remote")
   - Experience (in years)
   - Skills (comma-separated)
   - Industry/Job Category

2. Click "Start Job Search" to begin the search process

3. Review the job recommendations and industry trends analysis

## How It Works

1. The agent uses Firecrawl to search job sites for opportunities matching your criteria
2. It extracts detailed information about each job including title, company, location, salary, and requirements
3. The AI analyzes the job listings to find the best matches for your profile
4. The agent also analyzes industry trends to provide insights about salary ranges, growth potential, and in-demand skills
5. All information is presented in a structured format with actionable recommendations

## Supported Job Sites

- Naukri
- Indeed
- Monster
- PayScale (for industry trends)

## Note

This tool requires valid API keys to function. The Firecrawl API is used for web scraping, and the OpenAI API is used for AI analysis. You can enter these keys in the sidebar of the application or set them as environment variables. 
