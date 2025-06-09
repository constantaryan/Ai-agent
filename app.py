from typing import Dict, List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAILike
from firecrawl import FirecrawlApp
import streamlit as st
import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env files
load_dotenv()

# Point OpenAI client at Mistral AI's hosted endpoint
openai.api_base = "https://api.mistral.ai/v1"

# ----------------------------
# Schemas for structured data
# ----------------------------
class NestedModel1(BaseModel):
    """Schema for job posting data"""
    region: str = Field(description="Region or Area where Job is located", default=None)
    role: str = Field(description="Specific Role or function within the job category", default=None)
    job_title: str = Field(description="Title of the Job position", default=None)
    experience: str = Field(description="Experience required for the position", default=None)
    job_link: str = Field(description="Link to the Job posting", default=None)

class ExtractSchema(BaseModel):
    """Schema for postings extraction"""
    job_postings: List[NestedModel1] = Field(description="List of job postings")

class IndustryTrend(BaseModel):
    """Schema for Industry Trend data"""
    industry: str = Field(description="Industry Name", default=None)
    avg_salary: float = Field(description="Average salary in the industry", default=None)
    growth_rate: float = Field(description="Growth rate of the industry", default=None)
    demand_level: str = Field(description="Demand level in the industry", default=None)
    top_skills: List[str] = Field(description="Top skills in demand for this industry", default=None)

class IndustryTrendsSchema(BaseModel):
    """Schema for Industry Trends Extraction"""
    industry_trends: List[IndustryTrend] = Field(description="List of Industry Trends")

class FirecrawlResponse(BaseModel):
    """Schema for Firecrawl API response"""
    success: bool
    data: Dict
    status: str
    expiresAt: str

# ----------------------------
# Main Agent Implementation
# ----------------------------
class JobHuntingAgent:
    """Agent responsible for finding jobs and providing recommendations"""
    def __init__(
        self,
        firecrawl_api_key: str,
        mistral_api_key: str = "dummy_key",
        model_id: str = "mistral-small-latest"
    ):
        role_map = {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "__default__": "system"
        }
        # Configure Mistral via the OpenAI-compatible client
        self.agent = Agent(
            model=OpenAILike(
                id=model_id,
                api_key=mistral_api_key,
                base_url="https://api.mistral.ai/v1"
            ),
            markdown=True,
            system_message=(
                "This particular Agent command helps find and analyze "
                "job opportunities based on user preferences."
            )
        )
        self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)

    def find_jobs( #This method takes user input:
            self,
            job_title: str,
            location: str,
            experience_years:float,
            skills: List[str]
    )-> str:
        """Find and analyze jobs based on user prefrences"""
        # Format parameters for URL construction
        formatted_job_title = job_title.lower().replace(" ","-")
        formatted_location = location.lower().replace(" ","-")
        skills_string = ", ".join(skills)
        # example
        # "Data Scientist" → "data-scientist"
        # "New York" → "new-york"
        # ["Python", "Machine Learning"] → "Python, Machine Learning"

        # Define Job Search URL
        urls = [
            f"https://www.naukri.com/{formatted_job_title}-jobs-in-{formatted_location}",
            f"https://www.indeed.com/jobs?q={formatted_job_title}&l={formatted_location}",
            f"https://www.monster.com/jobs/search/?q={formatted_job_title}&where={formatted_location}",
        ]
        # for debugging 
        print(f"Searching for jobs with URLs: {urls}")

        try:
            # Extract the Job data using Firecrawl
            raw_response = self.firecrawl.extract(
                urls = urls,
                prompt= f""" Extract Job Posting by region , roles, job titles and  experience from these job sites.

                    Look for Jobs that match these criteria:
                    -Job Title: Should be related to {job_title}
                    -Location: {location} (include remote Jobs if available)
                    -Experience: Around {experience_years} years
                    -Skills: Should match at least some of these skills :{skills_string}
                    -Job Type: Full-time, Part-Time, Contract, Temperory, Internship

                    For each Job posting, extract:
                    -region: The Broader region or area where the job is located
                    -role: The specific role or function 
                    -job_title: The exact title of the job 
                    -experience: The experience requirement in years or levels
                    -job_link: The link to the job posting 

                    IMPORTANT: Return data for at least 3 different job opportunities. MAXIMUM 10.
                    """,
                schema =  ExtractSchema.model_json_schema()
            )
            print("Raw Job Response:",raw_response)

            #process the Raw response 
            jobs = []
            if raw_response.success:
                data = raw_response.data  # this is a dict, per SDK docs
                jobs = data.get("job_postings", [])

            else:
                jobs=[]

            print("Processed jobs",jobs)

            if not jobs:
                return "No job listing found matching your criteria. Try adjusting your search parameters or try different job sites."
            
            # Analysise the Job data using AI Agent 
            analysis = self.agent.run(
                f"""As a career expert, anaylze these job opportunities:
                
                Job Found in json format:
                {jobs}

                **IMPORTANT INSTRUCTIONS:**
                1.ONLY analyze jobs from the above JSON data that match the user's requirements:
                    -Job Title: Related to {job_title}
                    -Location/Region: Near {location}
                    -Experience: Around {experience_years} years
                    -skills: {skills_string}
                    -Job type: Full-time, Part-time, Contract, Temperory, Internship
                2.DO NOT CREATE new Job Listings
                3.From the matching jobs, select 5-6 jobs that best match the user's skills and experience

                Please provide your analysis in this format:
                
                💼 SELECTED JOB OPPORTUNITIES
                • List only 5-6 best matching jobs
                • For each job include:
                  - Job Title and Role
                  - Region/Location
                  - Experience Required
                  - Pros and Cons
                  - Job Link
                🔍 SKILLS MATCH ANALYSIS
                • Compare the selected jobs based on:
                  - Skills match with user's profile
                  - Experience requirements
                  - Growth potential

                💡 RECOMMENDATIONS
                • Top 3 jobs from the selection with reasoning
                • Career growth potential
                • Points to consider before applying

                📝 APPLICATION TIPS
                • Job-specific application strategies
                • Resume customization tips for these roles

                Format your response in a clear, structured way using the above sections.
                """
            )
            return analysis.content
        except Exception as e:
            print(f"Error in find_jobs: {str(e)}")
            return f"An error occured while searching for jobs: {str(e)}\n\nPlease try again with different search parameters or check if the job sites are supported by Firecrawl."
    def get_industry_trends(self,job_category:str)-> str:
        """Get Trends for the specified job category/industry"""
        # Define URLs for industry trend data
        urls =[
            f"https://www.payscale.com/research/US/Job={job_category.replace(' ','-')}/Salary",
            f"https://www.glassdoor.com/Salaries/{job_category.lower().replace(' ','-')}-salary-SRCH_KO0,{len(job_category)}.htm"
        ]
        print(f"Searching for industry trends with urls: {urls}")

        try:
            # Extract industry trend data using Firecrawl 
            raw_response = self.firecrawl.extract(
                urls = urls,
                prompt = f"""Extract industry trends data for the {job_category} industry.

                    For each industry trend, extract:
                    - industry: The specific industry or sub-category
                    - avg_salary: The average salary in this industry (as a number)
                    - growth_rate: The growth rate of this industry (as a number)
                    - demand_level: The demand level (e.g., "High", "Medium", "Low")
                    - top_skills: A list of top skills in demand for this industry
                    
                    IMPORTANT: 
                    - Extract data for at least 3-5 different roles or sub-categories within this industry
                    - Include salary trends, growth rate, and demand level
                    - Identify top skills in demand for this industry
                    """,
                schema= IndustryTrendsSchema.model_json_schema()
            )
            # for Debugging Purposes
            print("Raw Industry trends response:",raw_response)

            # Process the Raw response
            # if isinstance(raw_response,dict) and raw_response.get('success'):
            #     industries = raw_response['data'].get('industry_trends',[])
            
            industries = []
            if raw_response.success:
                data = raw_response.data
                industries = data.get("industry_trends", [])

                if not industries:
                    return f"No industry trends data available for {job_category}.Try a different industry category."
                
                # Analyze the industry trend data using the AI agent
                analysis = self.agent.run(
                    f"""As a career expert, analyze these industry trends for {job_category}:

                    {industries}

                    Please provide:
                    1. A bullet-point summary of the salary and demand trends
                    2. Identify the top skills in demand for this industry
                    3. Career growth opportunities:
                       - Roles with highest growth potential
                       - Emerging specializations
                       - Skills with increasing demand
                    4. Specific advice for job seekers based on these trends

                    Format the response as follows:
                    
                    📊 INDUSTRY TRENDS SUMMARY
                    • [Bullet points for salary and demand trends]

                    🔥 TOP SKILLS IN DEMAND
                    • [Bullet points for most sought-after skills]

                    📈 CAREER GROWTH OPPORTUNITIES
                    • [Bullet points with growth insights]

                    🎯 RECOMMENDATIONS FOR JOB SEEKERS
                    • [Bullet points with specific advice]
                    """
                )
                return analysis.content
            return f"No industry trend data available for {job_category}.Try different industry category."
        except Exception as e:
            print(f"Error in get_industry_trends: {str(e)}")
            return f"An error occurerd while fetching industry trends: {str(e)}\n\nPlease try again with a different industry category or check if the sites are supported by Firecrawl. "
    


def create_job_agent():
    """Create Job Hunting agent with API keys from session state."""
    if "job_agent" not in st.session_state:
        st.session_state.job_agent = JobHuntingAgent(
            firecrawl_api_key=st.session_state.firecrawl_key,
            mistral_api_key=st.session_state.get("mistral_key", "dummy_key"),
            model_id=st.session_state.model_id
        )

def main():
    # Configure the page
    st.set_page_config(
        page_title="AI Job Hunting Assistant",
        page_icon="💼",
        layout="wide"
    )

    # Load API keys & defaults
    env_firecrawl_key = os.getenv("FIRECRAWL_API_KEY", "")
    env_mistral_key = os.getenv("MISTRAL_API_KEY", "")
    # default_model could also come from env if desired
    default_model = "mistral-small-latest"

    with st.sidebar:
        st.title("🔑 API Configuration")
        st.subheader("🤖 Model Selection")

        # Let user pick a Mistral model
        model_id = st.selectbox(
            "Choose Mistral Model",
            options=[
                "mistral-small-latest",
                "mistral-medium-latest"
            ],
            index=0,
            help="Select the Mistral model to use."
        )
        st.session_state.model_id = model_id

        st.divider()
        st.subheader("🔐 API Keys")

        if env_firecrawl_key:
            st.success("✅ Firecrawl API Key found in environment variables")
        if env_mistral_key:
            st.success("✅ Mistral API Key found in environment variables")

        firecrawl_key = st.text_input(
            "Firecrawl API Key (optional if set in environment)",
            type="password",
            help="Enter your Firecrawl API key or set FIRECRAWL_API_KEY",
            value="" if not env_firecrawl_key else ""
        )
        mistral_key = st.text_input(
            "Mistral API Key (optional if set in environment)",
            type="password",
            help="Enter your Mistral API key or set MISTRAL_API_KEY",
            value="" if not env_mistral_key else ""
        )

        firecrawl_key = firecrawl_key or env_firecrawl_key
        mistral_key = mistral_key or env_mistral_key

        if firecrawl_key and mistral_key:
            st.session_state.firecrawl_key = firecrawl_key
            st.session_state.mistral_key = mistral_key
            create_job_agent()
        else:
            missing = []
            if not firecrawl_key:
                missing.append("Firecrawl API key")
            if not mistral_key:
                missing.append("Mistral API key")
            st.warning(f"⚠️ Missing required API keys: {', '.join(missing)}")
            st.info("Please provide the missing keys above or set them in environment variables.")

    # Main Interface
    st.title("💼 AI Job Hunting Assistant")
    st.info(
        """
        Welcome to the AI Job Hunting Assistant! 
        Enter your job search criteria below to get job recommendations 
        and industry insights.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        job_title = st.text_input(
            "Job Title",
            placeholder="Enter Job title (e.g., Software Engineer)"
        )
        location = st.text_input(
            "Location",
            placeholder="Enter Location (e.g., Bangalore, Remote)"
        )
    with col2:
        experience_years = st.number_input(
            "Experience (in years)",
            min_value=0,
            max_value=30,
            value=2,
            step=1
        )
        skills_input = st.text_area(
            "Skills (comma separated)",
            placeholder="e.g., Python, JavaScript, React, SQL"
        )

    skills = [s.strip() for s in skills_input.split(",")] if skills_input else []

    job_category = st.selectbox(
        "Industry/Job Category",
        options=[
            "Information Technology",
            "Software development",
            "Data Science",
            "Marketing",
            "Finance",
            "Healthcare",
            "Education",
            "Engineering",
            "Sales",
            "Human Resources"
        ]
    )

    if st.button("🔍 Start Job Search", use_container_width=True):
        if "job_agent" not in st.session_state:
            st.error("⚠️ Please enter your API keys in the sidebar first!")
            return
        if not job_title or not location:
            st.error("⚠️ Please enter both job title and location!")
            return
        if not skills:
            st.warning("⚠️ No skills provided. Adding skills will improve job matching.")

        try:
            with st.spinner("🔍 Searching for jobs..."):
                job_results = st.session_state.job_agent.find_jobs(
                    job_title=job_title,
                    location=location,
                    experience_years=experience_years,
                    skills=skills
                )
            if "error" in job_results.lower():
                st.error(job_results)
            else:
                st.success("✅ Job search completed!")
                st.subheader("💼 Job Recommendations")
                st.markdown(job_results)

                st.divider()
                with st.spinner("📊 Analyzing industry trends..."):
                    industry_trends = st.session_state.job_agent.get_industry_trends(job_category)
                if "error" in industry_trends.lower():
                    st.error(industry_trends)
                else:
                    st.success("✅ Industry analysis completed!")
                    with st.expander(f"📈 {job_category} Industry Trends Analysis"):
                        st.markdown(industry_trends)

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            if "website is no longer supported" in str(e).lower():
                st.info("One of the job sites isn’t supported by Firecrawl. Contact Firecrawl support.")
            elif "api key" in str(e).lower():
                st.info("Check that your API keys are correct and have necessary permissions.")
            else:
                st.info("Try again with different parameters or check your internet connection.")

if __name__ == "__main__":
    main()
