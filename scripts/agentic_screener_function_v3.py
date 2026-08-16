"""CrewAI multi-agent resume screening system."""

import warnings

warnings.filterwarnings("ignore")

import json
import os
import re
from textwrap import dedent
from typing import List, Optional

import pandas as pd
from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# Weight keys the caller must supply, and the score column each one applies to.
SCORE_COLUMNS = [
    "Technical_Score",
    "Experience_Score",
    "Education_Score",
    "Industry_Score",
]

DEFAULT_WEIGHTS = {
    "Technical_Score": 0.35,
    "Experience_Score": 0.30,
    "Education_Score": 0.15,
    "Industry_Score": 0.20,
}


class FinalRecommendation(BaseModel):
    technical_score: float
    technical_score_reason: str = Field(description="Detailed explanation for the technical score")
    industry_score: float
    industry_score_reason: str = Field(description="Detailed explanation for the industry score")
    experience_score: float
    experience_score_reason: str = Field(description="Detailed explanation for the experience score")
    education_score: float
    education_score_reason: str = Field(description="Detailed explanation for the education score")
    key_strengths: List[str] = Field(description="List of candidate's key strengths relevant to the role")
    areas_of_concern: List[str] = Field(description="List of potential concerns or gaps in the candidate's profile")
    match_category: str = Field(description="One of 'Strong Match' (>75), 'Potential Match' (50-75), 'Not a Match' (<50)")
    relative_ranking: Optional[str] = Field(None, description="Optional ranking relative to other candidates")
    justification: str = Field(description="Detailed justification for the recommendation")
    recommendation: str = Field(description="Clear hiring recommendation")


def build_llm():
    """LLM used by every agent. Model name is configurable via .env."""
    return LLM(model=os.getenv("CREWAI_LLM_MODEL", "azure/gpt-4o-mini"))


class ResumeScreeningSystem:
    def __init__(self, job_description, llm):
        """
        Initialize the Resume Screening System with a job description.

        Args:
            job_description (str): The job description to match candidates against
            llm: The language model to use for agents
        """
        if not job_description or not str(job_description).strip():
            raise ValueError("job_description is empty - cannot screen candidates against it.")

        self.job_description = job_description
        self.llm = llm
        self.setup_agents()
        self.setup_tasks()
        self.setup_crew()

    def setup_agents(self):
        """Set up the specialized agents for resume screening"""

        # Technical Skills Assessor
        self.technical_skills_agent = Agent(
            role="Technical Skills Assessor",
            goal="Evaluate candidate's technical skills against job requirements",
            backstory=dedent("""
                You are an expert in evaluating technical skills in resumes.
                Your expertise is in mapping technical skills mentioned in resumes
                to those required in job descriptions. You understand various
                technologies, programming languages, tools, and frameworks.
                You evaluate each candidate independently based on their merits.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        # Work Experience Evaluator
        self.work_experience_agent = Agent(
            role="Work Experience Evaluator",
            goal="Assess the relevance and quality of candidate's work experience",
            backstory=dedent("""
                You analyze work experience to determine if it aligns with the job.
                You look for relevant projects, responsibilities, and achievements
                that would translate well to the position. You can detect patterns
                of career progression and determine if the candidate has the right
                industry experience. You evaluate each candidate independently.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        # Education Qualification Assessor
        self.education_agent = Agent(
            role="Education Qualification Assessor",
            goal="Evaluate the candidate's educational background",
            backstory=dedent("""
                You assess educational qualifications to determine if they meet
                the job requirements. You understand different degree types,
                educational institutions, and how they relate to industry standards.
                You can evaluate the relevance of the candidate's academic background.
                You evaluate each candidate independently based on their qualifications.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        # Industry Fit Evaluator
        self.industry_agent = Agent(
            role="Industry Fit Evaluator",
            goal="Assess if the candidate has relevant industry experience",
            backstory=dedent("""
                You specialize in evaluating industry-specific experience and knowledge.
                You understand different industries, their specific requirements,
                terminology, and best practices. You can determine if a candidate's
                background aligns with the target industry. You evaluate each candidate
                independently based on their own industry experience.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        # Senior Resume Analyst
        self.senior_analyst = Agent(
            role="Senior Resume Analyst",
            goal="Provide a comprehensive evaluation and final recommendation",
            backstory=dedent("""
                As a senior hiring analyst, you review all aspects of a candidate's
                profile and make a final recommendation. You can synthesize information
                from multiple specialized assessments and provide a holistic evaluation.
                Your expertise lies in understanding how different aspects of a profile
                contribute to job fit. You evaluate each candidate purely on their own
                merits without comparison to others unless specifically asked to rank them.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

    def setup_tasks(self):
        """Set up the tasks for each agent"""

        # Technical Skills Assessment Task
        self.tech_skills_task = Task(
            description=dedent(f"""
                Analyze the technical skills listed in the candidate's profile.
                Compare them with the required skills in the job description:
                {self.job_description}

                Provide a detailed assessment of:
                1. Skills match percentage
                2. Critical skills present
                3. Critical skills missing
                4. Bonus skills the candidate has
                5. Reason for skill match score given

                IMPORTANT: Score the candidate on technical skills from 0-100.

                Output your analysis in JSON format with the following structure:
                {{
                    "skills_match_percentage": number,
                    "critical_skills_present": [list of skills],
                    "critical_skills_missing": [list of skills],
                    "bonus_skills": [list of skills],
                    "technical_score": number (0-100),
                    "detailed_assessment": string
                }}
            """),
            agent=self.technical_skills_agent,
            expected_output="An evaluation of technical experience in structured JSON format",
        )

        # Work Experience Assessment Task
        self.work_exp_task = Task(
            description=dedent(f"""
                Evaluate the candidate's work experience against the job requirements:
                {self.job_description}

                Analyze:
                1. Years of relevant experience
                2. Relevance of previous roles and responsibilities
                3. Achievements and projects that align with the job
                4. Career progression and growth
                5. Check that relevant skill is applied in previous role

                IMPORTANT: Score the candidate on work experience from 0-100.

                Output your analysis in JSON format with the following structure:
                {{
                    "years_relevant_experience": number,
                    "relevance_of_roles": string,
                    "key_achievements": [list of achievements],
                    "career_progression": string,
                    "experience_score": number (0-100),
                    "detailed_assessment": string
                }}
            """),
            agent=self.work_experience_agent,
            expected_output="An evaluation of work experience in structured JSON format",
        )

        # Education Assessment Task
        self.edu_task = Task(
            description=dedent(f"""
                Assess the candidate's educational qualifications against the job requirements:
                {self.job_description}

                Analyze:
                1. Degree relevance
                2. Institution reputation
                3. Specializations and majors
                4. Additional certifications and training

                IMPORTANT: Score the candidate on education from 0-100.

                Output your analysis in JSON format with the following structure:
                {{
                    "degree_relevance": string,
                    "institution_quality": string,
                    "relevant_specializations": [list of specializations],
                    "additional_certifications": [list of certifications],
                    "education_score": number (0-100),
                    "detailed_assessment": string
                }}
            """),
            agent=self.education_agent,
            expected_output="An evaluation of educational background in structured JSON format",
        )

        # Industry Fit Assessment Task
        self.industry_task = Task(
            description=dedent(f"""
                Evaluate the candidate's industry fit based on their profile:
                {self.job_description}

                Analyze:
                1. Industry-specific experience
                2. Knowledge of industry trends and practices
                3. Experience with industry-specific tools and methodologies
                4. Alignment with industry culture and values

                IMPORTANT: Score the candidate on industry fit from 0-100.

                Output your analysis in JSON format with the following structure:
                {{
                    "industry_experience": string,
                    "industry_knowledge": string,
                    "industry_tools_experience": [list of tools],
                    "culture_alignment": string,
                    "industry_score": number (0-100),
                    "detailed_assessment": string
                }}
            """),
            agent=self.industry_agent,
            expected_output="An evaluation of industry relevant experience in structured JSON format",
        )

        # Final Analysis Task
        self.final_analysis_task = Task(
            description=dedent(f"""
                Review all the assessments provided by the specialized agents.
                Synthesize the information to provide a comprehensive evaluation of the candidate.

                You will receive:
                - Technical Skills Assessment (score out of 100)
                - Work Experience Assessment (score out of 100)
                - Education Assessment (score out of 100)
                - Industry Fit Assessment (score out of 100)

                IMPORTANT: Using these individual assessments, determine:
                1. The candidate's key strengths (extract from high-scoring areas)
                2. Areas of concern (extract from low-scoring areas)
                3. Match category based on overall impression:
                   - "Strong Match" (average of all scores > 80)
                   - "Potential Match" (average of all scores 60-80)
                   - "Not a Match" (average of all scores < 60)
                4. A detailed justification for your recommendation
                5. A clear hiring recommendation

                DO NOT include overall score or weighted scores in your analysis.

                Provide your analysis in the following format:
                {{  "technical_score": number (0-100),
                    "technical_score_reason": string (detailed explanation for technical score),
                    "industry_score": number (0-100),
                    "industry_score_reason": string (detailed explanation for industry score),
                    "experience_score": number (0-100),
                    "experience_score_reason": string (detailed explanation for experience score),
                    "education_score": number (0-100),
                    "education_score_reason": string (detailed explanation for education score),
                    "key_strengths": [list of strengths],
                    "areas_of_concern": [list of concerns],
                    "match_category": string (category),
                    "justification": string,
                    "recommendation": string
                }}

                Be specific in your assessment and make a clear recommendation based on
                the job description: {self.job_description}
            """),
            agent=self.senior_analyst,
            expected_output="A final comprehensive evaluation in structured JSON format",
            context=[
                self.tech_skills_task,
                self.work_exp_task,
                self.edu_task,
                self.industry_task,
            ],
            output_pydantic=FinalRecommendation,
        )

        # Snapshot the pristine descriptions once, at build time. Previously this
        # was done lazily inside evaluate_candidate(), which meant the *first*
        # candidate's details leaked into the "original" template and were then
        # prepended to every subsequent candidate's prompt.
        self.original_task_descriptions = {
            "tech": self.tech_skills_task.description,
            "work": self.work_exp_task.description,
            "edu": self.edu_task.description,
            "industry": self.industry_task.description,
        }

    def setup_crew(self):
        """Set up the crew with the defined agents and tasks"""
        self.crew = Crew(
            agents=[
                self.technical_skills_agent,
                self.work_experience_agent,
                self.education_agent,
                self.industry_agent,
                self.senior_analyst,
            ],
            tasks=[
                self.tech_skills_task,
                self.work_exp_task,
                self.edu_task,
                self.industry_task,
                self.final_analysis_task,
            ],
            process=Process.sequential,
        )

    def evaluate_candidate(self, candidate_data):
        """
        Evaluate a single candidate against the job description

        Args:
            candidate_data (dict): Dictionary containing candidate information

        Returns:
            The raw Crew output for this candidate
        """

        def field(key):
            # The extractor returns "NA" for anything it can't find, and older
            # runs may be missing keys entirely - never raise a KeyError here.
            value = candidate_data.get(key, "NA")
            return "NA" if value is None or value == "" else value

        candidate_info = f"""
        Candidate Profile:
        Name: {field('Name')}
        Email: {field('Email')}
        Mobile: {field('Mobile number')}
        Skills: {field('Skills')}
        Total Experience: {field('Total experience in years')}
        Previous Work: {field('Work done in previous company')}
        College: {field('College name')}
        Degree: {field('Degree')}
        Designation: {field('Designation')}
        Companies: {field('Company names')}
        """

        # Reset task descriptions to the pristine template, then append this
        # candidate only.
        self.tech_skills_task.description = (
            self.original_task_descriptions["tech"] + f"\n\nCandidate Information:\n{candidate_info}"
        )
        self.work_exp_task.description = (
            self.original_task_descriptions["work"] + f"\n\nCandidate Information:\n{candidate_info}"
        )
        self.edu_task.description = (
            self.original_task_descriptions["edu"] + f"\n\nCandidate Information:\n{candidate_info}"
        )
        self.industry_task.description = (
            self.original_task_descriptions["industry"] + f"\n\nCandidate Information:\n{candidate_info}"
        )

        return self.crew.kickoff()


def _safe_float(value, default=0.0):
    """Coerce a model-supplied score to a float in the 0-100 range."""
    try:
        if value is None:
            return default
        number = float(value)
    except (ValueError, TypeError):
        return default
    if number != number:  # NaN
        return default
    return max(0.0, min(100.0, number))


def _parse_result(result):
    """Turn a Crew output into a plain dict, whatever shape it arrives in."""
    if result is None:
        return {}

    pydantic_obj = getattr(result, "pydantic", None)
    if pydantic_obj is not None:
        try:
            return pydantic_obj.model_dump()
        except AttributeError:
            return pydantic_obj.dict()

    json_dict = getattr(result, "json_dict", None)
    if json_dict:
        return json_dict

    raw = getattr(result, "raw", None) or str(result)
    try:
        return json.loads(raw)
    except Exception:
        # Greedy match so nested objects aren't truncated (the old non-greedy
        # pattern stopped at the first closing brace and lost most of the data).
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}


def _normalise_weights(weights):
    """Validate weights and rescale them to sum to 1."""
    if not weights:
        return dict(DEFAULT_WEIGHTS)

    cleaned = {key: _safe_float(weights.get(key), 0.0) for key in SCORE_COLUMNS}
    total = sum(cleaned.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {key: value / total for key, value in cleaned.items()}


def agentic_screener_v3(job_description, candidates_df, weights=None):
    """Score every candidate in `candidates_df` against `job_description`.

    Returns the input DataFrame with score, reason, category and recommendation
    columns appended, plus a weighted `Overall_Score`.
    """
    if candidates_df is None or candidates_df.empty:
        raise ValueError("No candidate data to screen - resume extraction returned no rows.")

    weights = _normalise_weights(weights)

    screener = ResumeScreeningSystem(job_description, build_llm())

    rows = []
    total = len(candidates_df)

    # Positional counter: the DataFrame index is not guaranteed to be a
    # 0-based integer range, so `idx + 1` could raise or print nonsense.
    for position, (_, row) in enumerate(candidates_df.iterrows(), start=1):
        candidate_data = row.to_dict()
        name = candidate_data.get("Name", "Unknown")
        print(f"Evaluating candidate {position}/{total}: {name}")

        try:
            parsed = _parse_result(screener.evaluate_candidate(candidate_data))
        except Exception as exc:
            print(f"Error evaluating {name}: {exc}")
            parsed = {}

        rows.append(
            {
                "Technical_Score": _safe_float(parsed.get("technical_score")),
                "Technical_Score_Reason": parsed.get("technical_score_reason", ""),
                "Experience_Score": _safe_float(parsed.get("experience_score")),
                "Experience_Score_Reason": parsed.get("experience_score_reason", ""),
                "Education_Score": _safe_float(parsed.get("education_score")),
                "Education_Score_Reason": parsed.get("education_score_reason", ""),
                "Industry_Score": _safe_float(parsed.get("industry_score")),
                "Industry_Score_Reason": parsed.get("industry_score_reason", ""),
                "Match_Category": parsed.get("match_category", "Not a Match"),
                "Key_Strengths": parsed.get("key_strengths", []),
                "Areas_of_Concern": parsed.get("areas_of_concern", []),
                "Justification": parsed.get("justification", ""),
                "Recommendation": parsed.get("recommendation", ""),
            }
        )

    # Join positionally instead of merging on Email. The old `pd.merge(...,
    # on='Email')` duplicated rows whenever two resumes shared an email or the
    # extractor returned "NA" for it, producing a cartesian blow-up and scores
    # attached to the wrong candidate.
    scores_df = pd.DataFrame(rows, index=candidates_df.index)
    final_result_df = pd.concat([candidates_df.copy(), scores_df], axis=1)

    final_result_df["Overall_Score"] = sum(
        final_result_df[column] * weights[column] for column in SCORE_COLUMNS
    ).round(2)

    # Flatten list-valued cells so the DataFrame can be written to Excel.
    for column in final_result_df.columns:
        final_result_df[column] = final_result_df[column].apply(
            lambda value: ", ".join(str(item) for item in value) if isinstance(value, list) else value
        )

    return final_result_df
