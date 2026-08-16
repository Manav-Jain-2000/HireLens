"""CrewAI crew that writes a tailored interview question set for one candidate."""

import os

from crewai import LLM, Agent, Crew, Task
from dotenv import load_dotenv

load_dotenv()


def build_llm():
    """LLM used by every agent. Model name is configurable via .env."""
    return LLM(model=os.getenv("CREWAI_LLM_MODEL", "azure/gpt-4o-mini"))


def build_interview_question_crew():
    """Construct a fresh crew.

    This used to run at import time, which meant importing the module crashed
    the whole Streamlit app when credentials were missing, and every candidate
    reused one long-lived Crew whose task state carried over between runs.
    """
    llm = build_llm()

    # Define the Case Study Researcher Agent
    candidate_assessor = Agent(
        role="Case Study Researcher",
        goal="Analyze job requirements and candidate profiles to design relevant case study questions",
        backstory="You're an expert technical recruiter specializing in designing effective interview processes. "
                  "You thoroughly analyze job descriptions to identify key technical skills, experience levels, "
                  "and competencies required. You then compare candidate profiles against these requirements "
                  "to identify strengths and areas to probe. Based on this analysis, you create tailored "
                  "case study questions for the post-technical interview round that specifically test "
                  "the candidate's ability to apply their skills to real-world scenarios relevant to the job.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    # Define the Technical Question Designer Agent
    technical_question_designer = Agent(
        role="Technical Question Designer",
        goal="Design targeted technical questions based on job requirements and candidate experience",
        backstory="You're an expert technical interviewer with extensive experience across multiple domains. "
                  "Your specialty is creating precise technical questions that assess a candidate's depth of "
                  "knowledge in specific technologies, frameworks, and concepts. You carefully analyze both "
                  "job requirements and a candidate's claimed experience to design questions that validate "
                  "their expertise, reveal their problem-solving approach, and identify the boundaries of "
                  "their knowledge. Your questions range from foundational concepts to advanced scenarios, "
                  "always tailored to the specific technologies and experience level mentioned in the candidate's profile.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    # Define the Question Reviewer Agent
    question_reviewer = Agent(
        role="Interview Question Reviewer",
        goal="Evaluate and refine technical and case study questions to ensure accuracy, relevance, and effectiveness",
        backstory="You're a senior technical hiring manager with extensive experience in technical interviews "
                  "across multiple domains. Your expertise lies in evaluating interview questions to ensure they "
                  "are technically accurate, job-relevant, fair to candidates, and effective at revealing true "
                  "capabilities. You've developed a keen eye for questions that might be too theoretical, "
                  "ambiguous, or disconnected from real-world job responsibilities. You provide actionable "
                  "feedback to improve questions and ensure the interview process accurately assesses "
                  "candidates while providing a positive candidate experience.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    # Task 1: Design Case Study Question
    design_case_study = Task(
        description="Analyze the job description and candidate profile to design a relevant case study question "
                    "that assesses the candidate's ability to apply their skills to a real-world scenario. "
                    "\n\nAnalysis Steps:"
                    "\n- Identify key responsibilities and required competencies from the job description"
                    "\n- Consider the candidate's background and experience from their profile"
                    "\n- Design a case study that tests relevant problem-solving abilities"
                    "\n\nCase Study Design Guidelines:"
                    "\n- Create a realistic scenario relevant to the job's domain"
                    "\n- Ensure it tests multiple competencies mentioned in the job description"
                    "\n- Make it appropriate for the candidate's experience level"
                    "\n- Include clear objectives and constraints"
                    "\n- Provide evaluation criteria for assessing the candidate's response"
                    "\n\nJob Description: {job_description}"
                    "\nCandidate Profile: {candidate_profile}",
        agent=candidate_assessor,
        expected_output="A comprehensive case study question with clear context, requirements, and evaluation criteria.",
    )

    # Task 2: Design Technical Questions
    design_technical_questions = Task(
        description="Analyze the job description and candidate profile to design a set of targeted technical "
                    "questions that assess the candidate's knowledge and experience in required technologies. "
                    "\n\nAnalysis Steps:"
                    "\n- Identify all technical skills, tools, and technologies listed in the job description"
                    "\n- Assess the candidate's claimed experience level with each technology"
                    "\n- Note any potential gaps or areas where verification is needed"
                    "\n\nQuestion Design Guidelines:"
                    "\n- Create 5-8 technical questions covering key required technologies"
                    "\n- Include a mix of:"
                    "\n  * Fundamental concept questions to verify basic understanding"
                    "\n  * Experience-based questions related to previous work"
                    "\n  * Problem-solving scenarios to assess applied knowledge"
                    "\n  * Advanced questions appropriate to their claimed experience level"
                    "\n- For each question, include:"
                    "\n  * The question itself"
                    "\n  * What skill/technology it assesses"
                    "\n  * What a good answer should demonstrate"
                    "\n  * Follow-up questions based on possible responses"
                    "\n\nJob Description: {job_description}"
                    "\nCandidate Profile: {candidate_profile}",
        agent=technical_question_designer,
        expected_output="A comprehensive set of targeted technical questions with clear evaluation criteria, "
                        "organized by skill area, with annotations explaining what each question aims to assess.",
    )

    # Task 3: Review Interview Questions
    review_interview_questions = Task(
        description="Review the combined set of interview questions to verify they align with the job description "
                    "and candidate's technical skills, experience, and education. Perform a thorough proofreading "
                    "and formatting check, then output ONLY the final questions in a clean, document-ready format."
                    "\n\nReview Process (internal only, DO NOT include in output):"
                    "\n- Verify each question directly relates to skills mentioned in the job description"
                    "\n- Confirm questions are appropriate for the candidate's experience level"
                    "\n- Check that questions cover the key technical skills from the candidate's profile"
                    "\n- Proofread for clarity, grammar, and professional tone"
                    "\n\nOutput Requirements:"
                    "\n- Format as a clean Markdown document with a simple title including the candidate's name"
                    "\n- Include ONLY the final interview questions (maximum 8): 6 technical and 2 case study "
                    "(the case study questions should be elaborated)"
                    "\n- Use clear, consistent numbering and organization"
                    "\n- Present questions ready for direct use in an interview"
                    "\n- EXCLUDE all review notes, feedback, commentary, analysis, or explanations"
                    "\n- DO NOT mention strengths or weaknesses of questions"
                    "\n- DO NOT include sections about question quality or improvement suggestions"
                    "\n\nJob Description: {job_description}"
                    "\nCandidate Profile: {candidate_profile}",
        agent=question_reviewer,
        # Tasks run in the order listed below, so both context tasks have
        # already completed by the time this one starts.
        context=[design_case_study, design_technical_questions],
        expected_output="A comprehensive question set combining technical and case study questions",
    )

    return Crew(
        agents=[candidate_assessor, technical_question_designer, question_reviewer],
        tasks=[design_case_study, design_technical_questions, review_interview_questions],
        verbose=True,
    )


def generate_interview_questions(job_description, candidate_profile):
    """Generate a Markdown interview question set for one candidate."""
    if not job_description or not str(job_description).strip():
        return "**Could not generate questions:** no job description is associated with this candidate."
    if not candidate_profile or not str(candidate_profile).strip():
        return "**Could not generate questions:** no readable text was extracted from this resume."

    crew = build_interview_question_crew()
    result = crew.kickoff(
        inputs={
            "job_description": job_description,
            "candidate_profile": candidate_profile,
        }
    )
    return getattr(result, "raw", None) or str(result)
