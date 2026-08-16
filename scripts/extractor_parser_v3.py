"""Resume PDF extraction and standardisation.

Reads every PDF in a folder, sends the raw text to Azure OpenAI, and returns a
DataFrame with one row per resume and a fixed set of columns.
"""

import json
import os
import re
from datetime import datetime
from functools import lru_cache

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

# PyPDF2 is unmaintained and fails to import on newer Python versions; pypdf is
# its successor. Prefer pypdf, fall back so existing installs keep working.
try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    from PyPDF2 import PdfReader

load_dotenv()

# Columns every returned row is guaranteed to have, so downstream code can rely
# on them existing even when the model omits a field.
EXPECTED_FIELDS = [
    "Name",
    "Email",
    "Mobile number",
    "Skills",
    "Total experience in years",
    "Work done in previous company",
    "College name",
    "Degree",
    "Designation",
    "Company names",
]

DEFAULT_API_VERSION = "2024-12-01-preview"


@lru_cache(maxsize=1)
def get_client():
    """Build the Azure OpenAI client lazily.

    Doing this at import time meant a missing/incorrect .env crashed the whole
    Streamlit app on start-up with an opaque error.
    """
    api_key = os.getenv("AZURE_API_KEY")
    endpoint = os.getenv("AZURE_API_BASE")

    missing = [
        name
        for name, value in (("AZURE_API_KEY", api_key), ("AZURE_API_BASE", endpoint))
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing Azure OpenAI credentials: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in your values."
        )

    return AzureOpenAI(
        api_key=api_key,
        api_version=os.getenv("AZURE_API_VERSION") or DEFAULT_API_VERSION,
        azure_endpoint=endpoint,
    )


def get_deployment_name():
    """Azure deployment name for the extraction model."""
    return os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini")


def text_extractor(pdf_path):
    """Return the plain text of a PDF.

    `extract_text()` returns None for pages with no text layer (scans, images),
    which previously raised `TypeError: can only concatenate str (not
    "NoneType") to str` and aborted the whole run.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # a single corrupt page shouldn't kill the file
            print(f"Warning: could not read a page of {pdf_path}: {exc}")
    return " ".join(pages).strip()


def _extract_json(response_text):
    """Pull the first JSON object out of a model response.

    The old pattern was non-greedy (`\\{[\\s\\S]*?\\}`) so it stopped at the first
    closing brace and silently truncated any response containing a nested
    object. This walks the braces instead.
    """
    text = response_text.strip()

    # Strip ```json ... ``` fences if the model added them.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No valid JSON object found in model response.")

    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])

    raise ValueError("No valid JSON object found in model response.")


def extract_resume_info(resume_text, current_date_str):
    """Ask the model for structured fields from one resume."""
    prompt = f"""
Today's date is {current_date_str}.

Extract the following information from the resume. For any field not found, return "NA":
- Name
- Email
- Mobile number
- Skills
- Total experience in years (if 'Present' is mentioned, treat it as up to today's date)
- Work done in previous company
- College name
- Degree
- Designation
- Company names

Resume:
{resume_text}

Return the result ONLY in valid JSON format. Do not include any markdown, explanation, or headers.
"""

    response = get_client().chat.completions.create(
        model=get_deployment_name(),
        messages=[
            {"role": "system", "content": "You extract structured data from resumes."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return _extract_json(response.choices[0].message.content or "")


def _normalise(record):
    """Force every expected field to exist and be a plain string."""
    clean = {}
    for field in EXPECTED_FIELDS:
        value = record.get(field, "NA")
        if value is None or value == "":
            value = "NA"
        elif isinstance(value, (list, tuple)):
            value = ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            value = json.dumps(value)
        clean[field] = str(value)
    return clean


def standard_data_retriever_v3(folder_path):
    """Parse every PDF in `folder_path` into a standardised DataFrame."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Resume folder does not exist: {folder_path}")

    resume_data = []
    current_date_str = datetime.now().strftime("%B %d, %Y")

    pdf_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")

    for filename in pdf_files:
        file_path = os.path.join(folder_path, filename)
        try:
            resume_text = text_extractor(file_path)
            if not resume_text:
                print(f"Skipping {file_path}: no extractable text (is it a scan?).")
                continue

            resume_json = _normalise(extract_resume_info(resume_text, current_date_str))
            resume_json["file_path"] = file_path
            resume_json["source_file"] = filename
            resume_data.append(resume_json)
        except Exception as exc:
            print(f"Error processing {file_path}: \n\n{exc}\n")

    if not resume_data:
        # An empty DataFrame with no columns broke every downstream `df['Name']`
        # lookup with a confusing KeyError. Return the right shape instead.
        return pd.DataFrame(columns=EXPECTED_FIELDS + ["file_path", "source_file"])

    return pd.DataFrame(resume_data)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python extractor_parser_v3.py <folder_of_pdfs>")
        raise SystemExit(1)
    print(standard_data_retriever_v3(sys.argv[1]).to_string())
