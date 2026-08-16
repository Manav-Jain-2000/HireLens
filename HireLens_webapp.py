"""HireLens - Streamlit front-end for the multi-agent resume screening system."""

import base64
import configparser
import os
import shutil
from datetime import datetime
from pathlib import Path

import pdfplumber
import plotly.express as px
import streamlit as st
from PIL import Image

from scripts.agentic_screener_function_v3 import agentic_screener_v3
from scripts.extractor_parser_v3 import standard_data_retriever_v3
from scripts.question_generator import generate_interview_questions

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CONFIG = {
    "base_path": "resume_folder_path",
    "resume_score_df": "resume_score/resume_result.xlsx",
}


def load_config(config_path="config.txt"):
    """Load configuration, resolving relative paths against the project root.

    Paths in config.txt used to be absolute and machine-specific, so the app
    crashed with a KeyError/FileNotFoundError on any other machine. Relative
    paths now work out of the box and absolute paths are still honoured.
    """
    parser = configparser.ConfigParser()
    parser.read(PROJECT_ROOT / config_path)

    folder = dict(DEFAULT_CONFIG)
    if parser.has_section("Folder"):
        for key, value in parser.items("Folder"):
            if value.strip():
                folder[key] = value.strip()

    resolved = {}
    for key, value in folder.items():
        path = Path(value)
        resolved[key] = str(path if path.is_absolute() else PROJECT_ROOT / path)

    # Make sure the directories we are about to write into actually exist.
    Path(resolved["base_path"]).mkdir(parents=True, exist_ok=True)
    Path(resolved["resume_score_df"]).parent.mkdir(parents=True, exist_ok=True)

    return resolved


CONFIG = load_config()
RESULTS_XLSX = CONFIG["resume_score_df"]
RESUME_ARCHIVE = CONFIG["base_path"]

TEMP_FOLDER = PROJECT_ROOT / "Temp_Uploads"
TEMP_FOLDER.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="HireLens", page_icon="🔍", layout="wide")


def _stretch(element=st.button):
    """Kwargs that make a widget fill its container, across Streamlit versions.

    `use_container_width` is deprecated (scheduled for removal after
    2025-12-31) and emits a warning on every call in recent versions; `width`
    only exists from Streamlit 1.49. Pick whichever the installed version has.
    """
    try:
        import inspect

        if "width" in inspect.signature(element).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


FULL_WIDTH = _stretch()

# ---------------------------------------------------------------- session state
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "investigate_file" not in st.session_state:
    st.session_state.investigate_file = None

if "is_processing_complete" not in st.session_state:
    st.session_state.is_processing_complete = False

if "resume_score_df" not in st.session_state:
    st.session_state.resume_score_df = None

# Clear the temp folder once per session rather than on every Streamlit rerun -
# the old top-level loop deleted freshly uploaded files whenever any widget was
# touched while processing was still in flight.
if "temp_cleared" not in st.session_state:
    for stale_file in TEMP_FOLDER.glob("*"):
        if stale_file.is_file():
            stale_file.unlink()
    st.session_state.temp_cleared = True

# ------------------------------------------------------------------------- CSS
st.markdown(
    """
        <style>
        /* Global Page Background */
        body, .stApp {
            background-color: #eaeaea;
            margin: 0;
            padding: 0;
            font-size: 1.4rem;
            color: #C8225A;
        }

        /* Dashboard Header Styling */
        .dashboard-header {
            background-color: #eaeaea;
            color: #C8225A;
            border-bottom: 2px solid #C8225A;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 1px 2px 4px rgba(0, 0, 0, 0.1);
            font-size: 0.9rem;
        }

        /* Metrics styling */
        .metrics-container {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .metric-box {
            background-color: #eaeaea;
            border-radius: 4px;
            padding: 0.5rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            text-align: center;
            flex: 1;
            min-width: 100px;
            border-left: 3px solid #3498db;
        }

        .metric-box h2 {
            margin: 0;
            font-size: 1.2rem;
            font-weight: 600;
            color: #C8225A;
        }

        .metric-box p {
            margin: 0.2rem 0 0 0;
            color: #C8225A;
            font-size: 0.7rem;
        }

        /* Card styling */
        .card {
            background-color: #eaeaea;
            border-radius: 4px;
            padding: 0.8rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }

        .card-header {
            padding-bottom: 0;
            margin-bottom: 0;
            display: flex;
            align-items: center;
        }

        .card-header h2 {
            margin: 0;
            color: #C8225A;
            font-size: 1.6rem;
            font-weight: 600;
        }

        /* File item styling */
        .file-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.4rem;
            border-radius: 4px;
            border: 1px solid #f1f1f1;
            margin-bottom: 0.2rem;
            transition: all 0.2s;
        }

        .file-item:hover {
            background-color: #dcdcdc;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }

        .file-info {
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .file-icon {
            font-size: 1rem;
            color: #C8225A;
        }

        .file-type {
            display: inline-block;
            padding: 0.1rem 0.3rem;
            border-radius: 3px;
            font-size: 0.7rem;
            color: #2c3e50;
            background-color: #dcdcdc;
        }

        /* Status badges - text was previously the same colour as the
           background, which made every badge unreadable. */
        .status-badge {
            padding: 0.2rem 0.5rem;
            border-radius: 1rem;
            font-size: 0.7rem;
            font-weight: 500;
        }

        .status-clean {
            background-color: #27ae60;
            color: #ffffff;
        }

        .status-sanctions {
            background-color: #e74c3c;
            color: #ffffff;
        }

        .status-pending {
            background-color: #bdc3c7;
            color: #2c3e50;
        }

        /* Buttons */
        .primary-button {
            background-color: #C8225A;
            color: #ffffff;
            padding: 0.3rem 0.8rem;
            border-radius: 4px;
            border: none;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.7rem;
        }

        .primary-button:hover {
            background-color: #a41c4a;
        }

        .secondary-button {
            background-color: #dcdcdc;
            color: #2c3e50;
            padding: 0.3rem 0.8rem;
            border-radius: 4px;
            border: none;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.7rem;
        }

        .secondary-button:hover {
            background-color: #cfcfcf;
        }

        /* Recent files in sidebar */
        .recent-file {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.4rem 0.6rem;
            background: rgba(200, 34, 90, 1);
            border-radius: 3px;
            margin-bottom: 0.3rem;
            color: #ecf0f1;
            transition: all 0.2s;
        }

        .recent-file:hover {
            background: rgba(164, 28, 74, 1);
        }

        /* Navigation styling */
        .nav-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.2rem 0.4rem;
            margin-bottom: 0.2rem;
            color: #ecf0f1;
            border-radius: 4px;
            transition: all 0.2s;
            cursor: pointer;
            font-size: 1rem;
        }

        .nav-item:hover, .nav-item.active {
            background-color: rgba(200, 34, 90, 1);
        }

        .nav-icon {
            width: 16px;
        }

        /* Investigation page */
        .investigation-header {
            background-color: #C8225A;
            color: #ffffff;
            padding: 0.8rem;
            border-radius: 6px;
            margin-bottom: 0.8rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
        }

        .investigation-actions {
            display: flex;
            gap: 0.5rem;
        }

        .action-approve {
            background-color: #27ae60;
            color: #ffffff;
            padding: 0.3rem 0.8rem;
            border-radius: 4px;
            border: none;
            cursor: pointer;
            font-size: 0.7rem;
        }

        .action-approve:hover {
            background-color: #1f8b4c;
        }

        .action-reject {
            background-color: #e74c3c;
            color: #ffffff;
            padding: 0.3rem 0.8rem;
            border-radius: 4px;
            border: none;
            cursor: pointer;
            font-size: 0.7rem;
        }

        .action-reject:hover {
            background-color: #c0392b;
        }

        /* Sidebar title */
        .sidebar-title {
            font-size: 2rem;
            font-weight: 800;
            color: #ecf0f1 !important;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid rgba(200, 34, 90, 1);
        }

        /* Sidebar container styling */
        [data-testid="stSidebar"] {
            background-color: #1e2a38;
            padding: 1rem;
        }

        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #C8225A;
            font-size: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------- utilities
def create_timestamped_copy(temp_folder, destination_folder):
    """Copy the contents of `temp_folder` into a new timestamped subfolder.

    Args:
        temp_folder (str | Path): Source folder whose contents will be copied.
        destination_folder (str | Path): Where the new timestamped folder goes.

    Returns:
        str: Path of the newly created timestamped folder.
    """
    os.makedirs(destination_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_folder_path = os.path.join(str(destination_folder), f"datafolder_{timestamp}")
    # exist_ok guards against two runs landing in the same second.
    os.makedirs(new_folder_path, exist_ok=True)

    if os.path.exists(temp_folder):
        for item in os.listdir(temp_folder):
            src_path = os.path.join(str(temp_folder), item)
            dst_path = os.path.join(new_folder_path, item)

            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)

    return new_folder_path


def get_base64_image(image_path):
    """Return (base64_data, mime_type) for an image, or ("", "") if unreadable."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
    except (FileNotFoundError, OSError):
        return "", ""

    suffix = Path(image_path).suffix.lower()
    # The header previously declared image/jpeg for what is actually a PNG.
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
    }.get(suffix, "image/png")
    return encoded_string, mime


def show_pdf_or_image(file_path):
    """Display a PDF or image file in Streamlit."""
    if not file_path:
        st.info("No file path provided.")
        return

    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == ".pdf":
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
            'width="100%" height="600" type="application/pdf"></iframe>',
            unsafe_allow_html=True,
        )
        # Several browsers block base64 PDF iframes outright, leaving a blank
        # box - always offer a direct download as a fallback.
        st.download_button(
            "📄 Open / download this resume",
            data=pdf_bytes,
            file_name=os.path.basename(file_path),
            mime="application/pdf",
            key=f"dl_{file_path}",
        )
    elif file_ext in [".jpg", ".jpeg", ".png", ".gif"]:
        st.image(Image.open(file_path), caption=os.path.basename(file_path), **FULL_WIDTH)
    else:
        st.info("Unsupported file type. Only PDF and image files are supported.")


def extract_pdf_text(source):
    """Extract all text from a PDF path or uploaded file object."""
    text = ""
    with pdfplumber.open(source) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def save_results(df):
    """Persist the scored DataFrame, surfacing (not swallowing) write errors."""
    try:
        df.to_excel(RESULTS_XLSX, index=False)
        return True
    except Exception as exc:
        st.error(f"Could not save results to {RESULTS_XLSX}: {exc}")
        return False


def update_recommendation(row_label, status):
    """Set a candidate's recommendation status by DataFrame row label.

    Matching used to be done on candidate *name*, which silently updated the
    wrong row (or several rows) whenever two resumes shared a name or the
    extractor returned "NA".
    """
    df = st.session_state.resume_score_df
    if df is None or row_label is None or row_label not in df.index:
        st.error("Could not find that candidate to update.")
        return

    if "Recommendation_Status" not in df.columns:
        df["Recommendation_Status"] = "Pending"

    df.at[row_label, "Recommendation_Status"] = status
    if save_results(df):
        st.success(f"Candidate marked as {status}")


# --------------------------------------------------------------------- sidebar
def sidebar_info():
    st.markdown(
        """
        <div class="sidebar-title">
            <i class="fas fa-search-location"></i> HireLens
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="nav-item active">
            <i class="fas fa-home nav-icon"></i>
            Dashboard
        </div>

        <div class="nav-item">
            <i class="fas fa-upload nav-icon"></i>
            Upload Documents
        </div>

        <div class="nav-item">
            <i class="fas fa-file nav-icon"></i>
            Reports
        </div>

        <div class="nav-item">
            <i class="fas fa-cog nav-icon"></i>
            Settings
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="padding: 2rem 1rem; margin-top: 2rem;">
            <h3 style="color: #C8225A; border-bottom: 2px solid rgba(200, 34, 90, 1); padding-bottom: 0.5rem;">
                <i class="fas fa-clock"></i> Current Run
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for file_info in reversed(st.session_state.processed_files):
        file_name = file_info.get("file_name", "Unknown file")
        # Only add an ellipsis when the name was actually truncated.
        display_name = file_name if len(file_name) <= 20 else file_name[:20] + "..."
        st.markdown(
            f"""
            <div class="recent-file">
                <i class="fas fa-file-pdf file-icon"></i>
                {display_name}
            </div>
            """,
            unsafe_allow_html=True,
        )


with st.sidebar:
    sidebar_info()


# ----------------------------------------------------------- investigation page
def investigation_page():
    st.header("Resume Viewer")

    file_path = st.session_state.investigate_file
    scored = st.session_state.resume_score_df

    if st.button("← Back to Dashboard", key="back_top"):
        st.session_state.investigate_file = None
        st.rerun()

    col1, col2 = st.columns([4, 1])

    with col1:
        show_pdf_or_image(file_path)

        candidate_resume_info = ""
        if file_path and os.path.exists(file_path) and file_path.lower().endswith(".pdf"):
            try:
                candidate_resume_info = extract_pdf_text(file_path)
            except Exception as exc:
                st.warning(f"Could not read text from this resume: {exc}")

        job_description = ""
        for file_info in st.session_state.processed_files:
            if file_info.get("Path") == file_path:
                job_description = file_info.get("job_description", "")
                break

        # Persist generated questions per candidate across reruns.
        candidate_key = f"q_{file_path}"
        st.session_state.setdefault(candidate_key, "")

        if st.button("🧠 Create Question Set", **FULL_WIDTH):
            with st.spinner("Generating custom questions..."):
                try:
                    st.session_state[candidate_key] = generate_interview_questions(
                        job_description, candidate_resume_info
                    )
                except Exception as exc:
                    st.error(f"Question generation failed: {exc}")

        questionaire = st.session_state[candidate_key]
        if questionaire:
            st.subheader("📋 Interview Question Set")
            st.markdown(questionaire)
            st.download_button(
                label="📥 Download as Markdown",
                data=questionaire,
                file_name="interview_questions.md",
                mime="text/markdown",
                **FULL_WIDTH,
            )

    with col2:
        st.subheader("Resume Details")

        # Guard against an empty/None results frame - iterating it directly
        # raised AttributeError when the page was opened before processing.
        candidate_row = None
        candidate_label = None
        if scored is not None and not scored.empty and "file_path" in scored.columns:
            matches = scored.index[scored["file_path"] == file_path].tolist()
            if matches:
                candidate_label = matches[0]
                candidate_row = scored.loc[candidate_label]

        if candidate_row is not None:
            st.write(f"**Name:** {candidate_row.get('Name', 'Unknown')}")
            for label, column in [
                ("Overall Score", "Overall_Score"),
                ("Technical Score", "Technical_Score"),
                ("Experience Score", "Experience_Score"),
                ("Education Score", "Education_Score"),
                ("Industry Score", "Industry_Score"),
            ]:
                value = candidate_row.get(column)
                if value is not None and value == value:  # skip missing / NaN
                    try:
                        st.write(f"**{label}:** {float(value):.1f}/100")
                    except (TypeError, ValueError):
                        st.write(f"**{label}:** {value}")

            st.write(f"**Current Status:** {candidate_row.get('Recommendation_Status', 'Pending')}")

            if st.button("👍 Recommend Candidate", **FULL_WIDTH):
                update_recommendation(candidate_label, "Recommended")
                st.rerun()

            if st.button("👎 Reject Candidate", **FULL_WIDTH):
                update_recommendation(candidate_label, "Rejected")
                st.rerun()
        else:
            st.write("Candidate information not found")


# -------------------------------------------------------------- upload + score
def render_upload_form():
    """Draw the upload/weights form. Returns True if processing just finished."""
    st.markdown('<div class="card-header"><h2>Upload Documents</h2></div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True)
    uploaded_jd = st.file_uploader("Upload Job Description", type=["pdf"], accept_multiple_files=False)

    st.markdown(
        """
        <div class="card-header">
            <h2>Scoring Weights</h2>
        </div>
        <p>Adjust the importance of each category (total must equal 100)</p>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        technical_weight = st.slider(
            "Technical Skills Weight", 0, 100, 35, 5, help="Weight given to technical skills matching"
        )
        experience_weight = st.slider(
            "Experience Weight", 0, 100, 30, 5, help="Weight given to relevant experience"
        )
    with col2:
        education_weight = st.slider(
            "Education Weight", 0, 100, 15, 5, help="Weight given to education qualifications"
        )
        industry_weight = st.slider(
            "Industry Weight", 0, 100, 20, 5, help="Weight given to industry relevance"
        )

    total_weight = technical_weight + experience_weight + education_weight + industry_weight
    if total_weight != 100:
        st.warning(f"Total weight is {total_weight}%. Please adjust to equal 100%.")
    else:
        st.success("Weights are properly distributed (100%)")

    if not st.button("🚀 Process Documents", **FULL_WIDTH):
        return False

    if total_weight != 100:
        st.error("Please adjust weights to sum to exactly 100% before processing.")
        return False

    if uploaded_jd is None:
        st.warning("Please upload a job description.")
        return False

    if not uploaded_files:
        st.warning("Please upload at least one resume file.")
        return False

    try:
        job_description = extract_pdf_text(uploaded_jd)
    except Exception as exc:
        st.error(f"Error extracting text from job description: {exc}")
        return False

    if not job_description.strip():
        st.error("No text could be extracted from the job description PDF (is it a scan?).")
        return False

    st.success("Job description extracted successfully!")
    with st.expander("View Job Description Preview"):
        st.write(job_description[:500] + "..." if len(job_description) > 500 else job_description)

    with st.spinner("Processing resumes..."):
        st.session_state.processed_files = []
        for uploaded_file in uploaded_files:
            file_path = TEMP_FOLDER / uploaded_file.name
            file_path.write_bytes(uploaded_file.getbuffer())
            st.session_state.processed_files.append(
                {
                    "file_name": uploaded_file.name,
                    "Path": str(file_path),
                    "Status": "Processing",
                    "verified": False,
                    "job_description": job_description,
                    "verification_status": "Pending",
                }
            )

        try:
            target_folder = create_timestamped_copy(TEMP_FOLDER, RESUME_ARCHIVE)
            for file_info in st.session_state.processed_files:
                file_info["Path"] = os.path.join(
                    target_folder, os.path.basename(file_info["Path"])
                )

            weights = {
                "Technical_Score": technical_weight / 100,
                "Experience_Score": experience_weight / 100,
                "Education_Score": education_weight / 100,
                "Industry_Score": industry_weight / 100,
            }

            standardised_data = standard_data_retriever_v3(target_folder)
            if standardised_data.empty:
                st.error("No resume data could be extracted from the uploaded PDFs.")
                return False

            resume_score = agentic_screener_v3(job_description, standardised_data, weights)
        except Exception as exc:
            st.error(f"Error processing resumes: {exc}")
            st.session_state.is_processing_complete = False
            return False

    if "Recommendation_Status" not in resume_score.columns:
        resume_score["Recommendation_Status"] = "Pending"

    # Attach scores back to the upload list by file path, not list position -
    # the extractor walks the folder alphabetically, so positional matching
    # showed one candidate's score next to another candidate's file.
    scores_by_path = resume_score.set_index("file_path") if "file_path" in resume_score.columns else None
    for file_info in st.session_state.processed_files:
        if scores_by_path is not None and file_info["Path"] in scores_by_path.index:
            row = scores_by_path.loc[file_info["Path"]]
            if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
                row = row.iloc[0]  # duplicate paths - take the first match
            file_info["Overall_Score"] = row.get("Overall_Score", 0)
            file_info["Name"] = row.get("Name", "")
        file_info["Status"] = "Processed"

    st.session_state.resume_score_df = resume_score
    st.session_state.is_processing_complete = True
    save_results(resume_score)
    return True


# ------------------------------------------------------------------- dashboard
def render_results():
    resume_score = st.session_state.resume_score_df

    if st.button("⬅️ Process New Files", **FULL_WIDTH):
        st.session_state.is_processing_complete = False
        st.session_state.resume_score_df = None
        st.session_state.processed_files = []
        st.session_state.investigate_file = None
        st.rerun()

    if resume_score.empty:
        st.info("No candidates were scored.")
        return

    st.subheader("Key Performance Indicators")

    scores = resume_score["Overall_Score"].dropna()
    total_resumes = len(resume_score)
    top_10_percentile = scores.quantile(0.9) if not scores.empty else 0.0
    avg_score = scores.mean() if not scores.empty else 0.0
    median_score = scores.median() if not scores.empty else 0.0

    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
    colA, colB, colC, colD = st.columns(4)
    colA.write(f"<h2>{total_resumes}</h2><p>Total Resumes</p>", unsafe_allow_html=True)
    colB.write(f"<h2>{top_10_percentile:.1f}</h2><p>Top 10% Score</p>", unsafe_allow_html=True)
    colC.write(f"<h2>{avg_score:.1f}</h2><p>Average Score</p>", unsafe_allow_html=True)
    colD.write(f"<h2>{median_score:.1f}</h2><p>Median Score</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Resume Analysis Visualizations")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_hist = px.histogram(
            resume_score,
            x="Overall_Score",
            nbins=10,
            labels={"Overall_Score": "Score", "count": "Number of Candidates"},
            color_discrete_sequence=["#3366cc"],
        )
        fig_hist.update_traces(marker=dict(line=dict(width=0.6)))
        fig_hist.update_layout(
            title=dict(text="Distribution", font=dict(size=22, color="#C8225A")),
            xaxis=dict(
                title="Score",
                tickmode="linear",
                dtick=5,
                gridcolor="white",
                title_font=dict(size=16, color="#C8225A"),
                tickfont=dict(size=14, color="#C8225A"),
            ),
            yaxis=dict(
                title="Number of Candidates",
                title_font=dict(size=18, color="#C8225A"),
                tickfont=dict(size=16, color="#C8225A"),
            ),
            showlegend=False,
            plot_bgcolor="#eaeaea",
            paper_bgcolor="#eaeaea",
        )
        st.plotly_chart(fig_hist, **FULL_WIDTH)

    with chart_col2:
        if "Match_Category" in resume_score.columns:
            match_counts = resume_score["Match_Category"].value_counts().reset_index()
            match_counts.columns = ["Match_Category", "Count"]

            fig_pie = px.pie(
                match_counts,
                values="Count",
                names="Match_Category",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(
                title=dict(text="Match Summary", font=dict(size=22, color="#C8225A")),
                legend=dict(font=dict(size=20, color="#C8225A")),
                plot_bgcolor="#eaeaea",
                paper_bgcolor="#eaeaea",
            )
            st.plotly_chart(fig_pie, **FULL_WIDTH)
        else:
            st.warning("Match Category data not available for pie chart visualization.")

    st.subheader("Resume Screening Results")
    try:
        # background_gradient needs matplotlib; fall back to a plain table.
        styled_df = resume_score.style.background_gradient(
            subset=["Overall_Score"], cmap="RdYlGn", vmin=0, vmax=100
        )
        st.dataframe(styled_df, **FULL_WIDTH)
    except Exception:
        st.dataframe(resume_score, **FULL_WIDTH)

    if os.path.exists(RESULTS_XLSX):
        # Read inside a context manager - the old `open(...).read()` leaked a
        # file handle on every rerun and kept the workbook locked on Windows.
        with open(RESULTS_XLSX, "rb") as fh:
            st.download_button(
                label="📥 Download Results as Excel",
                data=fh.read(),
                file_name="resume_screening_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    st.subheader("Top Candidates")

    if "Recommendation_Status" not in resume_score.columns:
        resume_score["Recommendation_Status"] = "Pending"

    sorted_candidates = resume_score.sort_values(by="Overall_Score", ascending=False)

    header_cols = st.columns([3, 2, 3, 2, 3])
    for col, label in zip(header_cols, ["Name", "Overall Score", "Recommendation", "View Resume", "Action"]):
        col.markdown(f"**{label}**")
    st.markdown("---")

    pending_action = None
    for idx, candidate in sorted_candidates.iterrows():
        candidate_name = candidate.get("Name", f"Candidate {idx}")
        candidate_score = candidate.get("Overall_Score", 0) or 0
        recommendation_status = candidate.get("Recommendation_Status", "Pending")
        resume_path = candidate.get("file_path")

        row_cols = st.columns([3, 2, 3, 2, 3])
        row_cols[0].write(candidate_name)

        score_color = "green" if candidate_score >= 75 else "orange" if candidate_score >= 50 else "red"
        row_cols[1].markdown(
            f"<span style='color:{score_color};'>{float(candidate_score):.1f}</span>",
            unsafe_allow_html=True,
        )

        status_color = (
            "green" if recommendation_status == "Recommended"
            else "red" if recommendation_status == "Rejected"
            else "gray"
        )
        row_cols[2].markdown(
            f"<span style='color:{status_color};'>{recommendation_status}</span>",
            unsafe_allow_html=True,
        )

        if resume_path and os.path.exists(str(resume_path)):
            if row_cols[3].button("View", key=f"view_{idx}"):
                st.session_state.investigate_file = str(resume_path)
                st.rerun()
        else:
            row_cols[3].write("N/A")

        button_col1, button_col2 = row_cols[4].columns(2)
        if button_col1.button("Approve👍", key=f"rec_{idx}"):
            pending_action = (idx, "Recommended")
        if button_col2.button("Reject👎", key=f"rej_{idx}"):
            pending_action = (idx, "Rejected")

        st.markdown("---")

    if pending_action:
        update_recommendation(*pending_action)
        st.rerun()


def main_dashboard():
    logo_base64, logo_mime = get_base64_image(PROJECT_ROOT / "Assets" / "logo.png")
    logo_html = (
        f'<img src="data:{logo_mime};base64,{logo_base64}" alt="Logo" style="height: 60px;">'
        if logo_base64
        else ""
    )
    st.markdown(
        f"""
        <div class="dashboard-header" style="display: flex; align-items: center; justify-content: space-between; padding: 0;">
            <h1 style="margin: 0;"> HireLens</h1>
            {logo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # NOTE: no blanket try/except around this block. st.rerun() works by raising
    # RerunException, which subclasses Exception - the old catch-all swallowed it
    # and reported "Error processing resumes:" immediately after a *successful*
    # run, then reset is_processing_complete back to False.
    if not st.session_state.is_processing_complete:
        if render_upload_form():
            st.rerun()

    if st.session_state.is_processing_complete and st.session_state.resume_score_df is not None:
        render_results()


if st.session_state.investigate_file:
    investigation_page()
else:
    main_dashboard()
