"""
PDF Resume Generator — uses ReportLab (free, open source).

Creates a clean, professional resume PDF with:
- Header with name, university, branch
- Sections: Summary, Skills, Experience, Education, Projects, Achievements
- Score badges (confidence, expression, eye contact)
- Clean typography and layout

Only includes sections that have data (no empty sections).
"""

import os
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from app.models.schemas import ResumeData, AudioAnalysis, FaceAnalysis


# Colors
PRIMARY = HexColor("#1a1a2e")
ACCENT = HexColor("#4a5ae8")
LIGHT_ACCENT = HexColor("#e8eaff")
TEXT_DARK = HexColor("#1a1a2e")
TEXT_MED = HexColor("#4a4a6a")
TEXT_LIGHT = HexColor("#7a7a9a")
DIVIDER = HexColor("#e0e2f0")

TEMP_DIR = tempfile.gettempdir()


def create_styles():
    """Create custom paragraph styles."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "ResumeName", fontSize=22, leading=28, textColor=PRIMARY,
        fontName="Helvetica-Bold", alignment=TA_LEFT, spaceAfter=2*mm,
    ))
    styles.add(ParagraphStyle(
        "ResumeSubtitle", fontSize=10, leading=14, textColor=TEXT_MED,
        fontName="Helvetica", alignment=TA_LEFT, spaceAfter=6*mm,
    ))
    styles.add(ParagraphStyle(
        "SectionTitle", fontSize=12, leading=16, textColor=ACCENT,
        fontName="Helvetica-Bold", alignment=TA_LEFT, spaceBefore=6*mm,
        spaceAfter=3*mm,
    ))
    styles.add(ParagraphStyle(
        "BodyText2", fontSize=10, leading=14, textColor=TEXT_DARK,
        fontName="Helvetica", alignment=TA_JUSTIFY, spaceAfter=2*mm,
    ))
    styles.add(ParagraphStyle(
        "SmallText", fontSize=9, leading=12, textColor=TEXT_LIGHT,
        fontName="Helvetica", alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        "BulletText", fontSize=10, leading=14, textColor=TEXT_DARK,
        fontName="Helvetica", alignment=TA_LEFT, leftIndent=12,
        spaceAfter=1.5*mm,
    ))
    styles.add(ParagraphStyle(
        "ExpTitle", fontSize=10.5, leading=14, textColor=TEXT_DARK,
        fontName="Helvetica-Bold", alignment=TA_LEFT,
    ))
    return styles


def generate_pdf(
    resume: ResumeData,
    audio: AudioAnalysis,
    face: FaceAnalysis,
    user_name: str,
    user_university: str,
    user_branch: str,
    job_id: str,
) -> str:
    """
    Generate a polished PDF resume.
    Returns path to the generated PDF file.
    """
    pdf_path = os.path.join(TEMP_DIR, f"{job_id}_resume.pdf")
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = create_styles()
    story = []

    # ---- HEADER ----
    story.append(Paragraph(user_name, styles["ResumeName"]))
    subtitle_parts = []
    if user_university:
        subtitle_parts.append(user_university)
    if user_branch:
        subtitle_parts.append(user_branch)
    if subtitle_parts:
        story.append(Paragraph(" · ".join(subtitle_parts), styles["ResumeSubtitle"]))

    # Score badges
    scores = []
    if audio.confidence_score > 0:
        scores.append(f"Confidence: {audio.confidence_score:.0f}%")
    if face.avg_expression_score > 0:
        scores.append(f"Expression: {face.avg_expression_score:.0f}%")
    if face.avg_eye_contact_score > 0:
        scores.append(f"Eye Contact: {face.avg_eye_contact_score:.0f}%")
    if audio.speaking_rate_wpm > 0:
        scores.append(f"Speaking Rate: {audio.speaking_rate_wpm:.0f} WPM")
    if scores:
        story.append(Paragraph("   |   ".join(scores), styles["SmallText"]))

    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER))

    # ---- SUMMARY ----
    if resume.summary:
        story.append(Paragraph("PROFESSIONAL SUMMARY", styles["SectionTitle"]))
        story.append(Paragraph(resume.summary, styles["BodyText2"]))

    # ---- SKILLS ----
    if resume.skills:
        story.append(Paragraph("SKILLS", styles["SectionTitle"]))
        skills_text = "   •   ".join(resume.skills)
        story.append(Paragraph(skills_text, styles["BodyText2"]))

    # ---- EXPERIENCE ----
    if resume.experience:
        story.append(Paragraph("EXPERIENCE", styles["SectionTitle"]))
        for exp in resume.experience:
            title = exp.get("title", "")
            company = exp.get("company", "")
            duration = exp.get("duration", "")
            desc = exp.get("description", "")

            header = f"<b>{title}</b>"
            if company:
                header += f" — {company}"
            if duration:
                header += f"  <font color='#7a7a9a' size='9'>({duration})</font>"
            story.append(Paragraph(header, styles["ExpTitle"]))
            if desc:
                story.append(Paragraph(f"• {desc}", styles["BulletText"]))
            story.append(Spacer(1, 2*mm))

    # ---- EDUCATION ----
    if resume.education:
        story.append(Paragraph("EDUCATION", styles["SectionTitle"]))
        for edu in resume.education:
            degree = edu.get("degree", "")
            institution = edu.get("institution", user_university)
            year = edu.get("year", "")

            line = f"<b>{degree}</b>" if degree else ""
            if institution:
                line += f" — {institution}" if line else f"<b>{institution}</b>"
            if year:
                line += f"  <font color='#7a7a9a' size='9'>({year})</font>"
            if line:
                story.append(Paragraph(line, styles["ExpTitle"]))
                story.append(Spacer(1, 2*mm))

    # ---- PROJECTS ----
    if resume.projects:
        story.append(Paragraph("PROJECTS", styles["SectionTitle"]))
        for proj in resume.projects:
            name = proj.get("name", "")
            desc = proj.get("description", "")
            techs = proj.get("technologies", [])

            header = f"<b>{name}</b>"
            if techs:
                header += f"  <font color='#7a7a9a' size='9'>({', '.join(techs)})</font>"
            story.append(Paragraph(header, styles["ExpTitle"]))
            if desc:
                story.append(Paragraph(f"• {desc}", styles["BulletText"]))
            story.append(Spacer(1, 2*mm))

    # ---- ACHIEVEMENTS ----
    if resume.achievements:
        story.append(Paragraph("ACHIEVEMENTS", styles["SectionTitle"]))
        for ach in resume.achievements:
            story.append(Paragraph(f"• {ach}", styles["BulletText"]))

    # ---- FOOTER ----
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=0.3, color=DIVIDER))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Generated by AI Video Resume Generator — this resume was created from a video self-introduction. "
        "Skills and experiences listed are extracted from the student's own statements.",
        styles["SmallText"],
    ))

    doc.build(story)
    return pdf_path
