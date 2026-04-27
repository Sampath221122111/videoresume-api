"""
Resume Generation Service — uses Groq Llama3 (free tier).
ATS-OPTIMIZED: Creates resumes that pass Applicant Tracking Systems 
and are relevant to the student's field and target companies.
"""

import json
import re
from groq import Groq
from app.config import get_settings
from app.models.schemas import ResumeData, TranscriptionResult, AudioAnalysis, FaceAnalysis


SYSTEM_PROMPT = """You are a senior career counselor and ATS resume expert who has helped 10,000+ students land jobs at top companies (TCS, Infosys, Wipro, Google, Amazon, Microsoft, Accenture, Cognizant, HCL, etc.).

Your job: Create an ATS-OPTIMIZED, COMPANY-READY resume from a student's video transcript.

## ATS OPTIMIZATION RULES:

1. USE ACTION VERBS to start every bullet point: Developed, Implemented, Designed, Built, Analyzed, Led, Created, Optimized, Managed, Automated, Integrated, Deployed, etc.

2. QUANTIFY achievements wherever possible: "Improved performance by 30%", "Managed team of 5", "Built system handling 1000+ records", etc. If no numbers are given, add reasonable estimates.

3. INCLUDE ATS KEYWORDS based on their branch that recruiters and ATS systems scan for.

4. CREATE PROPER SECTIONS:
   - Professional Summary (3-4 lines, keyword-rich, mentions career goal)
   - Technical Skills (15-25 skills: Languages, Frameworks, Tools, Databases, Soft Skills)
   - Education (with relevant coursework, CGPA if mentioned)
   - Projects (detailed with tech stack, problem, solution, impact — minimum 2)
   - Experience/Internships (if mentioned, with action-verb bullet points)
   - Achievements (academic + extracurricular + certifications)
   - Languages spoken

5. FOR PROJECTS: Every project MUST have:
   - Clear problem statement
   - Technologies used
   - Specific contribution with action verbs
   - Quantified impact/result

6. MAKE IT RELEVANT to companies that hire students from their branch.

7. OUTPUT FORMAT — Return ONLY valid JSON:

{
  "summary": "3-4 line ATS-optimized professional summary. Must mention: name, degree, university, top 3 skills, career goal.",
  "skills": ["Skill1", "Skill2", "...15-25 skills"],
  "experience": [
    {
      "title": "Professional role title",
      "company": "Company/Organization",
      "duration": "Month Year - Month Year",
      "description": "Action-verb bullet points with quantified results"
    }
  ],
  "education": [
    {
      "degree": "B.Tech in Branch Name",
      "institution": "Full University Name",
      "year": "Start - End (Expected)",
      "details": "Relevant Coursework: Subject1, Subject2, ... | CGPA: X.X (if mentioned)"
    }
  ],
  "projects": [
    {
      "name": "Professional project name",
      "description": "Action-verb description with problem-solution-impact format",
      "technologies": ["Tech1", "Tech2", "Tech3"]
    }
  ],
  "achievements": ["Quantified achievement 1", "Achievement 2"],
  "interests": ["Relevant interest 1", "Interest 2"],
  "languages": ["English (Professional)", "Hindi (Native)"]
}

## CRITICAL:
- NEVER leave sections empty. Infer from branch if needed.
- 15-25 skills minimum.
- At least 2 projects.
- Use professional language — make the student sound COMPETENT and HIRE-READY.
- Output ONLY valid JSON."""


def generate_resume(
    transcript: TranscriptionResult,
    audio_analysis: AudioAnalysis,
    face_analysis: FaceAnalysis,
    user_name: str = "Student",
    user_university: str = "",
    user_branch: str = "",
) -> ResumeData:
    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key)

    target_companies = _get_target_companies(user_branch)
    branch_keywords = _get_ats_keywords(user_branch)

    user_prompt = f"""Create an ATS-OPTIMIZED professional resume for this student.

## STUDENT PROFILE:
- Full Name: {user_name}
- University: {user_university}
- Branch/Major: {user_branch}
- Target Companies: {', '.join(target_companies)}

## VIDEO TRANSCRIPT:
\"\"\"{transcript.full_text}\"\"\"

## VIDEO SCORES:
- Confidence: {audio_analysis.confidence_score:.1f}% {'(Strong communicator — mention this)' if audio_analysis.confidence_score > 60 else ''}
- Speaking Rate: {audio_analysis.speaking_rate_wpm:.0f} WPM
- Eye Contact: {face_analysis.avg_eye_contact_score:.1f}%
- Expression: {face_analysis.avg_expression_score:.1f}%

## ATS KEYWORDS TO INCLUDE: {', '.join(branch_keywords[:15])}

## REQUIREMENTS:
1. Extract EVERYTHING from transcript — skills, projects, achievements, internships
2. EXPAND brief mentions into professional descriptions with action verbs and quantified results
3. Add branch-relevant skills that ATS systems scan for
4. Include coursework for {user_branch}
5. Generate 15-25 skills, 2-4 projects, proper education section
6. Make competitive for: {', '.join(target_companies[:5])}
7. Output ONLY valid JSON"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4000,
            top_p=0.95,
        )

        raw_text = response.choices[0].message.content.strip()
        print(f"[RESUME] Raw response length: {len(raw_text)} chars")

        raw_text = re.sub(r"^```json\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
        raw_text = re.sub(r"^```\s*", "", raw_text)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', raw_text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                print(f"[RESUME] JSON parse failed, using fallback")
                return _create_fallback_resume(user_name, user_university, user_branch, transcript)

        resume = ResumeData(
            summary=data.get("summary", f"{user_name} is a student at {user_university}."),
            skills=data.get("skills", []),
            experience=data.get("experience", []),
            education=data.get("education", []),
            projects=data.get("projects", []),
            achievements=data.get("achievements", []),
        )

        # ── POST-PROCESSING ──
        if not resume.education:
            resume.education = [{"degree": f"B.Tech in {user_branch}" if user_branch else "Bachelor's Degree", "institution": user_university or "University", "year": "Present", "details": _get_coursework(user_branch)}]

        if len(resume.skills) < 15 and user_branch:
            for skill in branch_keywords:
                if skill not in resume.skills and len(resume.skills) < 22:
                    resume.skills.append(skill)

        if len(resume.projects) < 2 and user_branch:
            for proj in _get_default_projects(user_branch):
                if len(resume.projects) < 2:
                    resume.projects.append(proj)

        if audio_analysis.confidence_score > 60:
            for s in ["Communication Skills", "Public Speaking"]:
                if s not in resume.skills: resume.skills.append(s)
        if face_analysis.avg_eye_contact_score > 60 and "Presentation Skills" not in resume.skills:
            resume.skills.append("Presentation Skills")

        print(f"[RESUME] Generated: {len(resume.skills)} skills, {len(resume.experience)} exp, {len(resume.projects)} projects")
        return resume

    except Exception as e:
        print(f"[RESUME] Error: {e}")
        return _create_fallback_resume(user_name, user_university, user_branch, transcript)


def _get_target_companies(branch):
    b = (branch or "").lower()
    if any(x in b for x in ["computer", "cse", "software", "it", "information"]):
        return ["TCS", "Infosys", "Wipro", "Google", "Amazon", "Microsoft", "Accenture", "Cognizant", "HCL"]
    elif any(x in b for x in ["electronic", "ece", "electrical", "eee"]):
        return ["Texas Instruments", "Qualcomm", "Intel", "Samsung", "Bosch", "Siemens", "TCS", "Infosys"]
    elif any(x in b for x in ["mechani", "mech"]):
        return ["Tata Motors", "Mahindra", "Bosch", "L&T", "Ashok Leyland", "Maruti Suzuki", "Caterpillar"]
    elif any(x in b for x in ["civil"]):
        return ["L&T", "Shapoorji Pallonji", "DLF", "NHAI", "Tata Projects", "Afcons"]
    return ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "HCL"]


def _get_ats_keywords(branch):
    b = (branch or "").lower()
    if any(x in b for x in ["computer", "cse", "software", "it", "information"]):
        return ["Python", "Java", "C++", "JavaScript", "SQL", "HTML/CSS", "React", "Node.js", "Git", "GitHub", "Data Structures", "Algorithms", "Machine Learning", "REST APIs", "MySQL", "MongoDB", "Docker", "AWS", "Agile", "OOP", "Problem Solving", "Critical Thinking", "Team Collaboration", "Communication Skills"]
    elif any(x in b for x in ["electronic", "ece", "electrical", "eee"]):
        return ["Embedded Systems", "VLSI Design", "IoT", "Signal Processing", "MATLAB", "Arduino", "Raspberry Pi", "PCB Design", "Verilog", "VHDL", "Microcontrollers", "C Programming", "Python", "Digital Electronics", "Circuit Design", "Problem Solving"]
    elif any(x in b for x in ["mechani", "mech"]):
        return ["AutoCAD", "SolidWorks", "CATIA", "ANSYS", "CNC Programming", "3D Printing", "Thermodynamics", "Fluid Mechanics", "Manufacturing", "Six Sigma", "Lean Manufacturing", "FEA", "GD&T", "Problem Solving", "Project Management"]
    elif any(x in b for x in ["civil"]):
        return ["AutoCAD", "STAAD Pro", "ETABS", "Revit", "Primavera", "Surveying", "Structural Analysis", "Concrete Technology", "Soil Mechanics", "Construction Management", "BIM", "MS Project", "Problem Solving"]
    return ["Problem Solving", "Communication", "Teamwork", "Analytical Thinking", "Project Management", "Microsoft Office", "Presentation Skills", "Critical Thinking", "Leadership", "Time Management"]


def _get_coursework(branch):
    b = (branch or "").lower()
    if any(x in b for x in ["computer", "cse", "software", "it"]):
        return "Relevant Coursework: Data Structures & Algorithms, DBMS, Operating Systems, Computer Networks, Software Engineering, Machine Learning, Web Technologies, OOP"
    elif any(x in b for x in ["electronic", "ece", "electrical"]):
        return "Relevant Coursework: Digital Electronics, Signal Processing, Embedded Systems, VLSI Design, Communication Systems, Control Systems, Microprocessors"
    elif any(x in b for x in ["mechani", "mech"]):
        return "Relevant Coursework: Thermodynamics, Fluid Mechanics, Machine Design, Manufacturing Processes, Strength of Materials, Kinematics, Heat Transfer"
    elif any(x in b for x in ["civil"]):
        return "Relevant Coursework: Structural Analysis, Surveying, Concrete Technology, Soil Mechanics, Transportation Engineering, Construction Management"
    return "Relevant Coursework: Engineering Mathematics, Physics, Professional Communication"


def _get_default_projects(branch):
    b = (branch or "").lower()
    if any(x in b for x in ["computer", "cse", "software", "it"]):
        return [
            {"name": "Student Management System", "description": "Developed a full-stack web application for managing student records using Python and MySQL. Implemented CRUD operations, authentication, and role-based access control. Deployed handling 500+ student records with responsive UI.", "technologies": ["Python", "MySQL", "HTML/CSS", "JavaScript"]},
            {"name": "Weather Prediction Application", "description": "Built a machine learning model to predict weather patterns using historical data. Achieved 85% accuracy using Random Forest and data preprocessing. Created interactive dashboard for visualization.", "technologies": ["Python", "Scikit-learn", "Pandas", "Matplotlib"]},
        ]
    elif any(x in b for x in ["electronic", "ece", "electrical"]):
        return [
            {"name": "IoT-based Home Automation System", "description": "Designed and implemented a smart home system using Arduino and ESP8266. Integrated temperature, humidity, and motion sensors for automated control. Developed mobile app for remote monitoring.", "technologies": ["Arduino", "ESP8266", "IoT", "C"]},
        ]
    elif any(x in b for x in ["mechani", "mech"]):
        return [
            {"name": "3D Printed Robotic Arm", "description": "Designed a 4-DOF robotic arm using SolidWorks with 3D printed components. Programmed servo motors using Arduino for pick-and-place operations. Achieved positioning accuracy of plus/minus 2mm.", "technologies": ["SolidWorks", "Arduino", "3D Printing", "C++"]},
        ]
    return []


def _create_fallback_resume(user_name, user_university, user_branch, transcript):
    skills = _get_ats_keywords(user_branch)
    return ResumeData(
        summary=f"Motivated and detail-oriented {user_branch} student at {user_university} with strong foundation in technical concepts and practical project experience. Demonstrated excellent communication skills and ability to work in collaborative team environments. Seeking entry-level opportunities to apply academic knowledge and contribute to innovative solutions.",
        skills=skills[:20],
        experience=[],
        education=[{"degree": f"B.Tech in {user_branch}" if user_branch else "Bachelor's Degree", "institution": user_university or "University", "year": "Present", "details": _get_coursework(user_branch)}],
        projects=_get_default_projects(user_branch)[:2],
        achievements=["Active participant in college technical festivals and coding competitions", "Completed online certification courses in relevant technologies"],
    )
