# 🎯 JobFit AI - Resume Optimization & Interview Preparation Tool

## 🚀 Executive Summary (TL;DR)

A **smart AI-powered career assistant** that analyzes resume-job description compatibility and provides **personalized optimization** and **interview preparation guidance**.

- **Objective:** Solve the challenge of job application success by combining **NLP analysis** with **generative AI** for career enhancement.
- **Solution:** Built a **multi-format resume parser** (PDF, DOCX, PPTX, TXT) with **skill matching algorithms** and integrated **DeepSeek LLM** for resume optimization and interview coaching.
- **Key Innovation:** Not just analyzing compatibility, but providing **actionable improvements**—a unique blend of technical analysis and career strategy.
- **Technical Highlight:** Deployed as an **interactive web app (Streamlit)** with dark theme UI that processes documents in **real-time**.
- **My Role:** End-to-end development: document parsing, NLP implementation, fuzzy matching algorithms, LLM integration, and professional UI/UX design.

**👉 Scroll down for the detailed methodology, features, and setup instructions.**

---

## 📌 Project Overview

Welcome to **JobFit AI**, an intelligent career assistant that bridges the gap between your resume and job requirements. This tool combines **natural language processing (NLP)** with **large language models (LLMs)** to provide comprehensive career guidance—from resume optimization to interview preparation.

This project showcases a complete **document processing pipeline**—from **multi-format parsing**, **text preprocessing**, **skill extraction**, **fuzzy matching**, to **AI-powered content generation** via a sleek Streamlit interface.

By leveraging **Python**, **SpaCy**, **NLTK**, **fuzzywuzzy**, and **DeepSeek (via OpenRouter)**, I built a robust system that not only analyzes job-resume compatibility but also generates actionable improvements and interview strategies.

---

## 🛠️ Technical Stack

### Core Technologies:
- **Python 3.11+**
- **Streamlit** - Web application framework
- **SpaCy** - NLP processing and entity recognition
- **NLTK** - Text preprocessing and stopwords
- **fuzzywuzzy** - Fuzzy string matching for skill comparison
- **pdfplumber** - PDF document parsing
- **python-docx** - Word document processing
- **python-pptx** - PowerPoint presentation parsing

### AI & APIs:
- **DeepSeek R1** (via OpenRouter) - LLM for content generation
- **OpenRouter API** - AI model integration

### File Format Support:
- **Resume Input:** PDF, DOCX, PPTX, TXT
- **Job Description:** TXT (with text input support)

---

## 🧠 Methodology

### 1️⃣ Multi-Format Document Parsing
- **PDF:** Uses `pdfplumber` for robust text extraction
- **DOCX:** Leverages `python-docx` for Word document processing
- **PPTX:** Implements `python-pptx` for PowerPoint text extraction
- **TXT:** Direct text processing with encoding support

### 2️⃣ Text Preprocessing & NLP
- **Confidentiality Filtering:** Removes confidential clauses and repeated patterns
- **Entity Recognition:** Uses SpaCy to filter out personal names and common resume terms
- **Token Processing:** Lemmatization, stopword removal, and punctuation filtering
- **Noun Chunk Extraction:** Identifies key phrases and technical terms

### 3️⃣ Skill Matching Algorithm
- **Predefined Skill Library:** 50+ common technical and soft skills
- **Fuzzy Matching:** Uses `fuzz.partial_ratio` (75+ threshold) for flexible skill matching
- **Experience Extraction:** Regex patterns to identify years of experience
- **Eligibility Check:** Compares resume experience with job requirements

### 4️⃣ AI-Powered Content Generation
- **Resume Optimization:** Rewrites resume to emphasize matching skills and address gaps
- **Interview Guidance:** Generates tailored questions and preparation strategies
- **Prompt Engineering:** Custom prompts for professional tone and conciseness

### 5️⃣ Match Scoring System
- **Skillset Match Score:** Percentage-based calculation of skill overlap
- **Eligibility Gate:** Experience-based eligibility affects final score
- **Three-Way Analysis:** Matching skills, missing skills, and extra skills

---

## 🎯 Key Features

### 🔍 Smart Analysis
- **Multi-format resume parsing**
- **Experience-based eligibility checking**
- **Skill matching with fuzzy logic**
- **Comprehensive match scoring**

### ✨ AI Optimization
- **Resume rewriting** for better job alignment
- **Missing skill addressing** strategies
- **Professional tone maintenance**
- **Concise output generation**

### 🎤 Interview Preparation
- **Tailored interview questions**
- **Skill-specific guidance**
- **Gap-bridging strategies**
- **Professional answer frameworks**

### 🎨 Professional UI/UX
- **Dark theme with orange accents**
- **Responsive card-based layout**
- **Real-time processing indicators**
- **Organized tabbed interface**

---

## 📊 Results & Metrics

### Analysis Capabilities:
- **Document Processing:** 4+ file formats supported
- **Skill Matching:** 50+ predefined skills with fuzzy matching
- **Experience Detection:** Accurate year extraction from text
- **Match Scoring:** Percentage-based compatibility assessment

### AI Generation:
- **Resume Optimization:** Professional rewriting in under 500 words
- **Interview Guidance:** 3-5 tailored questions with strategies
- **Response Quality:** Context-aware, job-specific recommendations

### User Experience:
- **Processing Time:** Real-time analysis (< 5 seconds)
- **Interface:** Intuitive, professional dark theme
- **Output:** Actionable, easy-to-understand recommendations

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Hrushikesh-katti/projects-portfolio.git
cd projects-portfolio/jobfit-ai
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
(# source venv/bin/activate  # Linux/Mac)
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt`
```

### 4. Download NLP Models
```bash
python -m spacy download en_core_web_sm
```

### 5. Run the Application
```bash
streamlit run JobFit_ai_app.py
```

### 6. Access the App
Open your browser and navigate to: http://localhost:8501

---

## 📁 Project Structure

```text
jobfit-ai/
├── JobFit_predictor_app.py   # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── assets/                   # Images and static files
│   ├── app-screenshot.png
│   └── sample_jd.txt
└── samples/                # Sample resumes and job descriptions
    ├── sample_resume.pdf
    └── sample_jd.txt
```

---

## 🧾 Requirements

```txt
streamlit>=1.28.0
spacy>=3.7.0
nltk>=3.8.0
pdfplumber>=0.10.0
python-docx>=1.1.0
python-pptx>=0.6.23
fuzzywuzzy>=0.18.0
openai>=1.3.0
```

---

## 🏆 Key Achievements

- ✅ **End-to-end document processing pipeline**
- ✅ **Intelligent skill matching** with fuzzy logic
- ✅ **AI-powered content generation** for career enhancement
- ✅ **Professional web interface** with dark theme
- ✅ **Multi-format support** for real-world usability
- ✅ **Real-time processing** with user-friendly feedback

---

## 🔮 Future Enhancements

- **Enhanced Skill Library:** Expand to 200+ skills with categories
- **ATS Compatibility:** Check resume compatibility with Applicant Tracking Systems
- **Multi-language Support:** Extend beyond English
- **Performance Analytics:** Track application success rates
- **Cover Letter Generation:** AI-powered cover letter creation
- **Salary Insights:** Market-appropriate salary recommendations
- **Company Research Integration:** Auto-pull company information

---

## 👨‍💻 About Me

**Hrushikesh S Katti**  
**Data Science Specialist | AI Certified Professional | Machine Learning Engineer | NLP Expert | Business Intelligence Analyst | Full-Stack Developer | Career Strategy Consultant**  
🔍 Open to Work | Actively Looking for Full-Time Opportunities  
📍 Open to Relocation: Bangalore | Hyderabad | Pune | Chennai | Remote  

- GitHub: [Hrushikesh-katti](https://github.com/Hrushikesh-katti)  
- Email: hrushikeshskatti7@gmail.com  
- Portfolio: [Projects Portfolio](https://github.com/Hrushikesh-katti/projects-portfolio)

---

> ✨ *JobFit AI represents the intersection of technical innovation and practical career solutions. This tool demonstrates how AI can be harnessed to create meaningful impact in job search and career development.* 🚀  
> *From document parsing to AI generation - proving that complex problems can be solved with elegant, user-focused solutions.* 💡

---

**Note:** Replace the OpenRouter API key in the application with your own from [openrouter.ai](https://openrouter.ai) for full functionality.