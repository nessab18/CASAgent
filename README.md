# CASAgent 
![CASAgent Logo](casagent_logo_github.png)
### AI-Powered Support Tool for CASA Volunteers

> *Augmenting the work of Court Appointed Special Advocates through intelligent, session-safe AI assistance.*
[![Live Demo](https://img.shields.io/badge/Live%20Demo-casagent.org-blue)](https://casagent.org)
[![Built with Claude](https://img.shields.io/badge/Built%20with-Anthropic%20Claude-orange)](https://www.anthropic.com)
[![Python](https://img.shields.io/badge/Python-LangChain%20%7C%20LangGraph-green)](https://github.com/nessab18/CASAgent/blob/main/casagent_assignment5.py)

🌐 **Live App:** https://casagent.org
💻 **Code Demo:** [casagent_assignment5.py](https://github.com/nessab18/CASAgent/blob/main/casagent_assignment5.py)
---
 
## Problem Statement
 
There are approximately **85,000 CASA (Court Appointed Special Advocate) volunteers** in the United States. These volunteers — many of them law students and professionals without prior child welfare experience — are appointed by judges to advocate for children in the foster care system. They prepare court reports, monitor placements, track developmental milestones, and ensure children receive appropriate services.
 
Despite their critical role, CASA volunteers face significant challenges:
 
- **Information overload**: Managing case files, court dates, service provider contacts, and prior history across multiple children simultaneously
- **Documentation burden**: Drafting court reports that must be accurate, professional, and preserve the advocate's firsthand voice
- **Knowledge gaps**: Identifying missing services, relevant legal precedents, and developmental red flags without clinical training
- **Time constraints**: Volunteers balance advocacy with full-time jobs and personal lives
 
Existing enterprise legal AI tools like Squary AI are built for paid attorneys at mid-to-large law firms — not for volunteer advocates working in the public interest. **CASAgent democratizes legal AI for the people who need it most.**
 
---

## Project Scope
 
CASAgent is a narrowly scoped AI tool designed specifically for CASA volunteers managing child welfare cases in Virginia. The prototype focuses on four core functions:
 
1. **Court Preparation** — Summarize case files, surface relevant Virginia statutes, and flag urgent concerns before hearings
2. **Milestone Tracking** — Analyze volunteer observations for developmental concerns and suggest follow-up questions
3. **Report Drafting** — Generate first-person court report drafts that preserve the advocate's voice
4. **Gap Analysis** — Identify missing services, evaluations, or documentation that could strengthen a case
 
The tool is session-only by design: no case data is stored, transmitted, or persisted beyond the browser session, ensuring confidentiality of sensitive child welfare information.
 
---

## Authors
 
| Name | GitHub | Role |
|------|--------|------|
| Vanessa Broadrup | [@nessab18](https://github.com/nessab18) | Team Member |
| Perry Irons | [@perryirons](https://github.com/perryirons) | Team Member |
| Redeemer Gawu | [@rsgawu](https://github.com/rsgawu) | Team Member |
| Javier Cruz | [@javiercruz090](https://github.com/javiercruz090) | Team Member |
 
**Client:** Ida Guerami
 
---
## Project Details
 
### Architecture Overview
 
CASAgent uses a **multi-agent AI architecture** built on LangChain and LangGraph, with a front-end HTML/CSS/JS web application powered by the Anthropic Claude API.
 
```
┌─────────────────────────────────────────────────────┐
│                  ORCHESTRATION AGENT                 │
│         Coordinates sub-agents & synthesizes        │
│                    final output                     │
└──────────────┬──────────────────────┬───────────────┘
               │                      │
               ▼                      ▼
┌──────────────────────┐  ┌──────────────────────────┐
│  COURT PREP AGENT    │  │   CASE ANALYSIS AGENT    │
│                      │  │                          │
│ Tools:               │  │ Tools:                   │
│ • summarize_case_    │  │ • flag_developmental_    │
│   flags              │  │   concern                │
│ • web_search         │  │ • identify_service_gaps  │
│   (Tavily API)       │  │ • draft_court_report_    │
│                      │  │   section                │
└──────────────────────┘  └──────────────────────────┘
```
### Web Application Features
 
The hosted web application ([casagent.org](https://casagent.org)) includes:
 
**Child Profile System**
- Longitudinal child profiles with case history across multiple court matters
- Key considerations flags (IEP compliance, placement disruptions, visit patterns)
- Languages spoken field for multilingual households
- Session-only data — nothing is stored or transmitted
 
**Case Workspace (4 AI Modules)**
- **Court Prep** — paste case notes, receive structured hearing preparation summary with legal context sourced via live web search
- **Milestone Tracker** — describe observations, receive developmental concern flags with suggested questions for service providers
- **Report Drafter** — input observations, receive first-person court report draft in the advocate's voice
- **Gap Analyzer** — describe current services, receive prioritized list of missing evaluations and documentation
 
**Conversational AI Assistant**
- Voice input via Web Speech API (no additional cost or API required)
- Text-to-speech output — volunteers can listen to responses hands-free
- Context-aware: the assistant automatically knows which child and case is active
- Multi-turn memory within a session
 
**Resource Directory**
- Clickable agency cards for local organizations (James City County CPS, Colonial Behavioral Health, Avalon Center, etc.)
- Contact info, hours, address, and website links
- Org names in AI output become clickable links automatically
 
**Case Management**
- Hearing tracker with judge, courtroom, date countdown, and prep checklist
- Visit log with date, type, volunteer attribution, and notes
- Interview transcript storage (session-only, with authorization disclaimer)
- Search by child initials or case number
 
### LangChain Multi-Agent System
 
The Python implementation (`casagent_assignment5.py`) demonstrates a full multi-agent agentic system:
 
| Component | Implementation |
|-----------|---------------|
| LLM Initialization | `init_chat_model("anthropic:claude-sonnet-4-20250514")` at temperature 0 and 0.7 |
| Agent Creation | Two distinct agents via `create_agent()` with tailored system prompts |
| Message Handling | `HumanMessage` + `AIMessage` multi-turn conversations |
| Streaming Output | `agent.stream()` with `stream_mode="messages"` |
| Custom Tools | 4 tools: `summarize_case_flags`, `identify_service_gaps`, `flag_developmental_concern`, `draft_court_report_section` |
| External API Tool | Tavily web search for live Virginia statute and case law lookup |
| Agent Memory | `InMemorySaver` checkpointer with per-case `thread_id` |
| Multi-Agent Orchestration | Orchestrator wraps sub-agents as callable tools, synthesizes unified pre-hearing brief |
 
---
## What's Next?
 
**Near-term development:**
- **Email integration** — automatically pull relevant case correspondence into the volunteer's workspace, eliminating inbox digging
- **Case law citations with source links** — surface clickable legal citations so volunteers can trace every AI-generated legal reference back to the original source
- **User authentication** — CASA-administered login system so the tool can be deployed internally without API key exposure
- **Backend server** — add server-side API key handling and user authentication for production deployment
 
**Longer-term vision:**
- **Multilingual interface** — serve CASA chapters in communities where children and families speak languages other than English
- **Child-facing resources** — age-appropriate materials in multiple languages explaining what a CASA volunteer does and offering additional support
- **Mobile-responsive design** — volunteers are frequently on the go; a mobile-optimized experience would support on-site visit note-taking
- **Volunteer handoff system** — when a new volunteer is assigned to a child, the profile and history provide immediate context, reducing onboarding friction
 
**Open questions and concerns:**
- What are CASA's data governance policies around digital storage of interview transcripts?
- How should volunteer authentication and access control be structured to protect child data?
- As the tool is adopted more broadly, how do we ensure it does not inadvertently introduce bias into advocacy recommendations?
 
---
 
## Responsible AI Considerations
 
**Confidentiality by design**
CASAgent is session-only. No case data, child information, or volunteer notes are stored, logged, or transmitted beyond the immediate API call. Every child is identified by initials only. A confidentiality banner is displayed persistently across all views.
 
**Human-in-the-loop**
Every AI output includes an explicit disclaimer: *"CASAgent augments your judgment — it does not replace it. Always review all outputs carefully before use in any official context."* The tool is designed to support advocates, not substitute for their direct observations and relationships with children.
 
**No diagnosis**
The milestone tracker tool is explicitly instructed not to diagnose. All developmental observations are framed as prompts for professional investigation, not clinical conclusions. Volunteers are directed to raise concerns with qualified service providers.
 
**No legal advice**
The court prep agent surfaces legal information for reference only. Volunteers are directed to consult supervising CASA attorneys for formal legal guidance.
 
**Bias awareness**
AI models can reflect biases present in training data. Recommendations about service gaps, developmental concerns, or legal strategy should always be evaluated by the advocate in the context of the specific child and family — never applied mechanically.
 
**Data minimization**
The tool is designed to need as little identifying information as possible. Child initials rather than names, case numbers rather than full court records.
 
---
## 📚 References
 
**Primary Research Papers:**
 
Kawakami, A., Sivaraman, V., Cheng, H., Stapleton, L., Cheng, Y., Qing, D., Perer, A., Wu, Z. S., Zhu, H., & Holstein, K. (2022). *Improving Human-AI Partnerships in Child Welfare: Understanding Worker Practices, Challenges, and Desires for Algorithmic Decision Support.* Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems. **Best Paper Honorable Mention.** https://arxiv.org/abs/2204.02310

Carroll, J. M. et al. (2025). From Sociotechnical Gaps to Solutions: Designing AI Tools with Parents to Address Special Education Advocacy Barriers in IEP Processes. Proceedings of the 2025 ACM Designing Interactive Systems Conference. https://doi.org/10.1145/3715336.3735778
 
**Additional Resources:**
 
- [National CASA/GAL Association](https://nationalcasagal.org/) — overview of CASA volunteer program and scale
- [Virginia CASA Association](https://virginiacasa.org/) — Virginia-specific CASA guidelines and volunteer requirements
- [Anthropic Claude API Documentation](https://docs.anthropic.com) — LLM powering CASAgent
- [LangChain Documentation](https://python.langchain.com) — multi-agent framework
- [Squary AI Legal](https://www.squary.ai/) — comparable commercial tool for law firms
- [Child Welfare Information Gateway](https://www.childwelfare.gov/) — federal resource for child welfare policy and practice
 
---
 
## 🔒 Privacy Notice
 
All information visible in the live demo is **simulated and fictional**. No real child, case, or personally identifying information has been used at any stage of this project. CASAgent is a prototype intended for demonstration purposes. Any deployment with real case data would require review and approval by CASA's data governance team.
 
---
 
*Built for the William & Mary Raymond A. Mason School of Business MSBA AI Course, Spring 2026.*
