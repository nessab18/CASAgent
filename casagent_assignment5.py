# -*- coding: utf-8 -*-
"""
CASAgent Multi-Agent System — Assignment 5
MSBA AI Course | Team Project

Domain: CASA (Court Appointed Special Advocate) Volunteer Support Tool

CASAgent is an AI-powered tool that helps CASA volunteers:
  - Prepare for court hearings by summarizing case files
  - Track child developmental milestones and flag concerns
  - Draft court reports while preserving the volunteer's voice
  - Identify service gaps that could strengthen a case

Based on: Chapter 10, AI for Business by Chung & Hojnicki
LangChain Colab: https://colab.research.google.com/drive/17gA1wr_0SNgjxFby8r5BIbeuziIbZKt8

SETUP — install dependencies before running:
  pip install langchain langchain-anthropic langgraph tavily
"""

# ============================================================
# IMPORTS SECTION
# ============================================================
import os
from pprint import pprint
from typing import Dict, Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient


# ============================================================
# API KEY CONFIGURATION
# Set your keys directly here, or load from environment.
# In Colab, use: userdata.get('ANTHROPIC_API_KEY')
# Note: We are not uploading this with our API keys here, we will remove them before submission.
# ============================================================

os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key-here"
TAVILY_API_KEY = "your-tavily-api-key-here"




# ============================================================
# SECTION 1: LLM INITIALIZATION
#
# Use init_chat_model() to initialize at least one LLM.
# Demonstrate invoking the model directly (without an agent)
# and experiment with the temperature parameter.
# ============================================================
print("\n" + "="*60)
print("SECTION 1: LLM INITIALIZATION")
print("="*60)

# Initialize Claude claude-sonnet-4-20250514 via init_chat_model()
# temperature=0   → precise, consistent (ideal for court prep & gap analysis)
# temperature=0.7 → more expressive, varied (ideal for report drafting)
llm = init_chat_model(
    model="anthropic:claude-sonnet-4-20250514",
    temperature=0
)

# Demo prompt for direct invocation
prompt0 = "What should a CASA volunteer focus on when preparing for a child's permanency hearing?"

# Invoke the model DIRECTLY — no agent
print("\n--- Direct model invocation (temperature=0, no agent) ---")
response = llm.invoke(prompt0)
pprint(response)

# Print just the answer
print("\nContent only:")
print(response.content)

# Print metadata
print("\nResponse metadata:")
pprint(response.response_metadata)

# Experiment with higher temperature for more creative output
print("\n--- Direct model invocation (temperature=0.7) ---")
response_creative = llm.invoke(prompt0, temperature=0.7)
print(response_creative.content)

print("\n[Note] temperature=0 → factual/consistent | temperature=0.7 → expressive/varied")


# ============================================================
# SECTION 2: AGENT CREATION
#
# Use create_agent() to build at least two distinct agents,
# each with its own system prompt tailored to a specific role.
# ============================================================

# (Agents are created after tools are defined below — see Sections 5 & 6)


# ============================================================
# SECTION 5: CUSTOM TOOLS
#
# Create a minimum of two custom tools using the @tool decorator.
# Each tool includes a clear docstring, typed parameters, and return values.
# Defined before agents so agents can be initialized with them.
# ============================================================
print("\n" + "="*60)
print("SECTION 5: CUSTOM TOOLS")
print("="*60)

# --- Custom Tool 1: Summarize Case Flags ---
@tool
def summarize_case_flags(case_notes: str) -> str:
    """
    Analyzes raw CASA case notes and extracts key flags for a volunteer.

    Parses the provided case notes and returns a structured summary
    of urgent concerns, upcoming deadlines, and placement details
    that a CASA volunteer should know before a court hearing.

    Args:
        case_notes: A string containing raw case notes, prior reports,
                    or relevant documentation for a child welfare case.

    Returns:
        A structured string summarizing: (1) urgent flags, (2) upcoming
        dates, (3) placement status, (4) recommended follow-up actions.
    """
    prompt = f"""You are a CASA case analyst. Review these case notes and extract:

1. URGENT FLAGS (safety, legal, or wellbeing concerns)
2. UPCOMING DATES (hearings, deadlines, service reviews)
3. PLACEMENT STATUS (current placement and stability)
4. RECOMMENDED ACTIONS (what the volunteer should do next)

Case notes:
{case_notes}

Format clearly under each numbered heading."""
    result = llm.invoke(prompt)
    return result.content


# --- Custom Tool 2: Identify Service Gaps ---
@tool
def identify_service_gaps(services_description: str, child_age: int) -> str:
    """
    Identifies missing or insufficient services for a child in CASA care.

    Compares services currently in place against what is typically
    required for a child of the given age in the Virginia child welfare
    system, and returns a prioritized list of gaps.

    Args:
        services_description: A string describing services, placements,
                               evaluations, and supports currently in place.
        child_age: The age of the child in years, used to calibrate
                   developmentally appropriate service expectations.

    Returns:
        A prioritized list of service gaps with recommended next steps
        for the CASA volunteer to advocate for.
    """
    prompt = f"""You are a child welfare services expert advising a CASA volunteer.
A {child_age}-year-old child has the following services in place:

{services_description}

Identify:
1. MISSING EVALUATIONS typically required at age {child_age}
2. ABSENT SERVICES given the child's situation
3. DOCUMENTATION GAPS that could weaken the case
4. PRIORITY NEXT STEPS for the advocate (ranked by urgency)

Be specific and actionable. Frame gaps as opportunities to strengthen support."""
    result = llm.invoke(prompt)
    return result.content


# --- Custom Tool 3: Flag Developmental Concern ---
@tool
def flag_developmental_concern(observation: str, child_age: int, domain: str) -> str:
    """
    Evaluates a volunteer observation and flags potential developmental concerns.

    Assesses whether an observation represents a developmental concern
    relative to typical expectations for the child's age.

    Args:
        observation: A specific observation about the child's behavior,
                     development, school performance, or emotional state.
        child_age: The child's age in years.
        domain: The developmental domain to assess. Valid values:
                'educational', 'emotional', 'physical', 'social', 'behavioral'.

    Returns:
        A structured concern flag with: severity level (none/low/medium/high),
        clinical context, suggested follow-up questions, and next steps.
        Returns 'No concern flagged' if the observation is within typical range.
    """
    prompt = f"""You are a child development specialist advising a CASA volunteer.

A {child_age}-year-old child's {domain} observation:
"{observation}"

Assess this observation:
1. CONCERN LEVEL: [None / Low / Medium / High]
2. CLINICAL CONTEXT: Why this may or may not be concerning for age {child_age}
3. FOLLOW-UP QUESTIONS: Specific questions to raise with teachers, therapists, or caseworkers
4. NEXT STEPS: What the volunteer should document or request

Do not diagnose. Frame as prompts for professional investigation."""
    result = llm.invoke(prompt)
    return result.content


# --- Custom Tool 4: Draft Court Report Section ---
@tool
def draft_court_report_section(
    section_name: str,
    volunteer_observations: str,
    child_initials: str,
    child_age: int
) -> str:
    """
    Drafts a single section of a CASA court report in the volunteer's voice.

    Produces a first-person draft of a specified court report section
    based on the volunteer's raw observations.

    Args:
        section_name: The section to draft. Valid values: 'placement',
                      'wellbeing', 'services', 'observations', 'recommendations'.
        volunteer_observations: Raw notes and observations from the volunteer.
        child_initials: The child's initials (e.g., 'A.M.') for reference.
        child_age: The child's age in years.

    Returns:
        A first-person draft of the specified court report section,
        prefaced with a reminder that it requires human review before submission.
    """
    llm_draft = init_chat_model(
        model="anthropic:claude-sonnet-4-20250514",
        temperature=0.7  # slightly creative for natural writing voice
    )
    prompt = f"""Help a CASA volunteer draft the '{section_name}' section
of a court report for Child {child_initials}, age {child_age}.

Write in first person as the advocate. Use natural, direct language.
Base the draft only on these observations:

{volunteer_observations}

Begin the draft immediately. Do NOT include preamble.
End with: [DRAFT — Review carefully. Verify all facts before submission.]"""
    result = llm_draft.invoke(prompt)
    return result.content


# Test a custom tool directly before using it in an agent
print("\n--- Testing summarize_case_flags tool directly ---")
test_result = summarize_case_flags.invoke({
    "case_notes": "Child A.M., age 9. In foster care. Hearing April 15. Missed 3 of 12 supervised visits with mother. IEP in place but school reports 8 absences."
})
print(test_result[:300] + "...")

print("\n--- Testing identify_service_gaps tool directly ---")
test_gaps = identify_service_gaps.invoke({
    "services_description": "Child is in foster home. Has IEP at school. No current therapist. Supervised visits with biological mother twice per month.",
    "child_age": 9
})
print(test_gaps[:300] + "...")


# ============================================================
# SECTION 6: EXTERNAL API TOOL
#
# Integrate at least one external service as a tool.
# Using Tavily web search for case law and legal resources.
# ============================================================
print("\n" + "="*60)
print("SECTION 6: EXTERNAL API TOOL — Tavily Web Search")
print("="*60)

# Set up Tavily client
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Create a web search tool using the Tavily client
@tool
def web_search(query: str) -> Dict[str, Any]:
    """
    Search the web for information relevant to CASA case preparation.

    Use this tool to find current case law, Virginia child welfare statutes,
    CASA guidelines, legal precedents, and local resource information to
    support a volunteer's court preparation or advocacy work.

    Args:
        query: A search query string describing what information is needed.

    Returns:
        A dictionary containing search results with titles, URLs,
        and relevant content snippets from the web.
    """
    return tavily_client.search(query)

# Test the web search tool
print("\n--- Testing web search tool ---")
search_result = web_search.invoke("Virginia child welfare permanency hearing requirements CASA volunteer")
print("Search returned results for: Virginia child welfare permanency hearing requirements")
pprint(search_result)


# ============================================================
# SECTION 2 (CONTINUED): AGENT CREATION
#
# Build two distinct agents, each with its own system prompt.
# Using create_agent() exactly as in the course Colab.
# ============================================================
print("\n" + "="*60)
print("SECTION 2: AGENT CREATION")
print("="*60)

# --- Agent 1: Court Preparation Agent ---
# Role: Helps volunteers prepare for upcoming court hearings.
# Tools: summarize_case_flags (custom), web_search (external API)
court_prep_agent = create_agent(
    model=llm,
    tools=[summarize_case_flags, web_search],
    system_prompt="""You are CASAgent's Court Preparation Specialist — an AI assistant
built exclusively for CASA (Court Appointed Special Advocate) volunteers
preparing for juvenile court hearings in Virginia.

YOUR ROLE:
- Summarize case files and extract the most critical facts
- Identify upcoming hearing dates, deadlines, and urgent concerns
- Surface relevant Virginia child welfare statutes and case law via web search
- Prepare the volunteer with questions to raise in court

YOUR PRINCIPLES:
- You AUGMENT the volunteer's judgment — you never replace it
- Never invent legal citations — use the web_search tool to find real ones
- Flag legal issues but defer to supervising attorneys for formal advice
- Always end with: "⚠ AI-generated assistance. Verify all facts with your
  case file and consult your CASA supervisor before court."
"""
)
print("✓ Agent 1 created: Court Preparation Agent")
print("  Tools: summarize_case_flags, web_search")

# --- Agent 2: Case Analysis Agent ---
# Role: Analyzes developmental observations and identifies service gaps.
# Tools: flag_developmental_concern, identify_service_gaps, draft_court_report_section
case_analysis_agent = create_agent(
    model=llm,
    tools=[flag_developmental_concern, identify_service_gaps, draft_court_report_section],
    system_prompt="""You are CASAgent's Case Analysis Specialist — an AI assistant
built exclusively for CASA volunteers conducting child welfare case analysis.

YOUR ROLE:
- Analyze volunteer observations for developmental concerns
- Identify gaps in services, evaluations, and documentation
- Draft court report sections that preserve the advocate's voice
- Help volunteers articulate what the child needs and why

YOUR PRINCIPLES:
- Be warm and accessible — these are dedicated volunteers, not clinicians
- Frame developmental observations as prompts for investigation, not diagnoses
- Preserve first-person voice when drafting reports
- Always end with: "⚠ AI-generated analysis. All observations require human
  verification. Report drafts must be reviewed before submission."
"""
)
print("✓ Agent 2 created: Case Analysis Agent")
print("  Tools: flag_developmental_concern, identify_service_gaps, draft_court_report_section")


# ============================================================
# SECTION 3: MESSAGE HANDLING
#
# Use HumanMessage and AIMessage to construct multi-turn
# conversations. Show that agents can process a sequence of messages.
# ============================================================
print("\n" + "="*60)
print("SECTION 3: MESSAGE HANDLING — Multi-turn conversation")
print("="*60)

# The main CASAgent prompt (equivalent to prompt0/prompt in the course Colab)
prompt = HumanMessage(content="""My case involves Child A.M., age 9, currently in a foster home.
She has a permanency hearing on April 15. Her biological mother completed parenting classes
but missed 3 of 12 supervised visits. The child has an IEP but has 8 school absences this
semester. The foster parent reports nightmares and difficulty concentrating.
What should I focus on to prepare for the hearing?""")

# Turn 1: Run the court prep agent with the initial case prompt
print("\n--- Turn 1: Volunteer submits case notes ---")
response1 = court_prep_agent.invoke(
    {"messages": [prompt]}
)
print(response1['messages'][-1].content)

# Turn 2: Multi-turn — volunteer follows up with additional context
# Using HumanMessage and AIMessage to construct the conversation sequence
print("\n--- Turn 2: Multi-turn with HumanMessage + AIMessage ---")
response_multiturn = court_prep_agent.invoke({
    "messages": [
        prompt,
        AIMessage(content="I'll look into those points before the hearing."),
        HumanMessage(content="Are there any Virginia statutes I should reference about missed supervised visits?")
    ]
})
print(response_multiturn['messages'][-1].content)

# Verify tool calls were made
print("\n--- Verifying tool calls in Turn 1 ---")
tool_calls = [m for m in response1['messages'] if hasattr(m, 'tool_calls') and m.tool_calls]
if tool_calls:
    for tc in tool_calls:
        print(f"Tool called: {tc.tool_calls}")
else:
    print("No tool calls made (agent answered from context)")


# ============================================================
# SECTION 4: STREAMING OUTPUT
#
# Implement streamed responses using agent.stream() with
# stream_mode="messages" for at least one agent interaction.
# ============================================================
print("\n" + "="*60)
print("SECTION 4: STREAMING OUTPUT")
print("="*60)

# Create a streaming-focused prompt for report drafting
stream_prompt = HumanMessage(content="""Please draft the 'observations' section of my court report
for Child A.M. (age 9). My observations: During my March 15 visit, the child was settled and
engaged with her foster family. She showed me artwork she made at school and seemed proud.
She asked about her mom and said she misses her. Foster parent reports improving behavior
and no incidents at school this week.""")

print("\n--- Streaming response from Case Analysis Agent ---")
print("(tokens print as they arrive)\n")

# Stream the agent output — exactly matching the course Colab pattern
for token, metadata in case_analysis_agent.stream(
    {"messages": [stream_prompt]},
    stream_mode="messages"
):
    if token.content:  # Check if there is actual content
        print(token.content, end="", flush=True)  # Print each token as it arrives

print("\n\n[Streaming complete]")


# ============================================================
# SECTION 7: AGENT MEMORY
#
# Add a checkpointer (InMemorySaver) to at least one agent
# so it can recall information from earlier in the conversation.
# ============================================================
print("\n" + "="*60)
print("SECTION 7: AGENT MEMORY — InMemorySaver checkpointer")
print("="*60)

# Create a memory-enabled version of the court prep agent
court_prep_agent_with_memory = create_agent(
    model=llm,
    tools=[summarize_case_flags, web_search],
    checkpointer=InMemorySaver(),
    system_prompt="""You are CASAgent's Court Preparation Specialist for CASA volunteers.
You have memory of our conversation and can recall earlier case details.
Always remind volunteers to verify AI output before court."""
)

# Thread config — thread_id ties messages together in memory
config = {"configurable": {"thread_id": "casa-case-AM-2024"}}

# First message — introduce the case
print("\n--- Memory Turn 1: Introduce the case ---")
response_mem1 = court_prep_agent_with_memory.invoke(
    {"messages": [HumanMessage(content="I'm preparing for Child A.M.'s permanency hearing on April 15. She is 9 years old, in foster care, and her mother missed 3 of 12 supervised visits.")]},
    config,
)
pprint(response_mem1['messages'][-1].content)

# Second message — agent should remember the case from Turn 1
print("\n--- Memory Turn 2: Testing recall from earlier in conversation ---")
question = HumanMessage(content="How many supervised visits did the mother miss, and why does that matter for the hearing?")
response_mem2 = court_prep_agent_with_memory.invoke(
    {"messages": [question]},
    config,
)
pprint(response_mem2['messages'][-1].content)

# Third message — further recall test
print("\n--- Memory Turn 3: Further recall test ---")
question2 = HumanMessage(content="What was the child's name and age we discussed?")
response_mem3 = court_prep_agent_with_memory.invoke(
    {"messages": [question2]},
    config,
)
pprint(response_mem3['messages'][-1].content)


# ============================================================
# SECTION 8: MULTI-AGENT ORCHESTRATION
#
# Build an orchestration agent that delegates tasks to sub-agents
# by wrapping them as callable tools. The orchestrator coordinates
# at least two sub-agents to produce a final result.
# ============================================================
print("\n" + "="*60)
print("SECTION 8: MULTI-AGENT ORCHESTRATION")
print("="*60)

# Wrap the Court Prep Agent as a callable tool
@tool
def call_court_prep_agent(case_description: str) -> str:
    """
    Call the Court Preparation Agent to analyze case notes and prepare
    a CASA volunteer for an upcoming court hearing.

    Use this tool when the user needs help summarizing case files,
    identifying urgent concerns, or finding relevant legal context
    for an upcoming juvenile court hearing.

    Args:
        case_description: A string describing the case, including child
                          details, hearing date, and any relevant context.

    Returns:
        A structured court preparation summary with flags, dates,
        placement status, and recommended actions.
    """
    response = court_prep_agent.invoke({
        "messages": [HumanMessage(content=case_description)]
    })
    return response["messages"][-1].content


# Wrap the Case Analysis Agent as a callable tool
@tool
def call_case_analysis_agent(observations_and_services: str) -> str:
    """
    Call the Case Analysis Agent to evaluate developmental observations
    and identify service gaps for a child in CASA care.

    Use this tool when the user needs help flagging developmental concerns,
    identifying missing services, or drafting a court report section based
    on the volunteer's observations.

    Args:
        observations_and_services: A string describing the volunteer's
                                   observations about the child and the
                                   services currently in place.

    Returns:
        An analysis of developmental concerns and service gaps with
        prioritized recommendations for the CASA volunteer.
    """
    response = case_analysis_agent.invoke({
        "messages": [HumanMessage(content=observations_and_services)]
    })
    return response["messages"][-1].content


# Create the Orchestration Agent
# It delegates to both sub-agents and synthesizes a combined result
orchestration_agent = create_agent(
    model=llm,
    tools=[call_court_prep_agent, call_case_analysis_agent],
    system_prompt="""You are CASAgent's Master Orchestrator — a senior AI coordinator
for CASA (Court Appointed Special Advocate) volunteers.

When a volunteer presents a case, you coordinate two specialist agents
to produce a comprehensive pre-hearing brief:

1. First, call call_court_prep_agent to analyze the case file,
   identify urgent flags, and find relevant legal context.

2. Then, call call_case_analysis_agent to evaluate developmental
   observations and identify service gaps.

3. Finally, synthesize both agents' outputs into a single, organized
   pre-hearing brief for the volunteer.

Always conclude with:
"⚠ This brief was generated by AI agents. All content must be reviewed
by the CASA volunteer and verified against official case files before court."
"""
)
print("✓ Orchestration Agent created")
print("  Sub-agent tools: call_court_prep_agent, call_case_analysis_agent")

# Run the orchestration agent with a complete case scenario
print("\n--- Running Orchestration Agent (coordinates both sub-agents) ---")
orchestration_prompt = HumanMessage(content="""
I need a complete pre-hearing brief for Child T.R., age 14, currently in a group home.

Case details:
- Placement: Group home since March 2023; prior placement with grandparents disrupted
- School: 12 absences this semester; teachers report declining engagement
- Mental health: No current therapist assigned; mental health evaluation overdue 4 months
- Upcoming hearing: Placement stability review, May 2, Judge Marcus Allen
- Recent visit observation: Child was polite but guarded; mentioned wanting to return to grandparents;
  group home staff report occasional conflicts with peers; no signs of physical harm

Please prepare me for this hearing by analyzing the case and identifying any gaps.
""")

response_orchestrated = orchestration_agent.invoke({
    "messages": [orchestration_prompt]
})

print("\n=== ORCHESTRATED PRE-HEARING BRIEF ===")
pprint(response_orchestrated['messages'][-1].content)

# Show that orchestration involved tool calls to sub-agents
print("\n--- Verifying orchestration called both sub-agents ---")
for i, msg in enumerate(response_orchestrated['messages']):
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  Orchestrator called: {tc['name']}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("CASAGENT MULTI-AGENT SYSTEM — COMPONENT SUMMARY")
print("="*60)
print("""
Component 1 — LLM Initialization:
  init_chat_model("anthropic:claude-sonnet-4-20250514")
  Demonstrated direct invocation at temperature=0 and temperature=0.7

Component 2 — Agent Creation:
  Agent 1: Court Preparation Agent (system prompt + 2 tools)
  Agent 2: Case Analysis Agent (system prompt + 3 tools)

Component 3 — Message Handling:
  Multi-turn conversation using HumanMessage + AIMessage sequences
  Demonstrated with court prep agent across 2 turns

Component 4 — Streaming Output:
  agent.stream() with stream_mode="messages"
  Applied to Case Analysis Agent drafting a court report section

Component 5 — Custom Tools (@tool decorator):
  summarize_case_flags(case_notes) → structured flag summary
  identify_service_gaps(services_description, child_age) → gap analysis
  flag_developmental_concern(observation, child_age, domain) → concern flag
  draft_court_report_section(section_name, observations, initials, age) → draft

Component 6 — External API Tool:
  web_search(query) via TavilyClient
  Used by Court Prep Agent to find Virginia statutes and case law

Component 7 — Agent Memory:
  InMemorySaver checkpointer on Court Prep Agent
  Thread ID: "casa-case-AM-2024"
  Demonstrated recall across 3 conversation turns

Component 8 — Multi-Agent Orchestration:
  Orchestration Agent wraps both sub-agents as callable tools:
    call_court_prep_agent() — delegates to Agent 1
    call_case_analysis_agent() — delegates to Agent 2
  Synthesizes both outputs into a unified pre-hearing brief
""")
