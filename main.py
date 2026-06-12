import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from tools import tools
from datetime import datetime

load_dotenv()

st.set_page_config(
    page_title="Arya AI",
    page_icon="🪷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: #09090b;
    color: #fafafa;
    height: 100%;
}

/* Hide all streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton { display: none !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #27272a; border-radius: 4px; }

/* ── NAVBAR ── */
.navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 56px;
    background: rgba(9,9,11,0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid #18181b;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    z-index: 1000;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 8px;
}
.nav-logo {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, #f97316, #ec4899);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
}
.nav-name {
    font-size: 1rem;
    font-weight: 700;
    color: #fafafa;
    letter-spacing: -0.02em;
}
.nav-right {
    display: flex;
    align-items: center;
    gap: 8px;
}
.nav-pill {
    font-size: 0.65rem;
    font-weight: 500;
    color: #f97316;
    background: rgba(249,115,22,0.1);
    border: 1px solid rgba(249,115,22,0.2);
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.nav-dot {
    width: 7px; height: 7px;
    background: #22c55e;
    border-radius: 50%;
    animation: blink 2s infinite;
}
@keyframes blink {
    0%,100% { opacity:1; }
    50% { opacity:0.3; }
}

/* ── CHAT AREA ── */
.chat-scroll {
    position: fixed;
    top: 56px;
    left: 0; right: 0;
    bottom: 80px;
    overflow-y: auto;
    padding: 24px 16px;
}

/* ── MESSAGES ── */
.msg-wrap {
    max-width: 760px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding-bottom: 8px;
}

.msg {
    display: flex;
    gap: 10px;
    align-items: flex-end;
}
.msg.user { flex-direction: row-reverse; }

.ava {
    width: 30px; height: 30px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 600;
}
.ava.ai {
    background: linear-gradient(135deg, #f97316, #ec4899);
    color: white;
}
.ava.user {
    background: #27272a;
    color: #a1a1aa;
    font-size: 11px;
}

.bbl {
    padding: 11px 15px;
    border-radius: 18px;
    font-size: 0.875rem;
    line-height: 1.6;
    max-width: min(520px, 75vw);
    word-break: break-word;
}
.bbl.ai {
    background: #18181b;
    border: 1px solid #27272a;
    color: #d4d4d8;
    border-bottom-left-radius: 4px;
}
.bbl.user {
    background: linear-gradient(135deg, #ea580c, #db2777);
    color: #fff;
    border-bottom-right-radius: 4px;
}

.msg-meta {
    font-size: 0.62rem;
    color: #3f3f46;
    margin-top: 4px;
    padding: 0 4px;
}
.msg.user .msg-meta { text-align: right; }

/* ── EMPTY STATE ── */
.empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: calc(100vh - 160px);
    gap: 16px;
    text-align: center;
    padding: 20px;
}
.empty-icon {
    font-size: 2.5rem;
    margin-bottom: 4px;
}
.empty-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #fafafa;
    letter-spacing: -0.03em;
}
.empty-sub {
    font-size: 0.82rem;
    color: #52525b;
    max-width: 320px;
    line-height: 1.6;
}
.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    max-width: 480px;
    margin-top: 8px;
}
.chip {
    background: #18181b;
    border: 1px solid #27272a;
    color: #71717a;
    padding: 8px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    cursor: default;
}

/* ── INPUT BAR ── */
.input-bar {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    height: 80px;
    background: rgba(9,9,11,0.95);
    backdrop-filter: blur(12px);
    border-top: 1px solid #18181b;
    display: flex;
    align-items: center;
}


/* Input field override */
div[data-testid="stTextInput"] {
    flex: 1 !important;
}
div[data-testid="stTextInput"] > div {
    background: transparent !important;
}
div[data-testid="stTextInput"] input {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: #fafafa !important;
    font-size: 0.875rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 8px 0 !important;
    caret-color: #f97316 !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #3f3f46 !important;
}
div[data-testid="stTextInput"] input:focus {
    border: none !important;
    box-shadow: none !important;
}

/* Send button */
.stButton > button {
    background: linear-gradient(135deg, #ea580c, #db2777) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 9px 18px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    white-space: nowrap !important;
    min-width: 70px !important;
    transition: opacity 0.15s !important;
    height: 38px !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}
.stButton > button:active {
    transform: scale(0.97) !important;
}

/* Spinner */
.stSpinner { position: fixed !important; bottom: 90px !important; left: 50% !important; transform: translateX(-50%) !important; }
.stSpinner > div { border-top-color: #f97316 !important; }

/* Error */
.stAlert { border-radius: 10px !important; font-size: 0.8rem !important; max-width: 760px; margin: 0 auto; }

/* ── RESPONSIVE ── */
@media (max-width: 640px) {
    .navbar { padding: 0 16px; }
    .nav-name { font-size: 0.9rem; }
    .chat-scroll { padding: 16px 10px; }
    .bbl { font-size: 0.82rem; max-width: 82vw; }
    .input-bar { padding: 0 0px; }
    .empty-title { font-size: 1.1rem; }
}
</style>
""", unsafe_allow_html=True)


# ── Agent ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_agent():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Arya, a smart and friendly AI assistant. Be helpful, clear and concise.
Answer naturally. Never output raw function/tool call syntax in your replies."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=5)


# ── Session ───────────────────────────────────────────────────────────────────
for key, val in [("chat_history", []), ("messages", []), ("input_key", 0)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="nav-brand">
        <div class="nav-logo">🪷</div>
        <span class="nav-name">Arya</span>
    </div>
    <div class="nav-right">
        <span class="nav-pill">AI Assistant</span>
        <div class="nav-dot"></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Messages ──────────────────────────────────────────────────────────────────
st.markdown('<div class="chat-scroll"><div class="msg-wrap">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="empty">
        <div class="empty-icon">🪷</div>
        <div class="empty-title">Namaste! Main Arya hoon.</div>
        <div class="empty-sub">Your intelligent AI assistant. Ask me anything in Hindi or English.</div>
        <div class="chips">
            <span class="chip">What is machine learning?</span>
            <span class="chip">Calculate 20% of 5000</span>
            <span class="chip">What time is it?</span>
            <span class="chip">Write a Python function</span>
            <span class="chip">Explain black holes</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
        time_str = msg.get("time", "")

        if role == "user":
            st.markdown(f"""
            <div class="msg user">
                <div class="ava user">You</div>
                <div>
                    <div class="bbl user">{content}</div>
                    <div class="msg-meta">{time_str}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg">
                <div class="ava ai">A</div>
                <div>
                    <div class="bbl ai">{content}</div>
                    <div class="msg-meta">Arya · {time_str}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)


# ── Input bar ─────────────────────────────────────────────────────────────────
st.markdown('<div class="input-bar">', unsafe_allow_html=True)

col1, col2 = st.columns([9, 1])
with col1:
    user_input = st.text_input(
        "msg",
        placeholder="Message Arya...",
        label_visibility="collapsed",
        key=f"input_{st.session_state.input_key}"
    )
with col2:
    send = st.button("Send")

st.markdown('</div></div>', unsafe_allow_html=True)


# ── Handle send ───────────────────────────────────────────────────────────────
if send and user_input and user_input.strip():
    text = user_input.strip()
    now = datetime.now().strftime("%I:%M %p")

    st.session_state.messages.append({"role": "user", "content": text, "time": now})
    st.session_state.input_key += 1  # clears input field

    with st.spinner(""):
        try:
            executor = get_agent()
            result = executor.invoke({
                "input": text,
                "chat_history": st.session_state.chat_history
            })
            reply = result["output"]

            st.session_state.chat_history.append(HumanMessage(content=text))
            st.session_state.chat_history.append(AIMessage(content=reply))
            st.session_state.messages.append({
                "role": "ai",
                "content": reply,
                "time": datetime.now().strftime("%I:%M %p")
            })

        except Exception as e:
            st.session_state.messages.append({
                "role": "ai",
                "content": f"Sorry, something went wrong: {str(e)}",
                "time": datetime.now().strftime("%I:%M %p")
            })

    st.rerun()
