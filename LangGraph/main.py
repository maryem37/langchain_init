# -- LangGraph Email Assistant - Fixed Version --

import os
import json
from dotenv import load_dotenv
from imap_tools import MailBox, AND
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

# ----------------- Config -----------------
load_dotenv()
IMAP_HOST = os.getenv("IMAP_HOST")
IMAP_USER = os.getenv("IMAP_USER")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD")
IMAP_FOLDER = "INBOX"
CHAT_MODEL = "qwen3:4b"

# ----------------- IMAP Connect -----------------
def connect():
    return MailBox(IMAP_HOST).login(IMAP_USER, IMAP_PASSWORD, initial_folder=IMAP_FOLDER)

# ----------------- Helper Functions -----------------
def list_unread_emails_func(limit=5):
    """Pure function to list unread emails with better error handling"""
    try:
        with connect() as mailbox:
            unread = []
            # Fetch unread emails directly
            for msg in mailbox.fetch(AND(seen=False), headers_only=True, mark_seen=False, reverse=True, limit=limit):
                unread.append({
                    "uid": msg.uid,
                    "subject": msg.subject or "(No Subject)",
                    "from": msg.from_ or "(Unknown)"
                })
            
            return unread
    except Exception as e:
        error_msg = f"Error listing emails: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

def summarize_email_func(uid: int):
    """Pure function to summarize an email with error handling"""
    try:
        with connect() as mailbox:
            # Fetch by UID
            messages = list(mailbox.fetch(AND(uid=str(uid)), mark_seen=False))
            if not messages:
                return {"error": f"Email with UID {uid} not found"}
            
            msg = messages[0]
            
            # Get text content safely
            text = msg.text or msg.html or "(No content)"
            
            return {
                "uid": uid,
                "subject": msg.subject or "(No Subject)",
                "from": msg.from_ or "(Unknown)",
                "date": msg.date.strftime("%Y-%m-%d %H:%M:%S") if msg.date else "Unknown",
                "summary": text[:300] + ("..." if len(text) > 300 else "")
            }
    except Exception as e:
        return {"error": f"Error summarizing email: {str(e)}"}

# ----------------- Tools (for LangChain) -----------------
@tool
def list_unread_emails(limit: int = 5):
    """Return the last unread emails (subject and UID)"""
    return list_unread_emails_func(limit)

@tool
def summarize_email(uid: int):
    """Summarize a single email"""
    return summarize_email_func(uid)

# ----------------- LangGraph Nodes -----------------
def tools_node(state):
    """Handle tool execution based on user input"""
    last_input = state["messages"][-1]["content"].lower()
    
    # List emails
    if "list" in last_input or "mails" in last_input or "emails" in last_input:
        result = list_unread_emails_func(limit=5)
        
        if isinstance(result, dict) and "error" in result:
            content = f"❌ {result['error']}"
        elif not result:
            content = "📭 No unread emails found."
        else:
            content = f"📬 Found {len(result)} unread email(s):\n\n"
            for i, email in enumerate(result, 1):
                content += f"{i}. [UID: {email['uid']}] {email.get('subject', 'No Subject')}\n"
                content += f"   From: {email.get('from', 'Unknown')}\n\n"
        
        # Mark as processed to avoid loop
        state["messages"].append({"role": "assistant", "content": content})
        state["processed"] = True
        return state
    
    # Summarize email
    elif "summary" in last_input or "summarize" in last_input or "read" in last_input:
        # Extract UID from input
        words = last_input.replace('[', ' ').replace(']', ' ').split()
        uid = None
        for word in words:
            if word.isdigit():
                uid = int(word)
                break
        
        if uid is None:
            content = "⚠️ Please specify an email UID to summarize (e.g., 'summarize 123')"
        else:
            summary = summarize_email_func(uid=uid)
            
            if "error" in summary:
                content = f"❌ {summary['error']}"
            else:
                content = f"📧 Email Summary:\n\n"
                content += f"Subject: {summary.get('subject', 'N/A')}\n"
                content += f"From: {summary.get('from', 'N/A')}\n"
                content += f"Date: {summary.get('date', 'N/A')}\n\n"
                content += f"Content:\n{summary.get('summary', 'No content')}"
        
        state["messages"].append({"role": "assistant", "content": content})
        state["processed"] = True
        return state
    
    # Unknown command
    else:
        content = "ℹ️ Available commands:\n- 'list unread emails'\n- 'summarize [UID]'\n- 'exit'"
        state["messages"].append({"role": "assistant", "content": content})
        state["processed"] = True
        return state

def router(state):
    """Route based on whether we've processed the command"""
    # If we've already processed in tools_node, go to END
    if state.get("processed", False):
        return "end"
    
    # Check if this is a tool command
    last_message = state["messages"][-1]["content"].lower()
    if any(word in last_message for word in ["list", "summary", "summarize", "mails", "emails", "read"]):
        return "tool"
    
    # Default: show help
    return "tool"

# ----------------- Builder -----------------
builder = StateGraph(dict)
builder.add_node("tool", tools_node)
builder.add_edge(START, "tool")
builder.add_conditional_edges("tool", router, {"tool": "tool", "end": END})
graph = builder.compile()

# ----------------- Main -----------------
if __name__ == "__main__":
    print("🚀 Starting LangGraph Email Assistant...")
    print("Commands: 'list unread emails', 'summarize [UID]', 'exit'\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Exiting LangGraph Email Assistant. Goodbye!")
                break
            
            if not user_input.strip():
                continue
            
            # Create fresh state for each interaction
            state = {
                "messages": [{"role": "user", "content": user_input}],
                "processed": False
            }
            
            # Invoke graph
            result = graph.invoke(state)
            
            # Print response
            if result["messages"]:
                print(f"\n{result['messages'][-1]['content']}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")