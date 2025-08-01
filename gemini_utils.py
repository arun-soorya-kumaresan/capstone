# gemini_utils.py
import os
import google.generativeai as genai
from datetime import datetime

# --- Configure Gemini API ---
try:
    genai.configure(api_key="AIzaSyDvtbfjwI8-dln_nahCAQ81VEdnAKqecrg")
    GEMINI_ENABLED = True
except KeyError:
    GEMINI_ENABLED = False

# --- Helper Function for AI ---
def get_ai_recommendation(alert_type, context, conversation_history=None):
    """
    Generates AI recommendations based on context.
    
    To debug API calls, you can add a print statement here.
    For example:
    print(f"[{datetime.now()}] Calling Gemini API for alert_type: {alert_type}")
    """

    # ---  DEBUGGING BLOCK ---
    print("="*50)
    print(f"!!! GEMINI API CALLED !!!")
    print(f"Timestamp: {datetime.now()}")
    print(f"Alert Type: {alert_type}")
    print("="*50)
    if not GEMINI_ENABLED:
        return "AI is disabled. Please set your GEMINI_API_KEY."
    
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    base_prompt = "You are a world-class data center sustainability expert. Provide a concise, actionable recommendation with 'Likely Cause' and 'Recommended Action' sections."
    prompt = ""

    if alert_type == 'PUE_SPIKE':
        prompt = f"{base_prompt}\n**Alert:** PUE has spiked to **{context['PUE_Power_Usage_Effectiveness']:.2f}**."
    elif alert_type == 'UTILIZATION_DIP':
        prompt = f"{base_prompt}\n**Alert:** Server Utilization has dropped to **{context['Server_Utilization_Rate_Percent']:.1f}%**."
    elif alert_type == 'WUE_SPIKE':
        prompt = f"{base_prompt}\n**Alert:** WUE has spiked to **{context['WUE_Water_Usage_Effectiveness_L_per_kWh']:.2f} L/kWh**."
    elif alert_type == 'MANUAL_FOLLOW_UP':
        history_text = "\n".join([f"{entry['author']}: {entry['content']}" for entry in conversation_history])
        prompt = f"Based on the conversation history and real-time data, answer the user's latest question concisely.\n\n--- HISTORY ---\n{history_text}\n--- REAL-TIME DATA ---\n{context.to_string()}\n--- YOUR ANSWER ---"
    
    try:
        # print(f"DEBUG: Sending prompt to Gemini: {prompt[:100]}...") # <-- Add this for debugging
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        return f"Error calling Gemini API: {e}"