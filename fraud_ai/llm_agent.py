from openai import OpenAI
from fraud_ai.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def chatgpt_response(history, user_input):
    history.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history
    )
    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply


def llm_user_verification(user_text, conversation_history, system_prompt):
    last_msgs = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
    context = "\n".join([f"{m['role']}: {m['content']}" for m in last_msgs])
    prompt = (
        f"System context:\n{system_prompt}\n\n"
        f"Recent conversation:\n{context}\n"
        f"Customer just said (in their language): '{user_text}'\n\n"
        "Classify the customer's reply as one of the following categories:\n"
        "YES: Customer confirmed their identity (e.g., 'Yes, it's me', 'That's correct')\n"
        "NO: The person answering is NOT the customer (e.g., 'No, I'm his wife', 'This is not him')\n"
        "REPEAT: The customer's answer is unclear or unintelligible; ask to repeat\n"
        "OFFTOPIC: The customer is talking about something unrelated to identity verification\n"
        "CLARIFY: The customer is asking for clarification or to repeat (e.g., 'Who are you?', 'Why are you calling me?')\n\n"
        "Reply with only ONE word from: YES, NO, REPEAT, OFFTOPIC, CLARIFY.\n"
        "Example of valid reply: YES"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content.strip().upper()
    
    valid_responses = ["YES", "NO", "REPEAT", "OFFTOPIC", "CLARIFY"]
    if resp not in valid_responses:
        resp = "REPEAT"
    return resp




def llm_classify_user_reply(user_text, conversation_history, system_prompt):
    last_msgs = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
    context = "\n".join([f"{m['role']}: {m['content']}" for m in last_msgs])
    prompt = (
        f"System context:\n{system_prompt}\n\n"
        f"Recent conversation:\n{context}\n"
        f"Customer just said (in their language): '{user_text}'\n"
        "Classify the customer's reply as one of the following:\n"
        "- OK: The reply is clear and relevant to the transaction verification."
        "Also, if the customer asks for clarification about security, phishing, fraud, or similar terms, classify as OK.\n"
        "- FRAUD: Customer explicitly indicates the transaction was NOT made by them and is cleary fraudulent.\n"
        "- NOT FRAUD: Customer explicitly indicates the transaction was made by them.\n"
        "- REPEAT: The reply is unclear or ambiguous.\n"
        "- OFFTOPIC: The reply is off-topic meaning is not relevant to the conversation.\n"
        "- END: The customer is closing the conversation (e.g., says 'all clear', 'no questions', 'thank you', 'tutto chiaro', etc.)\n"
        "- CANT_TALK: customer says they cannot talk right now."
        "- CALL_BACK_LATER: customer agrees to be called later."
        "- NO_CALL_BACK: customer says they do not want to be called later."
    
        "Reply with only one word: OK, FRAUD, NOT FRAUD, REPEAT, OFFTOPIC, END, CANT_TALK, CALL_BACK_LATER, NO_CALL_BACK.\n"
        "Example of valid reply: OK"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content.strip().upper()
    if resp not in ["OK", "FRAUD", "NOT FRAUD", "REPEAT", "OFFTOPIC", "END", "CANT_TALK", "CALL_BACK_LATER", "NO_CALL_BACK"]:
        resp = "REPEAT"
    return resp



def llm_classify_investigation_reply(user_text, conversation_history, system_prompt):
    """
    Classifies a customer's response during the fraud investigation phase.

    INFO_COMPLETE = Sufficient info or has nothing else to add.
    INFO_INCOMPLETE = Relevant info but missing key details.
    REPEAT = Asked to repeat or unclear.
    OFFTOPIC = Unrelated to fraud investigation.
    END = Wants to end the conversation.
    """
    last_msgs = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
    context = "\n".join([f"{m['role']}: {m['content']}" for m in last_msgs])
    prompt = (
        f"System context:\n{system_prompt}\n\n"
        f"Recent conversation:\n{context}\n"
        f"Customer just said (in their language): '{user_text}'\n\n"
        "Classify as one word:\n"
        "- INFO_COMPLETE: Customer has given all necessary info OR has said they don't know anything else.\n"
        "- INFO_INCOMPLETE: Customer is giving relevant info but missing key details like data inserted in phishing form.\n"
        "- REPEAT: Reply is unclear, they ask to repeat.\n"
        "- OFFTOPIC: Reply not related to fraud investigation.\n"
        "- END: Customer wants to stop talking / end call."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content.strip().upper()

    valid = ["INFO_COMPLETE", "INFO_INCOMPLETE", "REPEAT", "OFFTOPIC", "END"]
    if resp not in valid:
        resp = "REPEAT"
    return resp

def llm_classify_help_reply(user_text, conversation_history, system_prompt):
    """
    Classify at the final help-offer step.
    """
    last_msgs = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
    context = "\n".join([f"{m['role']}: {m['content']}" for m in last_msgs])
    prompt = (
        f"System context:\n{system_prompt}\n\n"
        f"Recent conversation:\n{context}\n"
        f"Customer just said (in their language): '{user_text}'\n\n"
        "Classify as one word:\n"
        "- YES: They want further help or have another request.\n"
        "- NO: They do not need further help.\n"
        "- REPEAT: They ask to repeat the question.\n"
        "- OFFTOPIC: Answer is unrelated.\n"
        "- END: They want to close the conversation."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content.strip().upper()

    if resp not in ["YES", "NO", "REPEAT", "OFFTOPIC", "END"]:
        resp = "REPEAT"
    return resp


import json
from fraud_ai import blocked, whitelist, reset_password, alerts


def finalize_call_summary(db, alert, alerted_tx, history):
    """
    Generate a final summary of the conversation and decide on security actions
    (whitelist, block card, reset password) based on the history.
    """

    context = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history])

    prompt = f"""
You are a fraud prevention analyst assistant. 
Analyze the following conversation history between assistant and customer:

{context}

Tasks:
1. Provide a clear, concise summary of the discussion and the outcome. 
2. Decide which actions must be taken, from this list:
   - WHITELIST: The customer confirmed the alerted transaction as legitimate (no fraudulent transactions).
   - BLOCK_CARD: Fraudulent transactions and card data are compromised. (only if customer confirmed fraud)
   - RESET_PASSWORD: Customer credentials (password, online banking login) suspected to be compromised (only if customer confirmed fraud).
   Multiple actions can apply together. If no action is necessary return None to actions.
3. Return the result in JSON with two fields:
   {{
      "summary": "short text summary for analyst",
      "actions": ["WHITELIST" | "BLOCK_CARD" | "RESET_PASSWORD", ...]
   }}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    content = resp.choices[0].message.content.strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # fallback: wrap into JSON if model gave plain text
        result = {"summary": content, "actions": []}

    # --- Save summary to alert ---
    alerts.update_alert(db, alert.id, analyst_notes=result["summary"])

    # --- Apply security actions ---
    for action in result.get("actions", []):
        if action == "BLOCK_CARD":
            blocked.add_to_blocked(db, alerted_tx.card_number)
        elif action == "WHITELIST":
            whitelist.add_to_whitelist(db, alerted_tx.card_number)
        elif action == "RESET_PASSWORD":
            reset_password.add_password_reset(db, alerted_tx.card_number, reason="compromised credentials")

    return result
