# response_handlers.py
from fraud_ai.conversation import add_message
from fraud_ai.llm_agent import chatgpt_response

def send_and_log(db, alert_id, history, text):
    print(f"Agatha: {text}")
    add_message(db, alert_id, "assistant", text)
    history.append({"role": "assistant", "content": text})

def handle_repeat(db, alert_id, history, system_prompt):
    send_and_log(db, alert_id, history,
                 chatgpt_response(history, "Please politely ask the customer to repeat their last response."))
    return None, "REPEAT", False

def handle_offtopic(db, alert_id, history, system_prompt):
    send_and_log(db, alert_id, history,
                 chatgpt_response(history, "The conversation went off-topic, gently bring the focus back to reviewing and addressing the recent suspicious activity, while answering any related questions along the way."))
    return None, "OFFTOPIC", False



def handle_end(db, alert_id, history, system_prompt):
    send_and_log(db, alert_id, history,
                 chatgpt_response(history, "Please politely thank the customer and say goodbye."))
    return None, "END", True

def handle_cant_talk(db, alert_id, history, system_prompt):
    send_and_log(db, alert_id, history,
                 chatgpt_response(history, "Politely ask the customer if you can call them back later."))
    return None, "CANT_TALK", False

def handle_call_back_later(db, alert_id, history, system_prompt):
    send_and_log(db, alert_id, history,
                 chatgpt_response(history, "Politely thank the customer and confirm you'll call them back later. Say goodbye."))
    return None, "CALL_BACK_LATER", True

def handle_no_call_back(db, alert_id, history, system_prompt):
    send_and_log(db, alert_id, history,
                 chatgpt_response(history, "Politely inform the customer you will send them an email instead. Say goodbye."))
    return None, "NO_CALL_BACK", True
