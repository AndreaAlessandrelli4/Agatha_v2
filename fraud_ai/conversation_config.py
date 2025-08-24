from fraud_ai.llm_agent import llm_classify_user_reply, llm_user_verification
from fraud_ai.response_handlers import (
    handle_repeat, handle_offtopic, handle_end,
    handle_cant_talk, handle_call_back_later, handle_no_call_back
)

# Map: classification label -> handler function
conversation_handlers = {
    "REPEAT": handle_repeat,
    "OFFTOPIC": handle_offtopic,
    "END": handle_end,
    "CANT_TALK": handle_cant_talk,
    "CALL_BACK_LATER": handle_call_back_later,
    "NO_CALL_BACK": handle_no_call_back
}

# Verification handlers (currently same as conversation_handlers)
verification_handlers = conversation_handlers.copy()

# Conversation state flow
CONVERSATION_FLOW = [
    {
        "name": "GREETING",
        "prompt": "Politely greet the customer, introduce yourself as Agata the AI Fraud Analyst from SAS Bank, and ask if you are speaking to {customer_first_name} {customer_last_name}.",
        "classifier": llm_classify_user_reply,
        "handlers": "conversation_handlers",
        "next": "MAIN_CONVO"
    },

    {
        "name": "MAIN_CONVO",
        "prompt": "Proceed with explaining the suspicious activity and ask relevant questions.",
        "classifier": llm_classify_user_reply,
        "handlers": "conversation_handlers",
        "next": "END"
    },
    {
        "name": "END",
        "prompt": "Politely thank the customer and say goodbye.",
        "classifier": None,
        "handlers": None,
        "next": None
    }
]

"""
    {
        "name": "VERIFICATION",
        "prompt": "Politely ask if you are speaking to the cardholder.",
        "classifier": llm_user_verification,
        "handlers": "verification_handlers",
        "next": "MAIN_CONVO"
    },
"""