import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# --- force load .env from project root (one level above fraud_ai) ---
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"[WARN] .env file not found at {dotenv_path}")

# --- Database URL ---
DATABASE_URL = "sqlite:///fraud_ai.db"


# --- Initialize session_state values if not set ---
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = os.getenv("OPENAI_API_KEY")

if "eleven_api_key" not in st.session_state:
    st.session_state["eleven_api_key"] = os.getenv("ELEVEN_API_KEY")


# --- DynamicKey helper class ---
class _DynamicKey:
    def __init__(self, state_name: str, env_var: str):
        self.state_name = state_name
        self.env_var = env_var

    def __call__(self) -> str | None:
        """
        Try session_state first (works inside Streamlit runtime).
        Fallback to environment variables (works in background threads).
        """
        value = None
        try:
            value = st.session_state.get(self.state_name, None)
        except Exception:
            # Session state not available (e.g. in non-Streamlit thread)
            value = None

        if value:
            return value
        return os.getenv(self.env_var, None)

    def __str__(self) -> str:
        value = self.__call__()
        if not value:
            raise RuntimeError(f"{self.env_var} missing! Did you set it in your .env file?")
        return value

    def __repr__(self) -> str:
        return f"<DynamicKey {self.env_var}={self.__call__()!r}>"


# --- Expose API keys as dynamic objects ---
OPENAI_API_KEY = _DynamicKey("openai_api_key", "OPENAI_API_KEY")
ELEVEN_KEY = _DynamicKey("eleven_api_key", "ELEVEN_API_KEY")

# --- Debug logging (optional, remove once working) ---
print("[CONFIG] OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("[CONFIG] ELEVEN_API_KEY loaded:", bool(os.getenv("ELEVEN_API_KEY")))