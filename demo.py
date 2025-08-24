import streamlit as st
from streamlit_autorefresh import st_autorefresh
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import pandas as pd
from sqlalchemy.orm import Session
from fraud_ai.data import get_db, update_transaction
from fraud_ai.alerts import get_alerts, update_alert
from fraud_ai.models import Transaction
from fraud_ai.whitelist import add_to_whitelist, is_card_whitelisted, remove_from_whitelist
from fraud_ai.blocked import add_to_blocked, is_card_blocked, remove_from_blocked
from fraud_ai.reset_password import add_password_reset, has_password_reset, remove_password_reset
from fraud_ai.conversation import get_conversation
from datetime import datetime

st.set_page_config(page_title="Fraud Alert Management", page_icon="🚨", layout="wide")

@st.cache_resource
def get_db_cached():
    return next(get_db())

db = get_db_cached()

@st.cache_data(ttl=15)
def load_alerts(refresh_counter):
    return get_alerts(db, limit=10)

@st.cache_data(ttl=15)
def load_transactions(card_number, refresh_counter):
    txs = db.query(Transaction).filter(
        Transaction.card_number == card_number
    ).order_by(Transaction.id.desc()).limit(10).all()
    return txs

def get_card_status(card_number):
    whitelisted = is_card_whitelisted(db, card_number) is not None
    blocked = is_card_blocked(db, card_number) is not None
    reset = has_password_reset(db, card_number) is not None
    return whitelisted, blocked, reset

def toggle_block_card(db: Session, card_number: str):
    try:
        if is_card_blocked(db, card_number):
            remove_from_blocked(db, card_number)
            return False
        else:
            add_to_blocked(db, card_number)
            return True
    except Exception:
        db.rollback()
        raise

def toggle_whitelist_card(db: Session, card_number: str):
    try:
        if is_card_whitelisted(db, card_number):
            remove_from_whitelist(db, card_number)
            return False
        else:
            add_to_whitelist(db, card_number)
            return True
    except Exception:
        db.rollback()
        raise

def toggle_password_reset(db: Session, card_number: str):
    try:
        if has_password_reset(db, card_number):
            remove_password_reset(db, card_number)
            return False
        else:
            add_password_reset(db, card_number, reason="manual analyst action")
            return True
    except Exception:
        db.rollback()
        raise

st_autorefresh(interval=500, limit=None, key="refresh")

if "alert_index" not in st.session_state:
    st.session_state.alert_index = 0
if "analyst_notes" not in st.session_state:
    st.session_state.analyst_notes = ""
if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0

def transactions_to_df_editable(txs, alert_tx_id):
    data = []
    for tx in txs:
        data.append({
            "id": tx.id,
            "Timestamp": tx.timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(tx.timestamp, datetime) else str(tx.timestamp),
            "Amount": tx.amount,
            "Merchant": tx.merchant_name,
            "Status": tx.status,
            "Fraud Score": f"{tx.fraud_score:.2f}" if tx.fraud_score is not None else "N/A",
            "Fraudulent": tx.is_fraud,
            "Alerted": "⚠️" if tx.id == alert_tx_id else ""
        })
    return pd.DataFrame(data)



######################################################
######################################################
######################################################
####### OLD ######
def render_conversation_old(alert_id):
    messages = get_conversation(db, alert_id)
    chat_text = ""
    for msg in messages:
        role = "Agata (AI)" if msg.role == "assistant" else "Customer"
        chat_text += f"**{role}:** {msg.content}\n\n"
    st.text_area("Call Conversation", value=chat_text, height=250, disabled=True)


####### NEW ######
def render_conversation(alert_id):
    messages = get_conversation(db, alert_id)

    st.markdown("""
        <style>
        .chat-bubble {
            border-radius: 12px;
            padding: 10px 15px;
            margin: 8px 0;
            max-width: 70%;
        }
        .assistant {
            background-color: #f0f2f6;
            color: #000;
            text-align: left;
        }
        .user {
            background-color: #4f8bf9;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("### 💬 Conversazione")

    for msg in messages:
        role_class = "assistant" if msg.role == "assistant" else "user"
        st.markdown(
            f'<div class="chat-bubble {role_class}"><b>{msg.role.capitalize()}:</b> {msg.content}</div>',
            unsafe_allow_html=True
        )

######################################################
######################################################
######################################################

alerts = load_alerts(st.session_state.refresh_counter)

def load_alert(index, refresh_counter):
    if index < 0 or index >= len(alerts):
        return None, None, None, None, None
    alert = alerts[index]
    tx = db.query(Transaction).filter(Transaction.id == alert.transaction_id).first()
    if not tx:
        return alert, None, None, None, None
    card_number = tx.card_number
    customer_name = f"{tx.customer_first_name} {tx.customer_last_name}" if tx.customer_first_name and tx.customer_last_name else "Unknown"
    notes = alert.analyst_notes or ""
    tx_list = load_transactions(card_number, refresh_counter)
    return alert, card_number, customer_name, notes, tx_list

alert, card_number, customer_name, notes, tx_list = load_alert(st.session_state.alert_index, st.session_state.refresh_counter)

if notes != st.session_state.get("analyst_notes", ""):
    st.session_state.analyst_notes = notes

#st.set_page_config(page_title="Fraud Alert Management", page_icon="🚨", layout="wide")

# ---- Styles ----
st.markdown("""<style>
body, .block-container {
    background-color: #121212;
    color: #e0e0e0;
    font-family: 'Poppins', sans-serif;
}
h1 { color: #00bcd4; }
.card-status-whitelisted { color: #4caf50; font-weight: 700; margin-right: 10px; }
.card-status-blocked { color: #d32f2f; font-weight: 700; margin-right: 10px; }
.card-status-reset { color: #ff9800; font-weight: 700; }
</style>""", unsafe_allow_html=True)

st.markdown('<div class="app-name">Your App Name Here</div>', unsafe_allow_html=True)

# ---- Alert Header ----
st.markdown('<div class="card">', unsafe_allow_html=True)
if alert is None:
    st.warning("No alerts to display.")
else:
    whitelisted, blocked, reset = get_card_status(card_number)
    status_parts = []
    if whitelisted:
        status_parts.append('<span class="card-status-whitelisted">✅ Whitelisted</span>')
    if blocked:
        status_parts.append('<span class="card-status-blocked">⛔ Blocked</span>')
    if reset:
        status_parts.append('<span class="card-status-reset">🔑 Password Reset</span>')
    status_html = " | ".join(status_parts) if status_parts else "No special status"

    st.markdown(f"<h1>Alert ID: {alert.id}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Transaction ID:</strong> {alert.transaction_id}</p>", unsafe_allow_html=True)
    st.markdown(f"""
    <p><strong>Card Number:</strong> {card_number} <br>
    <strong>Customer:</strong> {customer_name} <br>
    {status_html}</p>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)




# ---- Conversation ----
if alert is not None:
    render_conversation(alert.id)

# ---- Transactions ----
if alert is not None:
    with st.spinner("Loading transactions..."):
        df = transactions_to_df_editable(tx_list, alert.transaction_id)

        js_code = JsCode(f"""
        function(params) {{
            if (params.data.Fraudulent) {{
                return {{ 'class': 'ag-row-red' }};
            }}
            if (params.data.id === {alert.transaction_id}) {{
                return {{ 'class': 'ag-row-alerted' }};
            }}
            return null;
        }}
        """)

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_column("Fraudulent", editable=True, cellEditor='agCheckboxCellEditor')
        gb.configure_column("Fraud Score", type=["numericColumn"], precision=2)
        gb.configure_column("Alerted", editable=False)
        gb.configure_selection(selection_mode="single", use_checkbox=True)
        gb.configure_grid_options(getRowClass=js_code)
        grid_options = gb.build()

        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            allow_unsafe_jscode=True,
            theme='streamlit',
            height=300,
            fit_columns_on_grid_load=True,
        )

    edited_df = grid_response['data']

    # ---- Analyst Notes + Actions ----
    col_notes, col_buttons = st.columns([3,1])

    with col_notes:
        analyst_notes = st.text_area("Analyst Summary / Notes", value=st.session_state.analyst_notes)
        st.session_state.analyst_notes = analyst_notes

    with col_buttons:
        if st.button("✅ Whitelist Card", key="whitelist"):
            try:
                now_whitelisted = toggle_whitelist_card(db, card_number)
                if now_whitelisted:
                    st.success(f"Card {card_number} whitelisted.")
                else:
                    st.info(f"Card {card_number} removed from whitelist.")
                st.session_state.refresh_counter += 1
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error toggling whitelist: {e}")

        if st.button("⛔ Block Card", key="block"):
            try:
                now_blocked = toggle_block_card(db, card_number)
                if now_blocked:
                    st.error(f"Card {card_number} blocked.")
                else:
                    st.info(f"Card {card_number} unblocked.")
                st.session_state.refresh_counter += 1
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error toggling block: {e}")

        if st.button("🔑 Reset Password", key="reset"):
            try:
                now_reset = toggle_password_reset(db, card_number)
                if now_reset:
                    st.warning(f"Password reset flagged for card {card_number}.")
                else:
                    st.info(f"Password reset cleared for card {card_number}.")
                st.session_state.refresh_counter += 1
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error toggling reset: {e}")

    # ---- Nav & Save ----
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅️ Previous Alert"):
            if st.session_state.alert_index > 0:
                st.session_state.alert_index -= 1
                st.session_state.analyst_notes = ""
                st.session_state.refresh_counter += 1
                st.experimental_rerun()
            else:
                st.warning("This is the first alert.")

    with col2:
        if st.button("💾 Save Changes"):
            update_alert(db, alert.id, analyst_notes=analyst_notes)
            for _, row in edited_df.iterrows():
                update_transaction(db, row['id'], is_fraud=row['Fraudulent'])
            st.session_state.refresh_counter += 1
            st.success("Changes saved.")
            st.experimental_rerun()

    with col3:
        if st.button("Next Alert ➡️"):
            if st.session_state.alert_index < len(alerts) - 1:
                st.session_state.alert_index += 1
                st.session_state.analyst_notes = ""
                st.session_state.refresh_counter += 1
                st.experimental_rerun()
            else:
                st.warning("This is the last alert.")