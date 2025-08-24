import streamlit as st
from streamlit_autorefresh import st_autorefresh
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from st_aggrid import StAggridTheme
import pandas as pd
from sqlalchemy.orm import Session
from fraud_ai.data import get_db, update_transaction
from fraud_ai.config import OPENAI_API_KEY, ELEVEN_KEY, DATABASE_URL
from fraud_ai.alerts import get_alerts, update_alert
from fraud_ai.models import Transaction
from fraud_ai.whitelist import add_to_whitelist, is_card_whitelisted, remove_from_whitelist
from fraud_ai.blocked import add_to_blocked, is_card_blocked, remove_from_blocked
from fraud_ai.reset_password import add_password_reset, has_password_reset, remove_password_reset
from fraud_ai.conversation import get_conversation
from datetime import datetime
import time

st.set_page_config(page_title="Fraud Alert Management", page_icon="🕵️", layout="wide", initial_sidebar_state="expanded")

# inizializza le variabili di sessione se non esistono
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None
if "eleven_api_key" not in st.session_state:
    st.session_state.eleven_api_key = None

# se le chiavi non ci sono → chiedi login
if not st.session_state.openai_api_key or not st.session_state.eleven_api_key:
    st.title("Enter your API keys")

    openai_input = st.text_input("OpenAI API Key", type="password")
    eleven_input = st.text_input("ElevenLabs API Key", type="password")

    if st.button("Login"):
        if openai_input and eleven_input:
            st.session_state.openai_api_key = openai_input
            st.session_state.eleven_api_key = eleven_input
            st.rerun()
        else:
            st.error("You must enter both keys to continue.", icon="⚠️")
else:
    # ======= INTERFACCIA DELLA TUA APP =========

    page_style ="""
        <style>
        /* Reset generale */
        body, .block-container{
            background-color: #F9FAFB; /* quasi bianco */
            color: #333333;
            font-family: 'Inter', sans-serif;
        }

        .Title{
            color: #333333;
            font-size: 50px;
            font-family: 'Inter', sans-serif;
            font-weight: bold;
        }
        

        .subTitle{
            color: #333333;
            font-size: 20px;
            font-family: 'Inter', sans-serif;
            font-weight: bold;
        }

        .notes{
            color: #656E6A;
            font-size: 15px;
            font-family: 'Inter', sans-serif;
            font-weight: italic;
        }

        .identificativo{
            color: #656E6A;
            font-size: 10px;
            font-family: 'Inter', sans-serif;
            font-weight: bold;
        }

        .valore{
            color: #333333;
            font-size: 10px;
            font-family: 'Inter', sans-serif;
            font-weight: bold;
        }

        .identificativoCost{
            color: #656E6A;
            font-size: 15px;
            font-family: 'Inter', sans-serif;
            font-weight: bold;
        }

        .valoreCost{
            color: #333333;
            font-size: 15px;
            font-family: 'Inter', sans-serif;
            font-weight: bold;
        }

        .identificativoCard{
            color: #656E6A;
            font-size: 15px;
            font-family: 'Inter', sans-serif;
            font-weight: bold;
        }

        .valoreCard{
            color: #333333;
            font-size: 15px;
            font-family: 'Inter', sans-serif;
            font-weight: bold;
        }

        .valoreBlocked{
            color: #ff0000;
            font-size: 20px;
            font-family: 'Inter', sans-serif;
            font-weight: bold;
        }

        .valoreReset{
            color: #333333;
            font-size: 20px;
            font-family: 'Inter', sans-serif;
            font-weight: bold;
        }

        .valoreWhit{
            color: #333333;
            font-size: 20px;
            font-family: 'Inter', sans-serif;
            font-weight: bold;
        }

        .divisore{
            color: #333333;
            font-size: 10px;
            font-family: 'Inter', sans-serif;
        }

        h1 {
            color: #2F67F5;
            font-size: 30px;
        }

        h2 {
            color: #CBD1D0;
            font-size: 20px;
        }

        h3, h4 {
            color: #2F67F5;
            font-weight: 300;
        }

        /* Header card */
        .stMarkdown h2, .stMarkdown h3 {
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }

        /* Containers stile "card" */
        .st-key-gino {
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 2px 2px 2px 2px #333333;
            margin-bottom: 1rem;
        }

        .st-key-gino1 {
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 2px 2px 2px 2px #333333;
            margin-bottom: 1rem;
        }



        /* Bottoni */
        
        .st-key-saving .stButton > button {
            background-color: #DBEEDB;
            color: #333333;
            border: 2px solid #000;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
            font-size: 80px;
            transition: all 0.3s ease;
        }

        .st-key-saving .stButton > button:hover {
            background-color: #1D4ED8;
            transform: scale(1.05);
        }
        

        /* Note box */
        textarea {
            border-radius: 16px !important;
            border: 1px solid #E5E7EB !important;
            padding: 10px !important;
            font-size: 10px;
        }
        </style>
    """

    # --- DB CONNECTION ---
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

    # --- Helpers ---
    def get_card_status(card_number):
        whitelisted = is_card_whitelisted(db, card_number) is not None
        blocked = is_card_blocked(db, card_number) is not None
        reset = has_password_reset(db, card_number) is not None
        return whitelisted, blocked, reset

    # --- Toggle actions ---
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
    c_succ, _, c_logout = st.columns([2,2,1])
    with c_succ:
        st.markdown("""
        <style>
        .custom-success {
            background-color: #D4EDDA !important;   /* colore di sfondo */
            color: #155724 !important;              /* colore del testo */
            border: 2px solid #C3E6CB !important;   /* bordo */
            border-radius: 10px !important;         /* angoli arrotondati */
            padding: 15px !important;
            font-size: 20px !important;             /* dimensione testo */
            font-weight: bold !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # inizializza flag nella sessione
        if "show_success" not in st.session_state:
            st.session_state.show_success = True

        if st.session_state.show_success:
            st.markdown('<div class="custom-success">✅ API keys successfully uploaded!</div>', unsafe_allow_html=True)
            st.session_state.show_success = False  # non verrà più mostrato
    with c_logout:
        if st.button("Logout"):
            st.session_state.openai_api_key = None
            st.session_state.eleven_api_key = None
            st.rerun()
            
    # --- Auto refresh ---
    st_autorefresh(interval=10000, limit=None, key="refresh")

    if "alert_index" not in st.session_state:
        st.session_state.alert_index = 0
    if "analyst_notes" not in st.session_state:
        st.session_state.analyst_notes = ""
    if "refresh_counter" not in st.session_state:
        st.session_state.refresh_counter = 0
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

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

    # --- Conversation UI Premium Fixed (User Text Right-Aligned Corrected) ---
    def render_conversation(alert_id):
        messages = get_conversation(db, alert_id)

        # aggiorna solo i nuovi messaggi
        if len(messages) > len(st.session_state.chat_messages):
            st.session_state.chat_messages = messages

        
        # CSS premium corretto
        style ="""
            <style>

            /* Riga messaggio: icona + bubble */
            .chat-row {
                display: flex !important; /* forza span a comportarsi da div */
                align-items: flex-start;
                margin: 8px 0;
                width: 100%; /* evita background "a righe" */
            }

            /* Bubble messaggi */
            .chat-bubble {
                border-radius: 20px;
                padding: 10px 15px;
                max-width: 70%;
                word-wrap: break-word;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }

            /* Icona generica */
            .icon {
                width: 30px;
                height: 30px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                flex-shrink: 0;
                border: 2px solid #323232;
            }

            /* Messaggi assistente */
            .assistant {
                flex-direction: row; /* icona a sinistra, messaggio a destra */
                justify-content: flex-start;
            }
            .assistant .icon {
                background-color: #E9F7FE;
                margin-right: 10px;
            }
            .assistant .chat-bubble {
                background-color: #E9F7FE;
                color: #323232;
            }

            /* Messaggi utente */
            .user {
                flex-direction: row-reverse; /* icona a destra, messaggio a sinistra */
                justify-content: flex-end;
                text-align: right;
            }
            .user .icon {
                background-color: #FDF4F4;
                margin-left: 10px;
            }
            .user .chat-bubble {
                background-color: #FDF4F4;
                color: #323232;
                text-align: right;
                margin-left: auto;
            }
            .message-text{
                font-size: 12px;
            }
            </style>
        """
        # HTML chat
        with st.container(key = 'chatContainer', height=130):
            chat_html = ''
            for msg in st.session_state.chat_messages:
                role_class = "assistant" if msg.role == "assistant" else "user"
                icon = "🤖" if msg.role == "assistant" else "👤"
                content = msg.content.replace("\n", "<br>")  # gestisce a capo
                chat_html += f'''<span class="chat-row {role_class}">
                        <span class="icon">{icon}</span>
                        <span class="chat-bubble">
                        <span class="message-text">{content}</span>
                        </span>
                    </span>
                '''
            chat_html += ''

            return st.markdown(style+chat_html, unsafe_allow_html=True)





    # --- Load alert ---
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
    # --- Header ---
    if alert is None:
        st.markdown(page_style + f"<span class='Title'>No alerts to display.</span><br>", unsafe_allow_html=True)
    else:
        whitelisted, blocked, reset = get_card_status(card_number)
        status_parts = []
        if whitelisted:
            status_parts.append("✅ Whitelisted")
        if blocked:
            status_parts.append("⛔ Blocked")
        if reset:
            status_parts.append("🔑 Password Reset")
        col_id1, col_id2, _  = st.columns([2,1,1])
        with col_id1:
            col_11, col_22,_= st.columns([3,1,1])
            with col_11:
                st.markdown(page_style + f"<span class='subTitle'>🚨 Alert ID: {alert.id}</span> <br><span class='identificativoCost'>Customer: </span> <span class='valoreCost'>{customer_name}</span><span class='identificativoCost'> | </span><span class='identificativoCost'>  Transaction ID: </span> <span class='valoreCost'>{alert.transaction_id}</span>", unsafe_allow_html=True)


    # --- Main Layout ---
    if alert is not None:
        col_conv, col_annotation = st.columns([1.5,1])
        with col_conv:
            st.markdown('''<style>
                        /* set the background color of many elements across the grid */
                        .ag-theme-alpine {
                            --ag-background-color: #ddd !important;
                        }

                        /* change the font style of a single UI component */
                        .ag-theme-alpine .ag-header-cell-label {
                            font-style: italic !important;
                        }
                        .ag-row-red {
                            background-color: #ffcccc !important;
                        }
                        .ag-row-alerted {
                            background-color: #ffff99 !important;
                            font-weight: bold;
                        }
                        </style>''', unsafe_allow_html=True)
            with st.container(border=True, key='gino', height=250):
                    st.markdown("<span class='subTitle'> 💬 Conversation Transcript</span>", unsafe_allow_html=True)
                    render_conversation(alert.id)
        
        with col_annotation:
            # Analyst notes + actions
            st.markdown("<span class='subTitle'>✍️ Analyst Notes & Actions</span>", unsafe_allow_html=True)
            with st.container(border=True, height=130):
                text = st.session_state.analyst_notes.replace("$", "\$").replace("\n", "<br>")
                analyst_notes = st.markdown(f"<span class='notes'>{text}</span>", unsafe_allow_html=True)
                st.session_state.analyst_notes = analyst_notes
            st.markdown(f"<span class='identificativoCard'>Card: </span> <span class='valoreCard'>{card_number}</span><br><span class='identificativoCard'>Action: </span> <span class='valoreCard'>{' | '.join(status_parts) if status_parts else 'No special status'}</span>", unsafe_allow_html=True)

        _, col_buttons1, col_buttons2, col_buttons3,_ = st.columns([1,1,1,1,1])
        st.markdown("""
            <style>
            div.stButton > button {
            background-color: #F9FBFA;
            color: #333333;
            border-radius: 9999px; /* pill shape */
            padding: 0.6rem 1.2rem;
            box-shadow: 2px 2px 2px 2px #333333;
            font-weight: bold;
            font-size: 10px;
            transition: all 0.3s ease;
            }

            
            .st-key-Whitelist .stButton button {
            font-size: 3px;
            color: #333333;
            background-color: #F9FBFA;
            }
                    
            .st-key-Whitelist .stButton button:hover {
            background-color: #BEF6DC;
            transform: translateY(-2px);
            }
                    
            .st-key-Block .stButton button {
            font-size: 3px;
            color: #333333;
            background-color: #F9FBFA;
            }
                    
            .st-key-Block .stButton button:hover {
            background-color: #F6BEBE;
            transform: translateY(-2px);
            }
                    

            .st-key-Reset .stButton button {
            font-size: 3px;
            color: #333333;
            background-color: #F9FBFA;
            }
                    
            .st-key-Reset .stButton button:hover {
            background-color: #F6F3BE;
            transform: translateY(-2px);
            }
            </style>
            """, unsafe_allow_html=True)

        with col_buttons1:
            if st.button("Whitelist Card", key="Whitelist", icon="✅"):
                toggle_whitelist_card(db, card_number)
                st.session_state.refresh_counter += 1
                st.experimental_rerun()
        with col_buttons2:
            if st.button("Block Card", icon="⛔", key="Block"):
                toggle_block_card(db, card_number)
                st.session_state.refresh_counter += 1
                st.experimental_rerun()
        with col_buttons3:
            if st.button("Reset Password", icon = "🔑", key="Reset"):
                toggle_password_reset(db, card_number)
                st.session_state.refresh_counter += 1
                st.experimental_rerun()





        with st.container(border=True, height=200):
            with st.spinner("Loading transactions..."):
                df = transactions_to_df_editable(tx_list, alert.transaction_id)

                js_code = JsCode(f"""
                function(params) {{
                    if (params.data.id === {alert.transaction_id}) {{
                        return {{'fontWeight': 'bold', 'backgroundColor': '#ffff99'}};
                    }}
                    if (params.data.Fraudulent === true) {{
                        return {{'backgroundColor': '#ffcccc'}};
                    }}
                    return null;
                }}
                """)
                
                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_column("Fraudulent", editable=True, cellEditor='agCheckboxCellEditor')
                gb.configure_column("Fraud Score", type=["numericColumn"], precision=2)
                gb.configure_column("Alerted", editable=False)
                gb.configure_selection(selection_mode="multiple", use_checkbox=True, pre_selected_rows=[0,1])
                gb.configure_grid_options(getRowStyle=js_code)
                gb.configure_auto_height(autoHeight=True)
                grid_options = gb.build()
                #rowClassRules = {"bg-danger": "params.data.Fraudulent == 1"}



                custom_theme = (  
                    StAggridTheme(base="alpine") 
                    .withParams(fontSize=15,
                    rowBorder=True,
                    backgroundColor="#FFFFFF")  
                    .withParts('iconSetQuartz')  
                )


                grid_response = AgGrid(
                    df,
                    update_mode=GridUpdateMode.MODEL_CHANGED,
                    allow_unsafe_jscode=True,
                    columnSize="sizeToFit",
                    theme = custom_theme,
                    gridOptions=grid_options,
                    fit_columns_on_grid_load=True,
                    use_container_width=True
                )

                edited_df = grid_response['data']


        

        with st.container(border=True, height=66, key='saving'):
            _, col1, col2, col3,_ = st.columns([2,1,1,1,2])
            with col1:
                if st.button("Previous Alert", icon="⬅️"):
                    st.session_state.alert_index = max(0, st.session_state.alert_index - 1)
                    st.experimental_rerun()
            with col2:
                if st.button("Save Changes", icon="💾"):
                    update_alert(db, alert.id, analyst_notes=analyst_notes)
                    for _, row in edited_df.iterrows():
                        update_transaction(db, row['id'], is_fraud=row['Fraudulent'])
                    st.session_state.refresh_counter += 1
                    st.success("Changes saved.")
                    st.experimental_rerun()
            with col3:
                if st.button("Next Alert", icon = "➡️"):
                    st.session_state.alert_index = min(len(alerts)-1, st.session_state.alert_index + 1)
                    st.experimental_rerun()
