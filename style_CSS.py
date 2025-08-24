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
    """

# CSS premium corretto
chat_style ="""
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


tab_style ='''<style>
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
                        </style>'''


bottom_style = """
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
            """