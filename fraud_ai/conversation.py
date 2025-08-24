from sqlalchemy.orm import Session
from .models import AlertConversation
from datetime import datetime

def add_message(db: Session, alert_id: int, role: str, content: str):
    msg = AlertConversation(
        alert_id=alert_id,
        role=role,
        content=content,
        timestamp=datetime.utcnow()
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg

def get_conversation(db: Session, alert_id: int):
    return db.query(AlertConversation).filter(AlertConversation.alert_id == alert_id).order_by(AlertConversation.timestamp).all()