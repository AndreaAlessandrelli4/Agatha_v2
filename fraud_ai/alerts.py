from sqlalchemy.orm import Session
from .models import Alert

# CREATE
def create_alert(db: Session, transaction_id: int, status="open", analyst_notes=None):
    alert = Alert(
        transaction_id=transaction_id,
        status=status,
        analyst_notes=analyst_notes
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert

# READ (by ID)
def get_alert(db: Session, alert_id: int):
    return db.query(Alert).filter(Alert.id == alert_id).first()

# READ (all, with optional filters)
def get_alerts(db: Session, status=None, skip=0, limit=100):
    query = db.query(Alert)
    if status:
        query = query.filter(Alert.status == status)
    return query.offset(skip).limit(limit).all()

# UPDATE
def update_alert(db: Session, alert_id: int, **kwargs):
    alert = get_alert(db, alert_id)
    if not alert:
        return None
    for key, value in kwargs.items():
        setattr(alert, key, value)
    db.commit()
    db.refresh(alert)
    return alert

# DELETE
def delete_alert(db: Session, alert_id: int):
    alert = get_alert(db, alert_id)
    if not alert:
        return False
    db.delete(alert)
    db.commit()
    return True