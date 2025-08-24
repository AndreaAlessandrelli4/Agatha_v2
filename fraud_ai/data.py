from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .config import DATABASE_URL
from .models import Base, Transaction



engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------
# Transaction CRUD functions
# ---------------------------

# CREATE
def create_transaction(db: Session, **kwargs):
    tx = Transaction(**kwargs)
    db.add(tx)
    db.commit()
    db.refresh(tx)
    return tx

# READ (by ID)
def get_transaction(db: Session, tx_id: int):
    return db.query(Transaction).filter(Transaction.id == tx_id).first()

# READ (all, with optional filters)
def get_transactions(db: Session, skip=0, limit=100):
    return db.query(Transaction).offset(skip).limit(limit).all()

# UPDATE
def update_transaction(db: Session, tx_id: int, **kwargs):
    tx = get_transaction(db, tx_id)
    if not tx:
        return None
    for key, value in kwargs.items():
        setattr(tx, key, value)
    db.commit()
    db.refresh(tx)
    return tx

# DELETE
def delete_transaction(db: Session, tx_id: int):
    tx = get_transaction(db, tx_id)
    if not tx:
        return False
    db.delete(tx)
    db.commit()
    return True