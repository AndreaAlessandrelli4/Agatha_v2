from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Transaction(Base):
    __tablename__ = 'transactions'
    id = Column(Integer, primary_key=True)
    card_number = Column(String, index=True)
    amount = Column(Float)
    timestamp = Column(DateTime, default=datetime.now())
    status = Column(String, default="pending")
    fraud_score = Column(Integer)
    is_fraud = Column(Boolean, default=False)
    alert_id = Column(Integer, ForeignKey('alerts.id'), nullable=True)
    merchant_id = Column(String)
    merchant_name = Column(String)
    mcc = Column(String)
    country = Column(String)

    # Add customer name fields here:
    customer_first_name = Column(String, nullable=False)
    customer_last_name = Column(String, nullable=False)

class Alert(Base):
    __tablename__ = 'alerts'
    id = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey('transactions.id'))
    created_at = Column(DateTime, default=datetime.now())
    status = Column(String, default="open")
    analyst_notes = Column(String)

class Whitelist(Base):
    __tablename__ = 'whitelist'
    id = Column(Integer, primary_key=True)
    card_number = Column(String, unique=True)
    whitelisted_at = Column(DateTime, default=datetime.now())

class BlockedCard(Base):
    __tablename__ = 'blocked_cards'
    id = Column(Integer, primary_key=True)
    card_number = Column(String, unique=True)
    blocked_at = Column(DateTime, default=datetime.now())

class AlertConversation(Base):
    __tablename__ = 'alert_conversations'
    id = Column(Integer, primary_key=True)
    alert_id = Column(Integer, ForeignKey('alerts.id'), index=True)
    role = Column(String, nullable=False)  # 'assistant' or 'user'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.now())


class PasswordReset(Base):
    __tablename__ = "password_resets"

    id = Column(Integer, primary_key=True, index=True)
    card_number = Column(String, index=True)
    reason = Column(String)
    timestamp = Column(DateTime, default=datetime.now())
