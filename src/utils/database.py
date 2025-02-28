"""
Database module for experiment tracking.
Provides SQLAlchemy models and utilities for storing and retrieving
experiment configurations and training results.
"""

from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime

# Create database engine
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///federated_learning.db')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TrainingRound(Base):
    __tablename__ = "training_rounds"

    id = Column(Integer, primary_key=True, index=True)
    round_number = Column(Integer)
    accuracy = Column(Float)
    privacy_loss = Column(Float)
    num_clients = Column(Integer)
    privacy_budget = Column(Float)
    noise_scale = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ExperimentConfig(Base):
    __tablename__ = "experiment_configs"

    id = Column(Integer, primary_key=True, index=True)
    num_clients = Column(Integer)
    num_rounds = Column(Integer)
    local_epochs = Column(Integer)
    privacy_budget = Column(Float)
    noise_scale = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    description = Column(String, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)
print(f"Connected to database: {DATABASE_URL}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
