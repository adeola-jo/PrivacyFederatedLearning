
"""
Database utilities for storing and retrieving experimental results.
"""

import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Create a base class for declarative models
Base = declarative_base()

class TrainingRound(Base):
    """Model for storing training round results."""
    __tablename__ = 'training_rounds'
    
    id = Column(Integer, primary_key=True)
    round_number = Column(Integer)
    accuracy = Column(Float)
    privacy_loss = Column(Float)
    num_clients = Column(Integer)
    privacy_budget = Column(Float)
    noise_scale = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)

class ExperimentConfig(Base):
    """Model for storing experiment configurations."""
    __tablename__ = 'experiment_configs'
    
    id = Column(Integer, primary_key=True)
    num_clients = Column(Integer)
    num_rounds = Column(Integer)
    local_epochs = Column(Integer)
    privacy_budget = Column(Float)
    noise_scale = Column(Float)
    description = Column(String)
    timestamp = Column(DateTime, default=datetime.now)

# Create database engine
db_file = 'federated_learning.db'
engine = create_engine(f'sqlite:///{db_file}')

# Create all tables if they don't exist
Base.metadata.create_all(engine)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Generator to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_engine():
    """Get the database engine."""
    return engine

def init_db():
    """Initialize database if needed."""
    if not os.path.exists(db_file):
        Base.metadata.create_all(bind=engine)
        print(f"Created database at {db_file}")
    else:
        print(f"Database already exists at {db_file}")

# Initialize the database when the module is imported
init_db()
