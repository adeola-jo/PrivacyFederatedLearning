
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# We'll use a temporary in-memory SQLite database for testing
@pytest.fixture
def setup_test_db():
    """Set up a temporary database for testing"""
    # Create a temporary SQLite database
    temp_db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db_file.close()
    
    # Set the DATABASE_URL environment variable to our temporary file
    old_db_url = os.environ.get('DATABASE_URL', None)
    os.environ['DATABASE_URL'] = f'sqlite:///{temp_db_file.name}'
    
    # Import the database module after setting environment variable
    from database import Base, TrainingRound, ExperimentConfig, get_db
    
    # Create a test engine and tables
    engine = create_engine(os.environ['DATABASE_URL'])
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Return a session factory and cleanup function
    def get_test_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    yield get_test_db
    
    # Cleanup: remove the temporary database file and restore environment
    if old_db_url:
        os.environ['DATABASE_URL'] = old_db_url
    else:
        del os.environ['DATABASE_URL']
    
    try:
        os.unlink(temp_db_file.name)
    except:
        pass

def test_training_round_crud(setup_test_db):
    """Test CRUD operations for TrainingRound model"""
    from database import TrainingRound
    
    # Get a database session
    session_factory = setup_test_db
    db = next(session_factory())
    
    # Create a new training round
    new_round = TrainingRound(
        round_number=1,
        accuracy=85.5,
        privacy_loss=0.25,
        num_clients=5,
        privacy_budget=1.0,
        noise_scale=0.1,
        timestamp=datetime.utcnow()
    )
    
    # Add to database
    db.add(new_round)
    db.commit()
    
    # Read from database
    saved_round = db.query(TrainingRound).filter_by(round_number=1).first()
    assert saved_round is not None, "Failed to save and retrieve TrainingRound"
    assert saved_round.accuracy == 85.5, f"Expected accuracy 85.5, got {saved_round.accuracy}"
    assert saved_round.noise_scale == 0.1, f"Expected noise_scale 0.1, got {saved_round.noise_scale}"
    
    # Update the record
    saved_round.accuracy = 90.0
    db.commit()
    
    # Verify update
    updated_round = db.query(TrainingRound).filter_by(round_number=1).first()
    assert updated_round.accuracy == 90.0, f"Expected updated accuracy 90.0, got {updated_round.accuracy}"
    
    # Delete the record
    db.delete(saved_round)
    db.commit()
    
    # Verify deletion
    deleted_round = db.query(TrainingRound).filter_by(round_number=1).first()
    assert deleted_round is None, "Failed to delete TrainingRound"
    
    db.close()
    print("TrainingRound CRUD test passed!")

def test_experiment_config_crud(setup_test_db):
    """Test CRUD operations for ExperimentConfig model"""
    from database import ExperimentConfig
    
    # Get a database session
    session_factory = setup_test_db
    db = next(session_factory())
    
    # Create a new experiment config
    new_config = ExperimentConfig(
        num_clients=8,
        num_rounds=10,
        local_epochs=2,
        privacy_budget=1.5,
        noise_scale=0.05,
        description="Test experiment",
        timestamp=datetime.utcnow()
    )
    
    # Add to database
    db.add(new_config)
    db.commit()
    
    # Read from database
    saved_config = db.query(ExperimentConfig).filter_by(description="Test experiment").first()
    assert saved_config is not None, "Failed to save and retrieve ExperimentConfig"
    assert saved_config.num_clients == 8, f"Expected num_clients 8, got {saved_config.num_clients}"
    assert saved_config.privacy_budget == 1.5, f"Expected privacy_budget 1.5, got {saved_config.privacy_budget}"
    
    # Update the record
    saved_config.num_rounds = 15
    db.commit()
    
    # Verify update
    updated_config = db.query(ExperimentConfig).filter_by(description="Test experiment").first()
    assert updated_config.num_rounds == 15, f"Expected updated num_rounds 15, got {updated_config.num_rounds}"
    
    # Delete the record
    db.delete(saved_config)
    db.commit()
    
    # Verify deletion
    deleted_config = db.query(ExperimentConfig).filter_by(description="Test experiment").first()
    assert deleted_config is None, "Failed to delete ExperimentConfig"
    
    db.close()
    print("ExperimentConfig CRUD test passed!")

if __name__ == "__main__":
    print("Testing database operations...")
    
    # Create a pytest-compatible setup for running the tests
    pytest.main(["-xvs", __file__])
