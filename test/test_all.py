from src.core.strategy import DummyStrategy
from src.core.models import BaseModel
def test_dummy_strategy_valid_init():
    valid_init = {"take_profit": 1, "stop_loss": 1}
    strategy = DummyStrategy(**valid_init)
    valid_init = {"take_profit": 1, "stop_loss": 1, "model": BaseModel}
    strategy = DummyStrategy(**valid_init)

if __name__=="__main__":
    test_strategy()