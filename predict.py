from test import RandomPredictor
from test_umx import UMXPredictor

# This is random predictor which do nothing
random_predictor = RandomPredictor()

# UMX need .cache folder to be present in your submission, check test_umx.py to learn more
umx_predictor = UMXPredictor()


"""
PARTICIPANT_TODO: The implementation you want to submit as your submission
"""
submission = umx_predictor
submission.run()
print("Successfully completed music demixing...")
