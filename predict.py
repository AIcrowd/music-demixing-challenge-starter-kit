from test import RandomPredictor
from test_umx import UMXPredictor

"""
The implementation you want to submit as your submission
"""
# This is random predictor which do nothing
# submission = RandomPredictor()

# UMX need .cache folder to be present in your submission, check test_umx.py to learn more
submission = UMXPredictor()
submission.run()
print("Successfully completed music demixing...")
