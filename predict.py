from test import CopyPredictor, ScaledMixturePredictor
from test_lasaft import LaSAFTPredictor

# # Predictor which does nothing
# copy_predictor = CopyPredictor()
#
# # Predictor which uses 1/4*mixture as separations
# scaledmixture_predictor = ScaledMixturePredictor()
#
# # UMX needs `models` folder to be present in your submission, check test_umx.py to learn more
# umx_predictor = UMXPredictor()
#
# # X-UMX needs `models` folder to be present in your submission, check test_xumx.py to learn more
# xumx_predictor = XUMXPredictor()
from test_lightsaft import LightSAFTPredictor

lasaft_predictor = LightSAFTPredictor()
"""
PARTICIPANT_TODO: The implementation you want to submit as your submission
"""
import time

submission = lasaft_predictor

print("start")

start = time.time()
submission.run()
end = time.time()
print(end-start)
print("Successfully completed music demixing...")
