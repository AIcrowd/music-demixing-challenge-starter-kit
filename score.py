from evaluator.music_demixing import MusicDemixingPredictor

print("Calculating scores for local run...")
submission = MusicDemixingPredictor()
scores = submission.scoring()
print(scores)
