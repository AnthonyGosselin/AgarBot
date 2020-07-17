import os

# Playback settings settings
LOOP_PLAY = True
DEBUG = False
SAVE_IMAGES = True
SAVE_PROCESSED = False
VERBOSE = False
SAVE_GAME = True

# Bot settings
MOUSE_DIST = 200
DANGER_RANGE = 600
OUTER_RANGE = 400
SPIKES_RANGE = 400
BLOB_FACTOR = 1
OUTER_FACTOR = 1
SPIKES_FACTOR = 1

# Processing settings
BLUR_RAD = 21
SPIKES_BLUR_RAD = 75
BINARY_THRESHOLD = 220
SPIKES_BINARY_THRESHOLD = 190
SPIKES_COLOR = 171
END_SCREEN_COLOR = 127

# Positional settings
SCREEN_START = (0, 71)
SCREEN_END = (1918, 1038)
LEADERBOARD_LB_CORNER = (1637, 347)
SCORE_TR_CORNER = (130, 925)
QUEST_TL_CORNER = (875, 930)
QUEST_BR_CORNER = (1045, 960)
SCREEN_CENTER = (959, 519)
PLAY_BUTTON = (946, 379)
CONTINUE_BUTTON = (957, 465)

# File locations
SCREEN_GRABS_PATH = os.getcwd() + '\\..\\screen_grabs'
SAVED_GAMES_PATH = os.getcwd() + '\\..\\saved_Games'