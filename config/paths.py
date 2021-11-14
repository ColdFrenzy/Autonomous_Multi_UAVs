"""paths.py file.

It contains all the main files and directories paths needed in this project
"""
import os
# by defining the project dir like this, it will always be the same when
# other files import this file
project_dir = os.path.dirname(os.path.dirname(__file__))


IMG_DIR = os.path.join(project_dir, "images")
ENV_DIR = os.path.join(project_dir, "gym_environment")

# =============================================================================
# UAVS
# =============================================================================

UAV_LOW_IMG = os.path.join(IMG_DIR, "UAV_low_charge.png")
UAV_MID_IMG = os.path.join(IMG_DIR, "UAV_medium_charge.png")
UAV_HIGH_IMG = os.path.join(IMG_DIR, "UAV_full_charge.png")

# =============================================================================
# OBSTACLES
# =============================================================================
TREE1 = os.path.join(IMG_DIR, "tree1.png")
TREE2 = os.path.join(IMG_DIR, "tree2.png")
HOUSE1 = os.path.join(IMG_DIR, "house1_top_view.png")
HOUSE2 = os.path.join(IMG_DIR, "house2_top_view.png")

OBSTACLES = [TREE1, TREE2, HOUSE1, HOUSE2]


print(IMG_DIR)
