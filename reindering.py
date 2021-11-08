# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 11:29:55 2021

@author: Francesco
"""
import os
from config.multi_uav_env_2d_config import MULTI_UAV_ENV_2D_CONFIG as conf_1
from gym.envs.classic_control import rendering
img_dir = os.path.join(os.curdir, "images")
uav_img = os.path.join(img_dir, "UAV_Icon.png")


square_len = 40
screen_width = square_len * conf_1["y"]
screen_height = square_len * conf_1["x"]
background_width = screen_width
background_height = screen_height

viewer = rendering.Viewer(screen_width, screen_height)
l, r, t, b = (
    -background_width / 2,
    background_width / 2,
    background_height / 2,
    -background_height / 2,
)
background = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])


backtrans = rendering.Transform(
    translation=(screen_width / 2, screen_height / 2)
)
background.add_attr(backtrans)
background.set_color(1.0, 1.0, 1.0)
viewer.add_geom(background)
for i in range(conf_1["x"]):
    row_line = rendering.Line(start=(0, i*square_len),
                              end=(screen_width, i*square_len))
    viewer.add_geom(row_line)
for j in range(conf_1["y"]):
    col_line = rendering.Line(start=(j*square_len, 0),
                              end=(j*square_len, screen_height))
    viewer.add_geom(col_line)
    # for j in conf_1["y"]:
    #     col_line = rendering.Line()

# new elements are added in the bottom left corner. To center them, translate
# by (screen_width/2, screen_height/2)
uav = rendering.Image(uav_img, square_len, square_len)
uav_transf = rendering.Transform(translation=(screen_width/2-square_len/2,
                                              screen_height/2-square_len/2))
uav.add_attr(uav_transf)

viewer.add_geom(uav)

for i in range(500):
    if i % 100 == 0:
        print(i)
    viewer.render()

viewer.close()
