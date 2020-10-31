import json

from rl_utils.dotdic import DotDic
from env.grid_game_flat import GridGame

opt = DotDic(json.loads(open('grid.json', 'r').read()))

g = GridGame(opt, (5, 5))
g.show(vid=False)
