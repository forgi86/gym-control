import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='FirstOrderControl-v0',
    entry_point='gym_control.envs:FirstOrderEnv',
)

register(
    id='CartPoleControl-v0',
    entry_point='gym_control.envs:CartPoleEnv',
)