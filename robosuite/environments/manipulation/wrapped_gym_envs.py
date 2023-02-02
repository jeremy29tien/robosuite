from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper
import robosuite as suite


class LiftJaco(GymWrapper):
    def __init__(self, keys=None):
        env = suite.make(
            "Lift",
            robots="Jaco",  # use Jaco robot -- TODO: can we leave this argument out?
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=False,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards -- TODO: change this?
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
            controller_configs=load_controller_config(default_controller="OSC_POSITION"),
        )
        # Run super method
        super().__init__(env=env, keys=keys)
