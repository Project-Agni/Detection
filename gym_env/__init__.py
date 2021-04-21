from gym.envs.registration import register

register(
    id='ProjectAgni-v0',
    entry_point='gym_env.gym_classification:Env4RLClassification',
)


def dummy():
    pass
