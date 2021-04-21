from gym.envs.registration import register

register(
    id='ProjectAgni-v0',
    entry_point='gym_env.gym_classification:Env4RLClassification',
    kwargs={"train_loader": None, "test_loader": None}
)


def dummy():
    pass
