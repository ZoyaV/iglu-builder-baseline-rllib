import yaml
import ray
import os
import gym
import iglu
import sys
import wandb
import logging
from collections import defaultdict
from filelock import FileLock
from iglu.tasks import RandomTasks, TaskSet
from ray.rllib.execution.common import AGENT_STEPS_SAMPLED_COUNTER, _get_shared_metrics
import ray.tune.ray_trial_executor as trial_vars
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
from argparse import ArgumentParser
from wrappers import \
    SelectAndPlace, \
    Discretization, \
    flat_action_space, \
    SizeReward, \
    TimeLimit, \
    VectorObservationWrapper, \
    VisualObservationWrapper, \
    VisualOneBlockObservationWrapper, \
    CompleteReward, \
    CompleteScold, \
    Closeness, \
    SweeperReward, \
    Logger
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
import datetime
from model import GridBaselineModel, PovBaselineModel, LargePovBaselineModel
from model_with_target import PovBaselineModelTarget, LargePovBaselineModelTarget
from custom_tasks import make_3d_cube,make_plane
from one_block_wrapppers import SizeLongReward, RandomTarget

logging.basicConfig(stream=sys.stdout)

start = datetime.datetime.now()

print(trial_vars.DEFAULT_GET_TIMEOUT)
trial_vars.DEFAULT_GET_TIMEOUT = 60.0
print(trial_vars.DEFAULT_GET_TIMEOUT)

os.environ['WANDB_APIKEY'] = "e5ac79d62944a4e1910f83da82ae92c37b09ecdf"


def evaluate_separately(trainer, eval_workers):
    # w = next(iter(eval_workers.remote_workers()))
    # env_ids = ray.get(w.foreach_env.remote(lambda env: list(env.tasks.preset.keys())))[0]
    # print(f'env id: {env_ids}')
    # i = 0
    # all_episodes = []
    # while i < len(env_ids):
    #     for w in eval_workers.remote_workers():
    #         w.foreach_env.remote(lambda env: env.set_task(env_ids[i]))
    #         i += 1
    #     ray.get([w.sample.remote() for w in eval_workers.remote_workers()])
    #     episodes, _ = collect_episodes(
    #         remote_workers=eval_workers.remote_workers(), timeout_seconds=99999)
    #     all_episodes += episodes
    # metrics = summarize_episodes(episodes)
    # for eid, ep in zip(env_ids, all_episodes):
    #     metrics[f'env_{eid}_reward'] = ep.episode_reward
    metrics = {}
    end = datetime.datetime.now()
    METRIX = _get_shared_metrics()
    total_ts = METRIX.counters.get(AGENT_STEPS_SAMPLED_COUNTER, 0)
    time_spend = (end - start).total_seconds()
    metrics['FPS'] = total_ts / time_spend
    return metrics


def build_env(env_config=None, env_factory=None):
    """

    Args:
        env_config (dict): a dictionary with following keys:
            * action_space :: human-level | discrete | continuous
            * visual :: (bool) whether to expose only visual observation
            * size_reward :: (bool) whether to use reward for increasing size, otherwise default
            * task_mode :: possible values are: 'one_task', 'many_tasks', 'random_tasks'
                if task_mode is one_task -> string with task id
                if task_mode is many_tasks -> list of task ids
                if task_mode is random_tasks -> ignored
            * task_id :: (str or list[str]) task id list of task ids
            * random_tasks :: specification for the random tasks generator. for details,
                see the documentation of iglu.tasks.RandomTasks
        env_factory (callable, optional): function that returns a env instance

    """
    import iglu
    from iglu.tasks import TaskSet
    if env_config is None:
        env_config = defaultdict(lambda: defaultdict(dict))
    if env_factory is None:
        env = gym.make('IGLUSilentBuilder-v0', max_steps=5000)
        if env_config['task_mode'] == 'one_task':
            env.update_taskset(TaskSet(preset=[env_config['task_id']]))
            env.set_task(env_config['task_id'])
        elif env_config['task_mode'] == 'many_tasks':
            env.update_taskset(TaskSet(preset=env_config['task_id']))
        elif env_config['task_mode'] == 'custom_task':
            #env.update_taskset(make_3d_cube())
            env.update_taskset(make_plane())
        elif env_config['task_mode'] == 'random_task':
            env.update_taskset(RandomTasks(
                max_blocks=env_config['random_tasks'].get('max_blocks', 1),
                height_levels=env_config['random_tasks'].get('height_levels', 1),
                allow_float=env_config['random_tasks'].get('allow_float', False),
                max_dist=env_config['random_tasks'].get('max_dist', 1),
                num_colors=env_config['random_tasks'].get('num_colors', 1),
                max_cache=env_config['random_tasks'].get('max_cache', 10),
            ))

    else:
        env = env_factory()
    # env = Logger(env)
    env = SelectAndPlace(env)
    env = Discretization(env, flat_action_space(env_config['action_space']))
    # visual - pov + inventory + compass + target grid;
    # vector: grid + position + inventory + target grid
    if env_config['visual']:
        if env_config['visual_type'] == 'one_block':
            env = VisualOneBlockObservationWrapper(env)
        elif env_config['visual_type'] == 'target_grid':
            env = VisualObservationWrapper(env, True)
        else:
            env = VisualObservationWrapper(env)
    else:
        env = VectorObservationWrapper(env)
    if env_config.get('size_reward', False):
        env = SizeReward(env)
    if env_config.get('success_rate', False):
        env = CompleteReward(env)
        pass
    env = TimeLimit(env, limit=env_config['time_limit'])
    rand_init = True if 'random_target' not in env_config else env_config['random_target']
    cs_init = True if 'complete_scold' not in env_config else env_config['complete_scold']  # CompleteScold
    if cs_init:
        print("\n Complete Scold \n")
        env = Closeness(env)
        env = SweeperReward(env)

    if rand_init:
        print("\n RAND TARGET \n")
        env = RandomTarget(env)

    return env


def register_models():
    ModelCatalog.register_custom_model(
        "grid_baseline_model", GridBaselineModel)
    ModelCatalog.register_custom_model(
        "pov_baseline_model", PovBaselineModel)
    ModelCatalog.register_custom_model(
        "large_pov_baseline_model", LargePovBaselineModel)
    ModelCatalog.register_custom_model(
        "pov_baseline_target_model", PovBaselineModelTarget)
    ModelCatalog.register_custom_model(
        "large_pov_baseline_target_model", LargePovBaselineModelTarget)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', type=str, help='file')
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--wdb', action='store_true', default=True)
    parser.add_argument('--rnd_goal', action='store_true', default=True)
    parser.add_argument('--color_free', action='store_true', default=False)
    parser.add_argument('--complete_scold', action='store_true', default=False)  # complete scold
    args = parser.parse_args()
    if args.local:
        ray.init(local_mode=True)
    tune.register_env('IGLUSilentBuilder-v0', build_env)
    register_models()

    with open(args.f) as f:
        config = yaml.load(f)
    for key in config:
        if args.wdb:
            config[key]['config']['logger_config'] = {}
            config[key]['config']['logger_config']['wandb'] = {
                "api_key": os.environ.get('WANDB_APIKEY'),
                "project": key,
                "log_config": False
            }
        config[key]['config']['env'] = config[key]['env']
        run = config[key]['run']
        print(config)
        del config[key]['env'], config[key]['run']
        config[key]['config']['custom_eval_function'] = evaluate_separately

        if args.rnd_goal:
            config[key]['config']['env_config']['random_target'] = True
        else:
            config[key]['config']['env_config']['random_target'] = False

        if args.complete_scold:
            config[key]['config']['env_config']['complete_scold'] = True
        else:
            config[key]['config']['env_config']['complete_scold'] = False

        if args.color_free:
            config[key]['config']['env_config']['color_free'] = True
        else:
            config[key]['config']['env_config']['color_free'] = False

        if args.local:
            config[key]['config']['num_workers'] = 1
            config[key]['stop']['timesteps_total'] = 3000
            config[key]['config']['timesteps_per_iteration'] = 100
            # config[key]['config']['learning_starts'] = 0
            # if args.wdb:
            #     del config[key]['config']['logger_config']['wandb']
        if args.wdb:
            loggers = DEFAULT_LOGGERS + (WandbLogger,)
        else:
            loggers = DEFAULT_LOGGERS

        tune.run(run, **config[key], loggers=loggers, local_dir="./experiment")
