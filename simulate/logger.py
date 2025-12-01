from collections import defaultdict
import numpy as np
import pandas as pd
import os
import json
from typing import Dict, Any, List, Optional
import seaborn as sns
import matplotlib.pyplot as plt

from config import *


class EpisodeLogger():
    def __init__(self):
        self.player = []
        self.time_spent = []
        self.scores = []
        self.actions = []
        self.policy_logs = []

    def log(self, player, time_spent, scores, action, policy_log):
        self.player.append(player)
        self.time_spent.append(time_spent)
        self.scores.append(scores)
        self.actions.append(action)
        self.policy_logs.append(policy_log)

    def get_default_log(self, winner):
        return {
            'player': self.player,
            'time_spent': self.time_spent,
            'scores': self.scores,
            'winner': winner,
        }
    
    def get_action_log(self):
        return {'Action_log': f'{N_BOX}, {N_BOX}, ' + (", ".join(str(x) for a in self.actions for x in a))}
    
    def get_policy_log(self):
        return {
            'player': self.player,
            'logs': self.policy_logs
        }
    
class MultiEpisodeLogger():
    def __init__(self, save_path):
        self.default_logs = []
        self.action_logs = []
        self.policy_logs = []

        self.save_path = save_path
    
    def log(self, defualt_log, actions_log, policy_log, episode_id, first_player_is_p0):

        if not first_player_is_p0:
            defualt_log['player'] = [1 - p for p in defualt_log['player']]
            defualt_log['winner'] = 1 - defualt_log['winner']
            defualt_log['scores'] = [score[::-1] for score in defualt_log['scores']]
            policy_log['player'] = [1 - p for p in policy_log['player']]

        self.default_logs.append({
            'episode_id': episode_id,
            'default_log': defualt_log
        })

        self.action_logs.append({
            'episode_id': episode_id,
            'action_log': actions_log
        })

        self.policy_logs.append({
            'episode_id': episode_id,
            'policy_log': policy_log
        })

        self.save_log_jsons()

    def save_log_jsons(self):
        os.makedirs(self.save_path, exist_ok=True)
        
        with open(os.path.join(self.save_path, 'default_logs.json'), 'w') as f:
            json.dump(self.default_logs, f, indent=4)

        with open(os.path.join(self.save_path, 'action_logs.json'), 'w') as f:
            json.dump(self.action_logs, f, indent=4)

        with open(os.path.join(self.save_path, 'policy_logs.json'), 'w') as f:
            json.dump(self.policy_logs, f, indent=4)


    def log_stats(self):
    
        self.save_log_jsons()

        basic_eval_stat_df = _calc_basic_stats_from_default_logs(default_logs=self.default_logs)
        basic_eval_stat_df.to_csv(os.path.join(self.save_path, 'Evaluation_stats.csv'))
        
        basic_policy_stat_df = _calc_basic_stats_from_policy_logs(policy_logs=self.policy_logs)
        basic_policy_stat_df.to_csv(os.path.join(self.save_path, 'Policy_stats.csv'))
        
        print(basic_eval_stat_df)
        print(basic_policy_stat_df)
        _search_node_plot(policy_logs=self.policy_logs, save_path=self.save_path)

        print(f"Simulation results saved to {self.save_path}")

def _calc_basic_stats_from_default_logs(default_logs):
    """
    default_logs: simulate_multi_episode에서 생성된 기본 Evaluation 로그 리스트.

    반환: 한 줄짜리 pandas.DataFrame
      - n_episodes: 총 판수
      - n_p0_wins, n_p1_wins
      - p0_win_rate, p1_win_rate
      - p0_score_*: describe() 결과(prefix 붙임)
      - p1_score_*: describe() 결과(prefix 붙임)
      - p0_time_per_game, p1_time_per_game
      - p0_time_per_turn, p1_time_per_turn
    """

    episode_rows = []

    for ep in default_logs:
        ep_id = ep.get("episode_id", None)
        log = ep.get("default_log")

        winner = log["winner"]
        scores_list = log["scores"]
        final_p0, final_p1 = scores_list[-1]

        players = log["player"]
        times = log["time_spent"]

        p0_total_time = sum(t for t, p in zip(times, players) if p == 0)
        p1_total_time = sum(t for t, p in zip(times, players) if p == 1)

        p0_n_turns = sum(1 for p in players if p == 0)
        p1_n_turns = sum(1 for p in players if p == 1)

        episode_rows.append({
            "episode_id": ep_id,
            "winner": winner,
            "p0_score": final_p0,
            "p1_score": final_p1,
            "p0_total_time": p0_total_time,
            "p1_total_time": p1_total_time,
            "p0_n_turns": p0_n_turns,
            "p1_n_turns": p1_n_turns,
        })

    df = pd.DataFrame(episode_rows)

    if len(df) == 0:
        raise ValueError("default_logs가 비어 있습니다.")

    # 기본 승수/승률
    n_episodes = len(df)
    n_p0_wins = (df["winner"] == 0).sum()
    n_p1_wins = (df["winner"] == 1).sum()

    p0_win_rate = n_p0_wins / n_episodes
    p1_win_rate = n_p1_wins / n_episodes

    # 점수 describe
    p0_score_stats = df["p0_score"].describe().add_prefix("p0_score_")
    p1_score_stats = df["p1_score"].describe().add_prefix("p1_score_")

    # 시간 통계
    p0_time_per_game = df["p0_total_time"].mean()
    p1_time_per_game = df["p1_total_time"].mean()

    p0_total_time_all = df["p0_total_time"].sum()
    p1_total_time_all = df["p1_total_time"].sum()

    p0_total_turns_all = df["p0_n_turns"].sum()
    p1_total_turns_all = df["p1_n_turns"].sum()

    p0_time_per_turn = p0_total_time_all / p0_total_turns_all if p0_total_turns_all > 0 else np.nan
    p1_time_per_turn = p1_total_time_all / p1_total_turns_all if p1_total_turns_all > 0 else np.nan

    summary = {
        "n_episodes": n_episodes,
        "n_p0_wins": n_p0_wins,
        "n_p1_wins": n_p1_wins,
        "p0_win_rate": p0_win_rate,
        "p1_win_rate": p1_win_rate,
        "p0_time_per_game": p0_time_per_game,
        "p1_time_per_game": p1_time_per_game,
        "p0_time_per_turn": p0_time_per_turn,
        "p1_time_per_turn": p1_time_per_turn,
    }

    summary_df = pd.DataFrame([summary])

    p0_score_df = p0_score_stats.to_frame().T.reset_index(drop=True)
    p1_score_df = p1_score_stats.to_frame().T.reset_index(drop=True)

    stats_df = pd.concat([summary_df, p0_score_df, p1_score_df], axis=1)
    return stats_df

def _calc_basic_stats_from_policy_logs(policy_logs):
    """
    policy_logs: simulate_multi_episode에서 생성된 policy 로그 리스트.
    player별 key 통계를 계산하여 DataFrame 반환.
    """

    per_player_values: Dict[Any, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for ep in policy_logs:
        plog: Optional[Dict[str, Any]] = ep.get("policy_log", {})
        players = plog.get("player", [])
        logs = plog.get("logs", [])

        L = min(len(players), len(logs)) if players else len(logs)

        for t in range(L):
            player_id = players[t] if players else None
            log_entry = logs[t]
            if log_entry is None:
                continue
            for key, val in log_entry.items():
                if val is None or isinstance(val, bool):
                    continue
                if isinstance(val, (int, float)):
                    per_player_values[player_id][key].append(float(val))

    stats_dict = {}

    for player_id, key_dict in per_player_values.items():
        for key, vals in key_dict.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            prefix = 'p0_' if player_id == 0 else 'p1_'
            stats_dict[prefix + key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "count": int(arr.size),
            }

    df_stats = pd.DataFrame.from_dict(stats_dict, orient="index")
    if len(df_stats) > 0:
        df_stats = df_stats[["mean", "std", "min", "max", "count"]]

    return df_stats

def _search_node_plot(policy_logs, save_path):

    def logs_to_long_format(policy_logs):
        rows = []
        for ep_id, ep in enumerate(policy_logs):
            logs = ep["policy_log"]["logs"]
            player = ep["policy_log"]["player"]
            
            for t, log_entry in enumerate(logs):
                if log_entry is None:
                    continue
                for k, val in log_entry.items():
                    if isinstance(val, (int, float)):
                        rows.append({
                            "episode": ep_id,
                            "timestep": t,
                            "player": player[t],
                            "key": k,
                            "value": float(val),
                        })
        return pd.DataFrame(rows)


    df = logs_to_long_format(policy_logs)
    if df.empty:
        return
    
    keys = df["key"].unique()
    print(keys)

    for key in keys:
        if key == 'player':
            continue
        sub = df[df["key"] == key]

        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=sub,
            x="timestep",
            y="value",
            estimator="mean",
            hue="player"
        )
        plt.title(f"Key: {key}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(os.path.join(save_path, key + "_line_plot.jpg"))
        plt.close()


# def save_sim_logs(Evaluation_logs, Actions_logs, Policy_logs, save_path: str):
#     """
#     SimulateMultipleEpisodes의 결과를 지정된 경로에 저장합니다.
    
#     이거보단 나은 저장방식이 있을 것 같은데 Todo에 부치겠습니다.
#     """
#     os.makedirs(save_path, exist_ok=True)

#     with open(os.path.join(save_path, 'Action_logs.json'), 'w') as f:
#         json.dump(Actions_logs, f, indent=4)

#     with open(os.path.join(save_path, 'Evaluation_logs.json'), 'w') as f:
#         json.dump(Evaluation_logs, f, indent=4)

#     with open(os.path.join(save_path, 'Policy_logs.json'), 'w') as f:
#         json.dump(Policy_logs, f, indent=4)

#     basic_eval_stat_df = calc_basic_stats_from_eval_logs(Evaluation_logs=Evaluation_logs)
#     basic_eval_stat_df.to_csv(os.path.join(save_path, 'Evaluation_stats.csv'))
    
#     basic_policy_stat_df = calc_basic_stats_from_policy_logs(Policy_logs=Policy_logs)
#     basic_policy_stat_df.to_csv(os.path.join(save_path, 'Policy_stats.csv'))
    
#     search_node_plot(Policy_logs=Policy_logs, save_path=save_path)
#     print(f"Simulation results saved to {save_path}")