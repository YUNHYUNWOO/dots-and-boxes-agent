from collections import defaultdict
import numpy as np
import pandas as pd
import os
import json
from typing import Dict, Any, List, Optional
import seaborn as sns
import matplotlib.pyplot as plt
def calc_basic_stats_from_eval_logs(Evaluation_logs):
    """
    Evaluation_logs: SimulateMultipleEpisodes에서 생성된 로그 리스트.
    
    반환: 한 줄짜리 pandas.DataFrame
      - n_episodes: 총 판수
      - n_p0_wins, n_p1_wins
      - p0_win_rate, p1_win_rate
      - p0_score_*: describe() 결과(prefix 붙임)
      - p1_score_*: describe() 결과(prefix 붙임)
      - p0_time_per_game, p1_time_per_game: 플레이어별 '판당' 평균 시간
      - p0_time_per_turn, p1_time_per_turn: 플레이어별 '턴당' 평균 시간
    """

    episode_rows = []

    for ep in Evaluation_logs:
        ep_id = ep.get("episode_id", None)
        log = ep.get("Evaluation_log")

        winner = log["winner"]
        scores_list = log["scores"]
        # 각 에피소드의 최종 점수
        final_p0, final_p1 = scores_list[-1]

        players = log["player"]
        times = log["time_spent"]

        # 플레이어별 총 시간 / 턴 수
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
        raise ValueError("Evaluation_logs가 비어 있습니다.")

    # --- 기본 승수/승률 ---
    n_episodes = len(df)
    n_p0_wins = (df["winner"] == 0).sum()
    n_p1_wins = (df["winner"] == 1).sum()

    p0_win_rate = n_p0_wins / n_episodes
    p1_win_rate = n_p1_wins / n_episodes

    # --- 점수 describe ---
    p0_score_stats = df["p0_score"].describe().add_prefix("p0_score_")
    p1_score_stats = df["p1_score"].describe().add_prefix("p1_score_")

    # --- 시간 통계 ---
    # 판당 평균 시간: 각 에피소드의 total_time을 episode 축으로 평균
    p0_time_per_game = df["p0_total_time"].mean()
    p1_time_per_game = df["p1_total_time"].mean()

    # 턴당 평균 시간: 전체 에피소드의 total_time 합 / 전체 턴 수 합
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

    # describe 결과 붙이기
    p0_score_df = p0_score_stats.to_frame().T.reset_index(drop=True)
    p1_score_df = p1_score_stats.to_frame().T.reset_index(drop=True)

    stats_df = pd.concat([summary_df, p0_score_df, p1_score_df], axis=1)

    return stats_df

def calc_basic_stats_from_policy_logs(Policy_logs):
    """
    Policy_Log을 기반으로 player별 key 통계를 계산하여
    player_id -> DataFrame 형태로 반환한다.

    각 DataFrame은 index = key, columns = [mean, std, min, max, count].
    """

    # --- 플레이어별 컨테이너 ---
    # per_player_values[player_id][key] = list of float values
    per_player_values: Dict[Any, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    # --- flatten & collect values ---
    for ep in Policy_logs:
        plog: Optional[Dict[str, Any]] = ep.get("Policy_log", {})
        players = plog.get("player", [])
        logs = plog.get("logs", [])

        L = min(len(players), len(logs)) if players else len(logs)

        for t in range(L):
            player_id = players[t] if players else None
            log_entry = logs[t]
            if log_entry == None:
                continue
            for key, val in log_entry.items():
                if val is None:
                    continue
                if isinstance(val, bool):
                    continue
                if isinstance(val, (int, float)):
                    per_player_values[player_id][key].append(float(val))

    # --- 통계 계산 + DataFrame 생성 ---
    player_stats: Dict[Any, pd.DataFrame] = {}

    stats_dict: Dict[str, Dict[str, float]] = {}

    for player_id, key_dict in per_player_values.items():

        for key, vals in key_dict.items():
            if len(vals) == 0:
                continue
            arr = np.asarray(vals, dtype=float)
            prefix = 'p0_' if player_id == 0 else 'p1_'
            key = prefix + key
            stats_dict[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "count": int(arr.size),
            }

    df_stats = pd.DataFrame.from_dict(stats_dict, orient="index")
    # 컬럼 순서 고정
    if len(df_stats) > 0:
        df_stats = df_stats[["mean", "std", "min", "max", "count"]]

    return df_stats

def search_node_plot(Policy_logs, save_path):
    
    def logs_to_long_format(Policy_logs):
        rows = []
        for ep_id, ep in enumerate(Policy_logs):
            logs = ep["Policy_log"]["logs"]
            player = ep["Policy_log"]['player']

            for t, log_entry in enumerate(logs):
                for k, val in log_entry.items():
                    # player는 스킵 (이미 따로 뽑았음)

                    # numeric만 대상
                    if isinstance(val, (int, float)):
                        rows.append({
                            "episode": ep_id,
                            "timestep": t,
                            "player": player[t],
                            "key": k,
                            "value": float(val),
                        })
            
        return pd.DataFrame(rows)
    
    df = logs_to_long_format(Policy_logs)
    print(df)

    keys = df["key"].unique()
    print(keys)
    print
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
            hue='player'
        )
        plt.title(f"Key: {key}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, key + '_line_plot.jpg'))


def save_sim_logs(Evaluation_logs, Actions_logs, Policy_logs, save_path: str):
    """
    SimulateMultipleEpisodes의 결과를 지정된 경로에 저장합니다.
    
    이거보단 나은 저장방식이 있을 것 같은데 Todo에 부치겠습니다.
    """
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, 'Action_logs.json'), 'w') as f:
        json.dump(Actions_logs, f, indent=4)

    with open(os.path.join(save_path, 'Evaluation_logs.json'), 'w') as f:
        json.dump(Evaluation_logs, f, indent=4)

    with open(os.path.join(save_path, 'Policy_logs.json'), 'w') as f:
        json.dump(Policy_logs, f, indent=4)

    basic_eval_stat_df = calc_basic_stats_from_eval_logs(Evaluation_logs=Evaluation_logs)
    basic_eval_stat_df.to_csv(os.path.join(save_path, 'Evaluation_stats.csv'))
    
    basic_policy_stat_df = calc_basic_stats_from_policy_logs(Policy_logs=Policy_logs)
    basic_policy_stat_df.to_csv(os.path.join(save_path, 'Policy_stats.csv'))
    
    search_node_plot(Policy_logs=Policy_logs, save_path=save_path)
    print(f"Simulation results saved to {save_path}")