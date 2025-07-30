#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from collections import defaultdict
import math

def parse_args():
    parser = argparse.ArgumentParser(
        description="lerobot_data の episodes_stats.jsonl を読み込んで解析します"
    )
    parser.add_argument(
        "data_root",
        help="lerobot_data のルートディレクトリパス (例: /home/kanazawa/.imitator/.../data/lerobot_data/)"
    )
    return parser.parse_args()

def load_stats(stats_path: Path):
    """JSONL ファイルを読み込み、Python の辞書リストとして返す。"""
    stats = []
    with stats_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                stats.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[警告] 行 {i} の JSON デコードエラー: {e}")
    return stats

def summarize_numeric_fields(records):
    """
    数値フィールドごとに、全エピソードにまたがって
    min, max, mean を計算する（存在するキーのみ対象）。
    """
    sums = defaultdict(float)
    counts = defaultdict(int)
    mins = {}
    maxs = {}

    for rec in records:
        for k, v in rec.items():
            if isinstance(v, (int, float)):
                sums[k] += v
                counts[k] += 1
                mins[k] = v if k not in mins else min(mins[k], v)
                maxs[k] = v if k not in maxs else max(maxs[k], v)

    summary = {}
    for k in sums:
        summary[k] = {
            "min": mins[k],
            "max": maxs[k],
            "mean": sums[k] / counts[k] if counts[k] else None,
        }
    return summary

def main():
    args = parse_args()
    stats_path = Path(args.data_root) / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        print(f"[ERROR] ファイルが見つかりません: {stats_path}")
        return

    records = load_stats(stats_path)
    print(f"読み込んだエピソード数: {len(records)}\n")

    if records:
        # import ipdb
        # ipdb.set_trace()

        cnt = 0

        dim = len(records[0]['stats']['observation.state']['min'])
        global_s_min = [ math.inf ] * dim
        global_s_max = [ -math.inf ] * dim
        global_a_min = [ math.inf ] * dim
        global_a_max = [ -math.inf ] * dim

        for record in records:
            print(cnt)
            s_min = record['stats']['observation.state']['min']
            s_max = record['stats']['observation.state']['max']
            a_min = record['stats']['action']['min']
            a_max = record['stats']['action']['max']
            # 各次元ごとに比較更新
            for i in range(dim):
                if s_min[i] < global_s_min[i]:
                    global_s_min[i] = s_min[i]
                if s_max[i] > global_s_max[i]:
                    global_s_max[i] = s_max[i]
            for i in range(dim):
                if a_min[i] < global_a_min[i]:
                    global_a_min[i] = a_min[i]
                if a_max[i] > global_a_max[i]:
                    global_a_max[i] = a_max[i]
            cnt += 1
            # print("s_min", s_min)
            # print("s_max", s_max)
            # print("a_min", a_min)
            # print("a_max", a_max)
            # print("global_s_min", global_s_min)
            # print("global_s_max", global_s_max)
            # print("global_a_min", global_a_min)
            # print("global_a_max", global_a_max)

        print("=== 数値フィールドの全エピソード集計 ===")
        print("global_s_min", global_s_min)
        print("global_s_max", global_s_max)
        print("global_a_min", global_a_min)
        print("global_a_max", global_a_max)

if __name__ == "__main__":
    main()
