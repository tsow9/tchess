import os
import json
import matplotlib.pyplot as plt


def save_latest_graphs_from_logs(log_dir, out_dir, title="Title", x_label="x", y_label="y", max_files=5):
  os.makedirs(out_dir, exist_ok=True)

  # ログファイル一覧を時系列順に取得
  logs = sorted(
    [f for f in os.listdir(log_dir) if f.endswith(".json")],
    key=lambda x: os.path.getmtime(os.path.join(log_dir, x)),
    reverse=True
  )[:max_files]

  for log_file in logs:
    with open(os.path.join(log_dir, log_file), "r") as f:
      loss_history = json.load(f)

    # グラフ描画
    plt.figure()
    plt.plot(loss_history, marker="o")
    plt.title(f"{title} - {log_file}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

    # 保存
    name = log_file.replace("log_", "graph_").replace(".json", ".png")
    plt.savefig(os.path.join(out_dir, name))
    plt.close()