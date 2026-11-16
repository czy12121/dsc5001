"""Create the formal DSC5001 result figures from saved held-out predictions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
TEST = ROOT / "results/formal/test"
FIGURES = ROOT / "results/formal/figures"
KITTEN = "Cat <3 mo"
TEAL = "#117D73"
ORANGE = "#AB5A14"
NAVY = "#254A66"
SLATE = "#596D85"
PURPLE = "#7962A5"
MODELS = {
    "historical": "历史领养率",
    "logit_bands": "逻辑回归",
    "bagging_bands": "Bagging",
    "mlp_bands": "小型 MLP",
    "bayes_bands": "朴素贝叶斯",
    "logit_cont": "逻辑回归",
    "bagging_cont": "Bagging",
    "mlp_cont": "小型 MLP",
    "tree_cont": "单棵决策树",
    "forest_cont": "随机森林",
    "logit_ab": "改进逻辑回归",
}


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=180, facecolor="white")
    plt.close(fig)


def model_comparisons(metrics):
    overall = metrics.loc[
        metrics["cohort"].eq("primary") & metrics["group_type"].eq("all")
    ].set_index("run")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained", sharex=True)
    panels = [
        ("统一年龄档", ["historical", "logit_bands", "bagging_bands", "mlp_bands", "bayes_bands"]),
        ("连续年龄与树模型对照", ["logit_cont", "tree_cont", "bagging_cont", "forest_cont", "mlp_cont"]),
    ]
    for ax, (title, runs) in zip(axes, panels):
        values = overall.loc[runs, "ap"]
        colors = [TEAL if run in {"bagging_cont", "mlp_cont"} else SLATE for run in runs]
        y = np.arange(len(runs))
        bars = ax.barh(y, values, height=0.58, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=10)
        ax.set_yticks(y, [MODELS[run] for run in runs])
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.set_xlim(0, 0.83)
        ax.axvline(overall.loc["historical", "ap"], color="#999AA2", linestyle="--", linewidth=1)
        ax.set_xlabel("平均精确率 AP")
    fig.suptitle(f"30 天领养预测的留出测试 · {int(overall.iloc[0]['n']):,} 条入所记录",
                 fontsize=16, fontweight="bold")
    fig.supxlabel("虚线表示历史比例基准的 AP", fontsize=10)
    save(fig, "留出测试模型对比")


def ablation_and_trees(metrics, differences):
    overall = metrics.loc[
        metrics["cohort"].eq("primary") & metrics["group_type"].eq("all")
    ].set_index("run")
    contrasts = [
        ("logit_a", "logit_base", "给基础逻辑回归加 A"),
        ("logit_ab", "logit_cont", "已有 B 时加 A"),
        ("logit_cont", "logit_base", "给基础逻辑回归加 B"),
        ("logit_ab", "logit_a", "已有 A 时加 B"),
        ("bagging_ab", "bagging_cont", "给 Bagging 加 A"),
        ("bagging_cont", "tree_cont", "单棵树改为 Bagging"),
        ("forest_cont", "bagging_cont", "Bagging 改为随机森林"),
    ]
    spec = differences.set_index(["run", "reference"])
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.8, 4.8), layout="constrained", gridspec_kw={"width_ratios": [1, 1.5]}
    )
    runs = ["logit_base", "logit_a", "logit_cont", "logit_ab"]
    y = np.arange(len(runs))
    bars = left.barh(y, overall.loc[runs, "ap"], color=TEAL, height=0.52)
    left.bar_label(bars, fmt="%.3f", padding=5, fontsize=10)
    left.set_yticks(y, ["基础", "基础 + A", "基础 + B", "基础 + A + B"])
    left.invert_yaxis()
    left.set_xlim(0, 0.78)
    left.set_xlabel("平均精确率 AP")
    left.set_title("逻辑回归四格对照", loc="left")
    changes = spec.loc[[(run, reference) for run, reference, _ in contrasts]]
    y = np.arange(len(contrasts))
    d = changes["delta_ap"].to_numpy()
    right.errorbar(d, y, xerr=np.array([d - changes["ci_low"], changes["ci_high"] - d]).clip(min=0),
                   fmt="o", capsize=4, color=NAVY, linewidth=1.6)
    right.axvline(0, linewidth=1, color="#8593A4", linestyle="--")
    right.set_yticks(y, [label for _, _, label in contrasts])
    right.invert_yaxis()
    right.set_title("AP 变化及配对 bootstrap 95% 区间", loc="left")
    right.set_xlabel("前述调整带来的 AP 差")
    fig.suptitle("年龄表达与树模型的受控对照", fontsize=16, fontweight="bold")
    fig.supxlabel("A 为年龄档及猫狗交互，B 为入所原因及状况", fontsize=10)
    save(fig, "年龄消融与树模型对照")


def kitten_errors(metrics, outcomes):
    kittens = metrics.loc[
        metrics["cohort"].eq("primary") & metrics["group_type"].eq("species_age_band")
        & metrics["group"].eq(KITTEN)
    ].set_index("run")
    runs = ["logit_ab", "bagging_cont", "mlp_cont", "forest_cont", "bayes_bands"]
    fp = outcomes.loc[(outcomes["run"] == "bagging_cont") & (outcomes["error"] == "false_positive_outcome")]
    counts = fp.set_index("category")["n"]
    categories = [
        ("Adoption on days 31–60", "第 31 至 60 天领养"),
        ("Adoption after day 60", "超过 60 天才领养"),
        ("Transfer", "转移到合作机构等"),
        ("No unambiguous exit", "无明确离所配对"),
    ]
    selected = [int(counts.get(key, 0)) for key, _ in categories]
    rest = int(counts.sum()) - sum(selected)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.1, 4.7), layout="constrained")
    y = np.arange(len(runs))
    for field, color, offset, label in (("fp", ORANGE, -0.16, "误报"), ("fn", NAVY, 0.16, "漏报")):
        bars = left.barh(y + offset, kittens.loc[runs, field], height=0.28, color=color, label=label)
        left.bar_label(bars, padding=3, fontsize=9)
    left.set_yticks(y, [MODELS[run] for run in runs])
    left.invert_yaxis()
    left.set_title("各模型的误报与漏报", loc="left")
    left.set_xlim(0, 1.15 * max(kittens.loc[runs, ["fp", "fn"]].max().max(), 1))
    left.legend(frameon=False, loc="lower right")
    left.set_xlabel("入所记录数")
    labels = [label for _, label in categories] + ["其他去向"]
    bars = right.barh(labels, selected + [rest], height=0.55, color=[TEAL, NAVY, SLATE, PURPLE, "#9B8A72"])
    right.bar_label(bars, padding=4, fontsize=10)
    right.set_xlim(0, 1.20 * max(selected + [rest, 1]))
    right.invert_yaxis()
    right.set_title(f"Bagging 误报记录的实际去向（{int(counts.sum())} 条）", loc="left")
    right.set_xlabel("入所记录数")
    fig.suptitle(f"未满 3 月龄猫 · {int(kittens.iloc[0]['n']):,} 条测试记录 · 0.50 阈值",
                 fontsize=16, fontweight="bold")
    save(fig, "幼猫误报漏报与去向")


def calibration_plot():
    predictions = pd.read_csv(TEST / "predictions_primary_30d.csv")
    fig, ax = plt.subplots(figsize=(6.6, 4.9), layout="constrained")
    ax.plot([0, 1], [0, 1], color="#8C9AA6", linestyle="--", label="预测概率等于实际比例")
    for run, color in (("logit_ab", PURPLE), ("bagging_cont", TEAL), ("mlp_cont", NAVY), ("bayes_bands", ORANGE)):
        score = predictions[run]
        table = predictions.assign(bin=pd.cut(score, np.linspace(0, 1, 9), include_lowest=True))
        points = table.groupby("bin", observed=True).agg(
            average_p=(run, "mean"), actual_rate=("adopted_30d", "mean"), n=("adopted_30d", "size"),
        )
        points = points.loc[points["n"].ge(20)]
        ax.plot(points["average_p"], points["actual_rate"], "o-", label=MODELS[run], color=color, linewidth=1.6)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="分箱内平均预测概率", ylabel="分箱内实际 30 天领养比例")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("30 天领养概率的校准 · 留出测试", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.supxlabel("每个点对应一个概率区间，显示至少 20 条记录的区间", fontsize=9)
    save(fig, "留出测试概率校准")


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#C5CED8",
        "axes.axisbelow": True, "axes.grid": True, "grid.alpha": 0.22,
        "text.color": "#203043", "axes.labelcolor": "#203043",
    })
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(TEST / "metrics.csv")
    model_comparisons(metrics)
    ablation_and_trees(metrics, pd.read_csv(TEST / "ap_comparisons.csv"))
    kitten_errors(metrics, pd.read_csv(TEST / "kitten_error_outcomes.csv"))
    calibration_plot()


if __name__ == "__main__":
    main()
