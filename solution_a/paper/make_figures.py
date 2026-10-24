"""Render source-backed, 16 cm research figures without changing experiment outputs."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parent
SOLUTION = PAPER.parent
OUTPUT = SOLUTION / "outputs"
FIGURES = PAPER / "figures"
font_candidates = [os.environ.get("MATHORCUP_FONT", ""),
                   "C:/Windows/Fonts/simsun.ttc",
                   "/System/Library/Fonts/Supplemental/Songti.ttc",
                   "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]
CJK_FONT = next((Path(f) for f in font_candidates if f and Path(f).is_file()), None)
if CJK_FONT is None:
    raise FileNotFoundError("Set MATHORCUP_FONT to an installed Chinese font file.")
font_manager.fontManager.addfont(str(CJK_FONT))
FONT = FontProperties(fname=str(CJK_FONT))
BLUE = "#315F82"
GOLD = "#A98039"
GREY = "#A3A8AC"
DARK = "#30383D"
LIGHT = "#E5E8EA"
FAIL = "#A14D40"
WIDTH_CM = 16.0
plt.rcParams.update({
    "font.family": FONT.get_name(), "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.unicode_minus": False,
    "axes.edgecolor": "#69737A", "axes.linewidth": 0.6,
    "text.color": DARK, "axes.labelcolor": DARK,
    "xtick.color": DARK, "ytick.color": DARK,
    "svg.fonttype": "path", "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.spines.top": False,
    "axes.spines.right": False,
})


def read_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT / relative)


def new_figure(height_cm: float):
    return plt.figure(figsize=(WIDTH_CM / 2.54, height_cm / 2.54), dpi=150)


def save(figure, name: str):
    # Apply an explicit font file to every text object, including tick labels.
    for text in figure.findobj(match=Text):
        text.set_fontproperties(FontProperties(
            fname=str(CJK_FONT), size=text.get_fontsize(), weight=text.get_fontweight()))
    figure.savefig(FIGURES / f"{name}.png", dpi=300)
    figure.savefig(FIGURES / f"{name}.svg")
    plt.close(figure)


def grid(axis, direction="x"):
    axis.set_axisbelow(True)
    axis.grid(axis=direction, color=LIGHT, linewidth=0.6)
    axis.tick_params(length=2.5, width=0.6)


def figure1():
    fig = new_figure(11.2)
    ax = fig.add_axes([0.025, 0.025, 0.95, 0.95])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(x, y, w, h, text, highlighted=False):
        ax.add_patch(Rectangle((x, y), w, h,
                              facecolor="#F0F4F7" if highlighted else "white",
                              edgecolor=BLUE if highlighted else "#89959D", linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9,
                linespacing=1.35)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.7, color="#6C7A84"))

    box(0.06, 0.864, 0.88, 0.105,
        "250 款啤酒 / 22 种风格：化学测量、标准化感官评分、校准量与阈值", True)
    box(0.11, 0.717, 0.78, 0.105,
        "按 beer 精确连接；区分相对表示与浓度；在训练折内拟合预处理")
    arrow((0.5, 0.864), (0.5, 0.823))
    centers = [0.163, 0.5, 0.837]
    xs = [0.01, 0.347, 0.684]
    top = ["问题一：感官预测\n231 维化学 → 45 维感官",
           "问题二：化学关联\n231 个节点 / 218 个候选",
           "问题三：生成与迁移\n226 维：69 校准 + 157 相对"]
    mid = ["ExtraTrees 与基线\n训练折补值和缺失指示",
           "Spearman 网络与 Top-3\n200 次配对重抽样",
           "风格自适应分布与嵌套选择\n三风格 900 谱 + 对照 300 谱"]
    bottom = ["五折折外评价\n缺失压力检查",
              "135 个关联位置\n稳定性与风格敏感性",
              "排除全部 Pils：239 款训练\n11 款源谱向 Tripel 迁移"]
    for i, x in enumerate(xs):
        arrow((centers[i], 0.717), (centers[i], 0.657))
        box(x, 0.53, 0.306, 0.125, top[i], True)
        box(x, 0.34, 0.306, 0.13, mid[i])
        box(x, 0.15, 0.306, 0.13, bottom[i])
        arrow((centers[i], 0.53), (centers[i], 0.471))
        arrow((centers[i], 0.34), (centers[i], 0.281))
    box(0.355, 0.004, 0.632, 0.103,
        "迁移检查：十情景浓度保护 + 支持域 + 接近幅度\n双模型筛选 → 预留模型审计；失败源保留")
    arrow((0.837, 0.15), (0.837, 0.108))
    save(fig, "figure1_workflow")


def figure2():
    base = read_csv("q1/metrics_overall.csv").set_index("model")
    robust = read_csv("q1_robust/metrics_overall.csv").set_index("model")
    table = pd.concat([base.loc[["mean", "ridge"]], robust.loc[["ridge_winsor"]],
                       base.loc[["extra_trees"]]])
    fig = new_figure(11.2)
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.17, right=0.975, top=0.93, bottom=0.17,
                        hspace=0.68, wspace=0.65)
    labels = ["均值基线", "Ridge", "裁剪 Ridge", "ExtraTrees"]
    colors = [GREY, GREY, GOLD, BLUE]
    for ax, metric, title in zip(axes.flat[:3], ["rmse", "mae", "macro_r2"],
                                 ["(a) RMSE", "(b) MAE", "(c) 宏平均 $R^2$"]):
        values = table[metric].to_numpy()
        y = np.arange(4)
        ax.barh(y, values, height=0.56, color=colors)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", pad=5)
        if metric == "macro_r2":
            ax.set_xlim(-0.1, 0.31)
            ax.axvline(0, color="#69737A", linewidth=0.65)
        else:
            ax.set_xlim(0, 0.72 if metric == "rmse" else 0.52)
        for pos, value in zip(y, values):
            label_x = 0.014 if value < 0 else value + (0.012 if metric == "rmse" else 0.009)
            ax.text(label_x, pos,
                    f"{value:.3f}", va="center", fontsize=8.5)
        grid(ax)
    ax = axes[1, 1]
    rates = np.array([0, 10, 20])
    values = np.array([base.loc["extra_trees", "rmse"],
                       base.loc["extra_trees_mask10", "rmse"],
                       base.loc["extra_trees_mask20", "rmse"]])
    ax.plot(rates, values, "o-", color=BLUE, markersize=4, linewidth=1.2)
    ax.set_title("(d) ExtraTrees 缺失压力", loc="left", pad=5)
    ax.set(xlim=(-1.5, 21.5), ylim=(0.47, 0.525), xticks=rates,
           xlabel="额外隐藏已观测特征（%）", ylabel="RMSE")
    for rate, value in zip(rates, values):
        ax.annotate(f"{value:.4f}", (rate, value), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8.5)
    grid(ax, "y")
    fig.text(0.5, 0.025,
             f"五折折外评价；20% 缺失压力下 RMSE 上升 {(values[2]/values[0]-1)*100:.2f}%", 
             ha="center", fontsize=8.5)
    save(fig, "figure2_q1")


NETWORK_SHORT_NAMES = {
    "ethyl.tetradecanoate": "十四酸乙酯", "pentyl.octanoate": "辛酸戊酯",
    "X1.2.dihydro.8.hydroxy..2E..linalool": "二氢羟基芳樟醇",
    "X2.methylbutyl.octanoate": "辛酸2-甲基丁酯",
    "X2.phenyl.ethyl.butyrate": "丁酸苯乙酯",
    "X.Z..beta.farnesene": "Z-β-法呢烯",
    "ethyl.pentadecanoate": "十五酸乙酯", "isopentyl.hexanoate": "己酸异戊酯",
}


def spring_layout(adjacency, seed=20261023):
    """Force layout within one connected component."""
    count = len(adjacency)
    pos = np.random.default_rng(seed).normal(size=(count, 2))
    pos /= max(np.linalg.norm(pos, axis=1).max(), 1e-8)
    degree = (adjacency > 0).sum(axis=1)
    repulsion = 1 + 0.12 * (degree[:, None] + degree[None, :])
    scale = np.sqrt(1 / count)
    for step in range(550):
        delta = pos[:, None, :] - pos[None, :, :]
        distance = np.maximum(np.linalg.norm(delta, axis=2), 1e-4)
        forces = ((repulsion * scale**2 / distance**2
                   - adjacency * distance / scale)[:, :, None] * delta).sum(axis=1)
        forces -= 0.06 * pos
        length = np.maximum(np.linalg.norm(forces, axis=1), 1e-8)
        temperature = 0.075 * (1 - step / 550) + 0.001
        pos += forces * (np.minimum(length, temperature) / length)[:, None]
        pos -= pos.mean(axis=0)
    return 0.76 * pos / np.linalg.norm(pos, axis=1).max()


def network_layout(adjacency, seed=20261023):
    """Pack connected-component force layouts without shrinking their edges away."""
    seen, components = set(), []
    for node in range(len(adjacency)):
        if node in seen:
            continue
        pending, component = [node], []
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            pending.extend(np.flatnonzero(adjacency[current] > 0).tolist())
        components.append(sorted(component))
    components.sort(key=lambda component: (-len(component), component[0]))
    candidates = np.array(np.meshgrid(np.arange(-0.9, 0.901, 0.02),
                                      np.arange(-0.9, 0.901, 0.02))).reshape(2, -1).T
    order = np.argsort(np.linalg.norm(candidates, axis=1), kind="stable")
    candidates = candidates[order]
    positions = np.empty((len(adjacency), 2))
    circles = []
    radius_by_count = {15: 0.33, 11: 0.235, 6: 0.16, 4: 0.125, 3: 0.105, 2: 0.078}
    for number, component in enumerate(components):
        radius = radius_by_count[len(component)]
        valid = np.linalg.norm(candidates, axis=1) + radius <= 0.90
        for center, old_radius in circles:
            valid &= np.linalg.norm(candidates - center, axis=1) >= radius + old_radius + 0.022
        if not valid.any():
            raise ValueError("No room to display all connected components")
        center = candidates[np.flatnonzero(valid)[0]]
        local = radius * spring_layout(adjacency[np.ix_(component, component)], seed + number) / 0.76
        # Dense correlated hubs otherwise overlap after component normalization.
        # This collision pass changes display coordinates only.
        for _ in range(260):
            for i in range(len(component)):
                for j in range(i):
                    delta = local[i] - local[j]
                    distance = np.linalg.norm(delta)
                    if distance < 0.115:
                        direction = delta / max(distance, 1e-8)
                        shift = 0.51 * (0.115 - distance) * direction
                        local[i] += shift
                        local[j] -= shift
            norm = np.linalg.norm(local, axis=1)
            local *= np.minimum(1, radius / np.maximum(norm, 1e-8))[:, None]
        positions[component] = center + local
        circles.append((center, radius))
    return positions


def figure3():
    summary = json.loads((OUTPUT / "q2/summary.json").read_text(encoding="utf-8"))
    top3 = read_csv("q2/top3_compounds.csv")
    nodes = read_csv("q2/network_nodes.csv")
    edges = read_csv("q2/network_edges_rho_0.7.csv")
    active = nodes[nodes["degree_0.7"].gt(0)].copy().set_index("field")
    isolated = nodes[nodes["degree_0.7"].eq(0)]
    assert len(nodes) == 231 and len(active) == 81 and len(isolated) == 150
    assert len(edges) == 109 and edges.sign.eq(1).all()
    lookup = {field: i for i, field in enumerate(active.index)}
    adjacency = np.zeros((len(active), len(active)))
    for row in edges.itertuples():
        i, j = lookup[row.source], lookup[row.target]
        adjacency[i, j] = adjacency[j, i] = row.absolute_weight
    np.testing.assert_array_equal((adjacency > 0).sum(axis=1), active["degree_0.7"])
    positions = network_layout(adjacency)
    fig = new_figure(11.8)
    network = fig.add_axes([0.015, 0.28, 0.605, 0.65])
    network.set(xlim=(-1.19, 1.19), ylim=(-1.27, 1.22), aspect="equal")
    network.axis("off")
    network.set_title("(a) 化学关联网络（|ρ|≥0.7）", loc="left", fontsize=9.5, pad=10)
    for row in edges.itertuples():
        pair = positions[[lookup[row.source], lookup[row.target]]]
        network.plot(pair[:, 0], pair[:, 1], color=BLUE, alpha=0.40, linewidth=0.6, zorder=1)
    network.scatter(positions[:, 0], positions[:, 1], color=BLUE,
                    s=12 + 5 * active["degree_0.7"].to_numpy(), zorder=2)
    angle = np.linspace(0, 2 * np.pi, len(isolated), endpoint=False)
    network.scatter(1.02 * np.cos(angle), 1.02 * np.sin(angle),
                    s=5.5, color=GREY, zorder=2)
    network.text(0, 1.13, "外围：150 个孤立节点逐点保留", ha="center", fontsize=8.5)
    network.text(0, -1.17, "内部：81 个非孤立节点；109 条边均为正相关", ha="center", fontsize=8.5)
    ranked = active.reset_index().sort_values(["degree_0.7", "field"],
                                              ascending=[False, True]).head(8)
    for number, row in enumerate(ranked.itertuples()):
        label = chr(ord("A") + number)
        x, y = positions[lookup[row.field]]
        network.text(x, y, label, ha="center", va="center", fontsize=8,
                     color="white", zorder=3)
        fig.text(0.035 + (number % 4) * 0.245, 0.165 - (number // 4) * 0.047,
                 f"{label}  {NETWORK_SHORT_NAMES[row.field]}", ha="left", fontsize=8.3)

    threshold_ax = fig.add_axes([0.715, 0.66, 0.26, 0.24])
    thresholds = [row["absolute_rho_threshold"] for row in summary["networks"]]
    edge_counts = [row["edges"] for row in summary["networks"]]
    threshold_ax.bar(range(3), edge_counts, color=[GREY, BLUE, GREY], width=0.55)
    threshold_ax.set(xticks=range(3), xticklabels=[f"{v:.1f}" for v in thresholds],
                     ylim=(0, 315), xlabel="|ρ| 下限", ylabel="无向边数")
    threshold_ax.set_title("(b) 阈值敏感性", loc="left", fontsize=9.5)
    for x, value in enumerate(edge_counts):
        threshold_ax.text(x, value + 9, str(value), ha="center", fontsize=8.5)
    grid(threshold_ax, "y")
    stability_ax = fig.add_axes([0.715, 0.295, 0.26, 0.205])
    frequencies = top3.bootstrap_top3_frequency.to_numpy()
    bins = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.00001])
    count, _ = np.histogram(frequencies, bins=bins)
    assert len(frequencies) == 135 and int((frequencies >= 0.8).sum()) == 14
    stability_ax.bar(np.arange(5), count, color=[GREY] * 4 + [BLUE], width=0.75)
    stability_ax.set(xticks=np.arange(5), xticklabels=["0", ".2", ".4", ".6", ".8"],
                     xlabel="频率区间下界（组距 0.2）", ylabel="关联位置数",
                     ylim=(0, max(count) + 12))
    stability_ax.set_title("(c) Top-3 入选稳定性", loc="left", fontsize=9.5)
    for x, value in enumerate(count):
        stability_ax.text(x, value + 1.5, str(value), ha="center", fontsize=8.5)
    grid(stability_ax, "y")
    fig.text(0.5, 0.065, "A—H 为度最高的八个节点；节点面积随度增加；布局位置不代表浓度", ha="center", fontsize=8.5)
    fig.text(0.5, 0.02, "200 次配对重抽样；14/135 个关联位置的入选频率达到 0.8", ha="center", fontsize=8.5)
    save(fig, "figure3_q2")


def figure4():
    summary = read_csv("q3_optimized/generation/cv_summary.csv")
    styles = ["Tripel", "Kriek", "Lambic"]
    methods = ["legacy_copula", "nested_selected", "style_bootstrap"]
    names = ["基准 Copula", "嵌套选择器", "风格重采样"]
    fig = new_figure(8.6)
    ax = fig.add_axes([0.105, 0.21, 0.87, 0.67])
    width = 0.23
    for k, (method, name, color) in enumerate(zip(methods, names, [GREY, BLUE, GOLD])):
        part = summary[summary.method.eq(method)].set_index("style").loc[styles]
        ax.bar(np.arange(3) + (k - 1) * width, part.energy_pca5_mean, width=width,
               color=color, label=name, yerr=part.energy_pca5_std,
               error_kw={"ecolor": DARK, "capsize": 2.5, "elinewidth": 0.7, "capthick": 0.7})
    ax.set(xticks=np.arange(3), xticklabels=["Tripel（n=28）", "Kriek（n=10）", "Lambic（n=11）"],
           ylabel="五维 PCA 能量差异", ylim=(0, 1.4))
    grid(ax, "y")
    fig.legend(loc="upper center", bbox_to_anchor=(0.54, 0.99), ncol=3, frameon=False,
               handlelength=1.5, columnspacing=1.2)
    fig.text(0.5, 0.07, "外层两次三折；每次生成 300 条、两个模拟种子", ha="center", fontsize=8.5)
    fig.text(0.5, 0.025, "误差线为六个折均值的折间标准差，不是置信区间；指标越低越好", ha="center", fontsize=8.5)
    save(fig, "figure4_generation")


PROFILE_TARGETS = [
    ("A_malt_all", "麦芽香气"), ("A_hops_all", "酒花香气"),
    ("A_esters_fruity", "水果酯香"), ("F_malt_all", "麦芽风味"),
    ("F_hops_all", "酒花风味"), ("acidity", "酸度"),
    ("bitternes", "苦度"), ("sweetness", "甜度"),
    ("X4vg", "4VG 酚味"), ("stale_hops", "陈旧酒花"),
    ("lactic", "乳酸味"), ("barnyard", "畜棚气味"),
]


def figure5():
    summary = read_csv("q3_optimized/generated_sensory_summary.csv")
    fig = new_figure(12.7)
    axes = fig.subplots(1, 3, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.90, bottom=0.27, wspace=0.15)
    keys, labels = zip(*PROFILE_TARGETS)
    y = np.arange(len(keys))
    for ax, style in zip(axes, ["Tripel", "Kriek", "Lambic"]):
        part = summary[summary["style"].eq(style)].set_index("target").loc[list(keys)]
        assert np.all(part.predicted_p05 <= part.predicted_mean)
        assert np.all(part.predicted_mean <= part.predicted_p95)
        ax.axvline(0, color="#7C858C", linewidth=0.6)
        ax.axhspan(8.55, 9.45, color="#F0F2F3", zorder=-1)
        ax.hlines(y, part.predicted_p05, part.predicted_p95, color=BLUE, alpha=0.6, linewidth=2.4)
        ax.scatter(part.predicted_mean, y, color=BLUE, s=15, zorder=3)
        ax.set_title(style, pad=8)
        ax.set(xlim=(-1.3, 2.85), xticks=[-1, 0, 1, 2], yticks=y, yticklabels=labels,
               ylim=(11.6, -0.6))
        ax.tick_params(axis="y", length=0)
        grid(ax)
    fig.text(0.58, 0.19, "标准化感官值", ha="center", fontsize=9)
    fig.text(0.5, 0.12, "圆点：生成谱预测均值；线段：5%—95% 分位范围（每风格 300 条）", ha="center", fontsize=8.5)
    fig.text(0.5, 0.075, "该范围描述生成分布，不是预测置信区间，也不是原始 0—6 分", ha="center", fontsize=8.5)
    fig.text(0.5, 0.025, "灰行陈旧酒花使用非 Pils 单目标代理；其余维度使用全量多输出代理", ha="center", fontsize=8.5)
    save(fig, "figure5_profiles")


def figure6():
    summary = read_csv("q3_refined/transfer/transfer_summary.csv")
    part = summary[summary.arm.eq("fixed") & summary.pool.eq("contracted_6004")
                   & summary.bound.eq("retain80")]
    source_order = [124, 81, 218, 96, 230, 185, 139, 209, 119, 90, 28]
    margin = part[part.ranking.eq("margin")].set_index("source_beer").loc[source_order]
    geometry = part[part.ranking.eq("geometry")].set_index("source_beer").loc[source_order]
    assert len(margin) == len(geometry) == 11
    assert margin.fresh_consistency_passed.sum() == geometry.fresh_consistency_passed.sum() == 10
    assert margin.loc[~margin.fresh_consistency_passed].index.tolist() == [96]
    assert geometry.loc[~geometry.fresh_consistency_passed].index.tolist() == [96]
    fig = new_figure(10.3)
    ax = fig.add_axes([0.16, 0.28, 0.80, 0.60])
    y = np.arange(11)
    m = margin.target_improvement.to_numpy() * 100
    g = geometry.target_improvement.to_numpy() * 100
    ax.hlines(y, m, g, color="#A7B1B8", linewidth=1)
    ax.scatter(m, y, color=BLUE, marker="o", s=25, label="余量优先", zorder=3)
    ax.scatter(g, y, edgecolor=GOLD, facecolor="white", marker="^", linewidth=1.1,
               s=35, label="几何优先", zorder=3)
    failed_y = source_order.index(96)
    ax.scatter([m[failed_y]], [failed_y], color=FAIL, marker="x", s=55, linewidth=1.4,
               label="审计未通过", zorder=4)
    ax.set(yticks=y, yticklabels=[str(beer) + (" *" if beer == 96 else "") for beer in source_order],
           ylim=(10.7, -0.7), xlim=(0, 85), xticks=[0, 20, 40, 60, 80],
           xlabel="向 Tripel 的目标距离改善（%）", ylabel="源 beer 编号")
    grid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False,
              handletextpad=0.4, columnspacing=1.0)
    fig.text(0.5, 0.12, "固定树 / 中央 6004 候选 / retain80；两种排序均为 10/11 通过", ha="center", fontsize=8.5)
    fig.text(0.5, 0.07, "所选谱改善中位数：余量优先 57.74%，几何优先 70.08%", ha="center", fontsize=8.5)
    fig.text(0.5, 0.02, "* beer 96 的审计失败保留；通过数为模型检查结果，不是实际品评成功数", ha="center", fontsize=8.5)
    save(fig, "figure6_transfer")


def main(only=None):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for number, draw in enumerate([figure1, figure2, figure3, figure4, figure5, figure6], 1):
        if only is None or number == only:
            draw()
    descriptions = [
        ("figure1_workflow", "图1 三问的数据处理、模型与验证流程。第一问使用231维原表示；第三问使用69维校准量与157维相对量组成的226维混合表示。迁移训练排除全部11款Pils。", 11.2,
         ["outputs/q1/summary.json", "outputs/q2/summary.json", "outputs/q3_optimized/generation/summary.json", "outputs/q3_refined/stale/summary.json", "src/q3_data.py"]),
        ("figure2_q1", "图2 第一问四种方法的五折折外误差、宏平均R²及ExtraTrees缺失压力结果。裁剪Ridge为稳健性对照；额外隐藏20%已观测特征后，ExtraTrees的RMSE由0.4862增至0.4992。", 11.2,
         ["outputs/q1/metrics_overall.csv", "outputs/q1/missingness_stress.csv", "outputs/q1_robust/metrics_overall.csv"]),
        ("figure3_q2", "图3 化学关联网络及稳定性。(a) 实际|Spearman ρ|≥0.7网络，保留全部231节点和109条正相关边；81个非孤立节点采用固定种子20261023的分量内力导向布局及分量打包，并进行显示避碰，150个孤立节点在外围环逐点保留，节点面积随度增加。A—H为度最高八个节点，图内列简名；布局距离不表示化学浓度或因果关系。(b) 三个阈值下的边数。(c) 135个Top-3位置的200次配对重抽样入选频率，其中14个达到0.8；频率区间左闭右开，最后一组包含1。", 11.8,
         ["outputs/q2/network_nodes.csv", "outputs/q2/network_edges_rho_0.7.csv", "outputs/q2/summary.json", "outputs/q2/top3_compounds.csv"]),
        ("figure4_generation", "图4 三种目标风格的生成能量差异，比较基准Copula、完整嵌套选择器及重采样对照。柱为外层六个折均值的平均值，误差线为其标准差；每折先平均两个模拟种子。折间标准差不是置信区间，真实样本数列于横轴。", 8.6,
         ["outputs/q3_optimized/generation/cv_summary.csv", "outputs/q3_optimized/generation/cv_fold_means.csv", "outputs/q2/summary.json"]),
        ("figure5_profiles", "图5 三种风格在列示的12个代表感官维度上的生成画像。圆点为每风格300条谱的预测均值，线段为5%—95%生成分布范围，三个面板使用共同横轴。陈旧酒花来自239款非Pils训练的固定单目标树，其余维度来自226维全量多输出代理；区间不是校准预测置信区间，数值不是原始0—6分。", 12.7,
         ["outputs/q3_optimized/generated_sensory_summary.csv", "outputs/q3_optimized/generation/summary.json", "outputs/q3_optimized/stale_fixed/summary.json"]),
        ("figure6_transfer", "图6 固定树、中央6004候选池与retain80条件下，11款Pils源的所选谱目标距离改善。余量优先和几何优先均通过10/11款，beer 96的审计失败保留；改善中位数分别为57.74%与70.08%，包括已选但失败的谱。几何优先为探索性应用示例，模型检查通过不代表实际品评成功。", 10.3,
         ["outputs/q3_refined/transfer/transfer_summary.csv", "outputs/q3_refined/transfer/frontier.csv"]),
    ]
    manifest = {
        "renderer": "matplotlib", "font_file": str(CJK_FONT), "dpi": 300,
        "widthcm": WIDTH_CM, "source_root": "solution_a", "sources_mutated": False,
        "figures": [{"id": index, "caption": caption,
                     "relativepath": f"figures/{name}.png",
                     "svg_relativepath": f"figures/{name}.svg",
                     "widthcm": WIDTH_CM, "heightcm": height,
                     "sources": sources}
                    for index, (name, caption, height, sources) in enumerate(descriptions, 1)],
        "profile_targets": [{"target": key, "label": label} for key, label in PROFILE_TARGETS],
        "verification": {
            "figure3_nodes": 231, "figure3_edges": 109,
            "figure3_active_nodes": 81, "figure3_isolated_nodes": 150,
            "figure3_all_edges_positive": True, "figure3_layout_seed": 20261023,
            "figure3_top3_positions": 135, "figure3_frequency_at_least_0_8": 14,
            "figure6_sources": 11, "figure6_passed_each_ranking": 10,
            "figure6_failed_source_each_ranking": 96,
        },
    }
    for figure in manifest["figures"]:
        assert all((SOLUTION / source).is_file() for source in figure["sources"])
    (PAPER / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {'six figure pairs' if only is None else 'figure ' + str(only)} and figure_manifest.json in {PAPER}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=int, choices=range(1, 7), help="Render one figure only")
    main(parser.parse_args().only)
