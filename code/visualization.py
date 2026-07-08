import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter1d

os.makedirs("images", exist_ok=True)

# input data
df = pd.read_excel("MASS_SHOOTING.xlsx")

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor":  "#161b22",
    "axes.edgecolor":   "#30363d", "axes.labelcolor": "#e6edf3",
    "xtick.color":      "#8b949e", "ytick.color":     "#8b949e",
    "text.color":       "#e6edf3", "grid.color":      "#21262d",
    "grid.linewidth":   0.8,       "font.family":     "DejaVu Sans",
})
RED    = "#e05252"
AMBER  = "#f0a500"
BLUE   = "#58a6ff"
GREEN  = "#3fb950"
GRAY   = "#8b949e"
WHITE  = "#e6edf3"
PURPLE = "#bc8cff"   # ← median / reference lines always this color


# ── Fig 1: Incidents per year  (+2010 vertical, smooth trend) ─────────────────
fig, ax = plt.subplots(figsize=(13, 5))
yearly = df.groupby("YEAR").size().reset_index(name="incidents")
ax.bar(yearly["YEAR"], yearly["incidents"], color=RED, alpha=0.85, width=0.8, zorder=3)

smooth = yearly["incidents"].rolling(5, center=True).mean()
ax.plot(yearly["YEAR"], smooth, color=AMBER, lw=2.2, label="5-yr moving avg", zorder=4)

ax.axvline(2010, color=GREEN, lw=2, linestyle="--", zorder=5)
ax.text(2009.5, 13.5, "2010:\nSharp increase",
        color=GREEN, fontsize=8.5, va="top", ha="right")

ax.set_ylim(0, 18)  # headroom so annotation stays inside

peak = yearly.loc[yearly["incidents"].idxmax()]
ax.annotate(f"Peak: {int(peak.incidents)} incidents ({int(peak.YEAR)})",
            xy=(peak.YEAR, peak.incidents),
            xytext=(peak.YEAR - 7, peak.incidents - 3.5),  # below the tip
            arrowprops=dict(arrowstyle="->", color=WHITE, lw=1.2),
            color=WHITE, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22", edgecolor="#30363d", alpha=0.85))

ax.set_title("US Mass Shootings Per Year  (1966–2026)", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Year"); ax.set_ylabel("Number of Incidents")
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(facecolor="#161b22", edgecolor="#30363d")
ax.grid(axis="y", zorder=0)
plt.tight_layout()
plt.savefig("images/num_incident_over_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 1")


# ── Fig 2a: Fatalities & injuries  (smooth lines) ─────────────────────────────
yearly2 = df.groupby("YEAR").agg(
    fatalities=("FATALITIES", "sum"), injured=("INJURED", "sum")
).reset_index()

fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(yearly2["YEAR"], yearly2["injured"],    alpha=0.35, color=BLUE, label="Injured")
ax.fill_between(yearly2["YEAR"], yearly2["fatalities"], alpha=0.65, color=RED,  label="Fatalities")
ax.plot(yearly2["YEAR"], yearly2["fatalities"], color=RED,  lw=2)
ax.plot(yearly2["YEAR"], yearly2["injured"],    color=BLUE, lw=2)

lv_inj = int(yearly2.loc[yearly2.YEAR == 2017, "injured"].values[0])
ax.annotate("Las Vegas 2017\n(546 injured)",
            xy=(2017, lv_inj),
            xytext=(2005, 200),   # well below the spike, inside chart
            arrowprops=dict(arrowstyle="->", color=WHITE, lw=1.2),
            color=WHITE, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22", edgecolor="#30363d", alpha=0.85))

ax.set_title("Fatalities & Injuries Per Year  (1966–2026)", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Year"); ax.set_ylabel("Number of People")
ax.legend(facecolor="#161b22", edgecolor="#30363d")
ax.grid(axis="y", zorder=0)
plt.tight_layout()
plt.savefig("images/num_victim_overtime.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 2a")


# ── Fig 2b: Same chart, Las Vegas removed ─────────────────────────────────────
df_nolv  = df[~df["CASE"].str.contains("LAS VEGAS", na=False)]
ynolv    = df_nolv.groupby("YEAR").agg(
    fatalities=("FATALITIES", "sum"), injured=("INJURED", "sum")
).reset_index()

fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(ynolv["YEAR"], ynolv["injured"],    alpha=0.35, color=BLUE, label="Injured")
ax.fill_between(ynolv["YEAR"], ynolv["fatalities"], alpha=0.65, color=RED,  label="Fatalities")
ax.plot(ynolv["YEAR"], ynolv["fatalities"], color=RED,  lw=2)
ax.plot(ynolv["YEAR"], ynolv["injured"],    color=BLUE, lw=2)

ax.text(0.99, 0.97, "Las Vegas 2017 removed\n(60 killed, 546 injured)",
        transform=ax.transAxes, ha="right", va="top", color=AMBER, fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor=AMBER, alpha=0.8))

ax.set_title("Fatalities & Injuries Per Year — Las Vegas Massacre Removed",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Year"); ax.set_ylabel("Number of People")
ax.legend(facecolor="#161b22", edgecolor="#30363d")
ax.grid(axis="y", zorder=0)
plt.tight_layout()
plt.savefig("images/num_victim_overtime_no_lv.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 2b")


# ── Fig 3: Top 10 deadliest  (median lines, single-line labels) ───────────────
fig, ax = plt.subplots(figsize=(13, 6))
top10 = df.nlargest(10, "TOTAL_VICTIMS")[
    ["CASE", "YEAR", "FATALITIES", "INJURED", "TOTAL_VICTIMS"]
].copy()
top10["label"] = top10.apply(lambda r: f"{r['CASE']}  ({int(r['YEAR'])})", axis=1)
top10 = top10.sort_values("TOTAL_VICTIMS")

ax.barh(top10["label"], top10["FATALITIES"], color=RED,  label="Fatalities")
ax.barh(top10["label"], top10["INJURED"],    left=top10["FATALITIES"],
        color=BLUE, alpha=0.75, label="Injured")

med_tot = top10["INJURED"].median()
med_fat = top10["FATALITIES"].median()
ax.axvline(med_tot, color=PURPLE, lw=2,   linestyle="--", label=f"Median injured: {med_tot:.0f}")
ax.axvline(med_fat, color=AMBER,  lw=1.8, linestyle=":",  label=f"Median fatalities: {med_fat:.0f}")

for i, (_, row) in enumerate(top10.iterrows()):
    ax.text(row["TOTAL_VICTIMS"] + 4, i, str(row["TOTAL_VICTIMS"]),
            va="center", color=WHITE, fontsize=8.5, fontweight="bold")

ax.set_title("Top 10 Deadliest Mass Shooting Incidents", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Number of Victims")
ax.legend(facecolor="#161b22", edgecolor="#30363d", fontsize=8, loc="lower right")
ax.grid(axis="x", zorder=0)
plt.tight_layout()
plt.savefig("images/top10_most_victim_incident.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 3")


# ── Fig 4: Incidents per state  (median line, purple) ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
state_cnt = df.groupby("STATE").size().sort_values(ascending=True).tail(15)
med_state = df.groupby("STATE").size().median()

colors4 = [RED if s == "CALIFORNIA" else GRAY for s in state_cnt.index]
bars4   = ax.barh(state_cnt.index, state_cnt.values, color=colors4)
ax.axvline(med_state, color=PURPLE, lw=2, linestyle="--",
           label=f"Median: {med_state:.0f} per state")

for bar, val in zip(bars4, state_cnt.values):
    ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", color=WHITE, fontsize=8.5)

ax.set_title("Mass Shootings by State — Top 15  (1966–2026)",
             fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Number of Incidents")
ax.legend(facecolor="#161b22", edgecolor="#30363d")
ax.grid(axis="x", zorder=0)
plt.tight_layout()
plt.savefig("images/num_incident_per_state.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 4")


# ── Fig 5: Locations  (OTHER removed, median line added) ──────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
place_cnt = (
    df[df["LOCATION_PLACE"] != "OTHER"]["LOCATION_PLACE"]
    .value_counts().sort_values()
)
med_place = place_cnt.median()
colors5   = [RED if p == "WORKPLACE" else BLUE for p in place_cnt.index]
bars5     = ax.barh(place_cnt.index, place_cnt.values, color=colors5)
ax.axvline(med_place, color=PURPLE, lw=2, linestyle="--",
           label=f"Median: {med_place:.0f} incidents")

for bar, val in zip(bars5, place_cnt.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", color=WHITE, fontsize=8.5)

ax.set_title('Top Locations of Mass Shootings  ("Other" excluded)',
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Number of Incidents")
ax.legend(facecolor="#161b22", edgecolor="#30363d")
ax.grid(axis="x", zorder=0)
plt.tight_layout()
plt.savefig("images/top10_most_place.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 5")


# ── Fig 6: Race (sorted) + Gender pie (with counts) ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

race_cnt   = df["RACE"].value_counts().sort_values()
colors_r   = ["#85c1e9", "#f0b27a", "#c39bd3", "#58d68d", AMBER, BLUE, RED]
axes[0].barh(race_cnt.index, race_cnt.values, color=colors_r[: len(race_cnt)])
axes[0].set_title("Shooter Race", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Number of Incidents")
axes[0].grid(axis="x")
for i, (idx, val) in enumerate(race_cnt.items()):
    axes[0].text(val + 0.3, i, str(val), va="center", color=WHITE, fontsize=8.5)

gender_cnt = df["GENDER"].value_counts()
labels_g   = [f"{k}\n({v} cases, {v / len(df) * 100:.1f}%)" for k, v in gender_cnt.items()]
axes[1].pie(gender_cnt.values, labels=labels_g, colors=[BLUE, RED], startangle=90,
            textprops={"color": WHITE, "fontsize": 10},
            wedgeprops={"edgecolor": "#0d1117", "linewidth": 2})
axes[1].set_title("Shooter Gender", fontsize=13, fontweight="bold")

fig.suptitle("Shooter Demographics", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("images/shooter_race_and_gender.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 6")


# ── Fig 7: Age histogram  (median line purple, under-18 highlight) ─────────────
fig, ax = plt.subplots(figsize=(11, 5))
ages     = df["AGE_OF_SHOOTER"].dropna()
mean_age = ages.mean()
med_age  = ages.median()
ci_low   = mean_age - 1.96 * ages.std() / np.sqrt(len(ages))
ci_high  = mean_age + 1.96 * ages.std() / np.sqrt(len(ages))
under18  = (ages < 18).sum()

ax.axvspan(ages.min(), 18, color=AMBER, alpha=0.12, zorder=1)
ax.hist(ages, bins=20, color=BLUE, alpha=0.8, edgecolor="#0d1117", zorder=3)
ymax = ax.get_ylim()[1]

ax.axvline(18,      color=AMBER,  lw=1.8, linestyle="--",
           label=f"Age 18  ({under18} shooters under 18)", zorder=4)
ax.axvline(med_age, color=PURPLE, lw=2.2, linestyle="-",
           label=f"Median: {med_age:.0f} yrs", zorder=5)
ax.axvspan(ci_low, ci_high, color=RED, alpha=0.15,
           label=f"95% CI mean: [{ci_low:.1f}, {ci_high:.1f}]")

ax.text(14, ymax * 0.6, f"Under 18\n{under18} cases",
        color=AMBER, fontsize=8.5, ha="center")

ax.set_title("Age Distribution of Mass Shooting Perpetrators",
             fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Age"); ax.set_ylabel("Frequency")
ax.legend(facecolor="#161b22", edgecolor="#30363d", fontsize=8.5)
ax.grid(axis="y", zorder=0)
plt.tight_layout()
plt.savefig("images/average_age_of_shooter.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 7")


# ── Fig 8: Weapon types ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
weapons  = {"Handgun": 210, "Rifle": 149, "Shotgun": 49, "Revolver": 40, "Knife": 3}
colors_w = [RED if k in ["Handgun", "Rifle"] else BLUE for k in weapons]
bars8    = ax.bar(weapons.keys(), weapons.values(), color=colors_w, edgecolor="#0d1117", zorder=3)
for bar, val in zip(bars8, weapons.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 2,
            str(val), ha="center", color=WHITE, fontsize=10, fontweight="bold")
ax.set_title("Weapons Used in Mass Shootings\n(Many incidents use multiple weapons)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_ylabel("Number of Incidents")
ax.grid(axis="y", zorder=0)
plt.tight_layout()
plt.savefig("images/weapon_type.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 8")


# ── Fig 9: Weapons obtained legally (pie with counts) ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
legal     = df["WEAPONS_OBTAINED_LEGALLY"].value_counts()
colors_l  = {"YES": RED, "NO": BLUE, "UNKNOWN": GRAY}
c9        = [colors_l.get(k, GRAY) for k in legal.index]
labels_l  = [f"{k}\n({v} cases, {v / len(df) * 100:.1f}%)" for k, v in legal.items()]
ax.pie(legal.values, labels=labels_l, colors=c9, startangle=140,
       textprops={"color": WHITE, "fontsize": 11},
       wedgeprops={"edgecolor": "#0d1117", "linewidth": 2})
ax.set_title("Weapons Obtained Legally?", fontsize=15, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("images/weapon_obtained_legally.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 9")


# ── Fig 10: Motivation  (OTHER removed) ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
motiv   = {
    "Workplace / Financial":  60, "Domestic / Relationship": 56,
    "Paranoia":               42, "Revenge / Anger":         36,
    "Hate / Racism":          16, "Terrorism / Ideology":     9,
}
motiv_s  = dict(sorted(motiv.items(), key=lambda x: x[1]))
colors_m = [RED if v == max(motiv_s.values()) else BLUE for v in motiv_s.values()]
bars10   = ax.barh(list(motiv_s.keys()), list(motiv_s.values()),
                   color=colors_m, edgecolor="#0d1117", zorder=3)
for bar, val in zip(bars10, motiv_s.values()):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", color=WHITE, fontsize=9)
ax.set_title('Motivations Behind Mass Shootings\n("Other/Not Specified" excluded)',
             fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("Number of Incidents")
ax.grid(axis="x", zorder=0)
plt.tight_layout()
plt.savefig("images/motivation_behide_shooting.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 10")


# ── Fig 11: Mental health pie ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
mh       = df["PRIOR_SIGNS_MENTAL_HEALTH_ISSUES"].value_counts()
colors_mh = {"YES": RED, "NO": BLUE, "UNKNOWN": GRAY}
c11      = [colors_mh.get(k, GRAY) for k in mh.index]
labels_mh = [f"{k}\n({v} cases, {v / len(df) * 100:.1f}%)" for k, v in mh.items()]
ax.pie(mh.values, labels=labels_mh, colors=c11, startangle=140,
       textprops={"color": WHITE, "fontsize": 11},
       wedgeprops={"edgecolor": "#0d1117", "linewidth": 2})
ax.set_title("Prior Signs of Mental Health Issues", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("images/piechart_mental_health_issue.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 11")

print("\n✅  All charts saved to images/")
