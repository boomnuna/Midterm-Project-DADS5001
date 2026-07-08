import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
import os

os.makedirs("images", exist_ok=True)

#TODO: population and GDP data
# read population data 
population = "State_Population_1900_2025.csv"
population  = pd.read_csv(population)
# read GDP data
gdp = "GDP.xlsx"
gdp = pd.read_excel(gdp)

# merge population with GDP
merged_data = population.merge(gdp, on=['STATE','YEAR'], how='left')
merged_data = merged_data.dropna(subset=['GDP'])
# change type of data 
merged_data['POPULATION'] = (
    merged_data['POPULATION']
    .astype(str)
    .str.replace(',', '')
    .astype(float)
)
merged_data['GDP'] = (
    merged_data['GDP']
    .astype(str)
    .str.replace(',', '')
    .astype(float)
)
merged_data['STATE'] = merged_data['STATE'].str.lower()
# create new column
merged_data['GDP_PER_POP'] = merged_data['GDP'] / merged_data['POPULATION']

#TODO: mass shooting data
# load data 
mass_shoooting = pd.read_excel('MASS_SHOOTING.xlsx')
mass_shoooting = mass_shoooting[['DATE', 'STATE', 'FATALITIES', 'INJURED']]
mass_shoooting['DATE'] = pd.to_datetime(mass_shoooting['DATE'], dayfirst=True, errors='coerce')
# change data type
mass_shoooting['STATE'] = mass_shoooting['STATE'].str.lower()
mass_shoooting['YEAR'] = mass_shoooting['DATE'].dt.year
# aggregation
mass_shoooting = mass_shoooting.groupby(['YEAR', 'STATE']).agg(
    FATALITIES=('FATALITIES', 'sum'),
    INJURED=('INJURED', 'sum'),
    INCIDENT_COUNT=('STATE', 'count')
).reset_index()

#TODO: final data
# merge population, GDP, mass shooting data
final_df = merged_data.merge(mass_shoooting, on=['YEAR', 'STATE'], how='left')

# final aggregation
state_summary = final_df.groupby('STATE').agg(
    INCIDENT_COUNT=('INCIDENT_COUNT', 'sum'),
    POPULATION=('POPULATION', 'mean'),
    GDP=('GDP', 'mean'),
    GDP_PerCapita=('GDP_PER_POP', 'mean'),
    FATALITIES=('FATALITIES', 'sum'),
    INJURED=('INJURED', 'sum')
).reset_index()

df0 = state_summary.copy()
df1 = df0[df0['INCIDENT_COUNT'] > 0].copy()
df1['INCIDENT_PER_MILLION'] = df1['INCIDENT_COUNT'] / df1['POPULATION'] * 1_000_000
df1['FATALITIES_PER_INCIDENT'] = df1['FATALITIES'] / df1['INCIDENT_COUNT']

#TODO: theme
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor":  "#161b22",
    "axes.edgecolor":   "#30363d", "axes.labelcolor": "#e6edf3",
    "xtick.color":      "#8b949e", "ytick.color":     "#8b949e",
    "text.color":       "#e6edf3", "grid.color":      "#21262d",
    "grid.linewidth":   0.8,       "font.family":     "DejaVu Sans",
})
BLUE  = "#58a6ff"
WHITE = "#e6edf3"
GRAY  = "#8b949e"

HIGHLIGHT = {
    "california": "#ff4d4d",   # RED
    "texas":      "#ffd700",   # YELLOW
    "florida":    "#00e5ff",   # CYAN
}

#TODO: combined figure — 1 row, 2 cols

fig, (ax_heat, ax_scatter) = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor("#0d1117")
# ── RIGHT: Heatmap — Correlation Matrix ───────────────────────────────────────
ax_heat.set_facecolor("#0d1117")

df_corr = df1[[
    'INCIDENT_COUNT',
    'POPULATION',
    'GDP',
    'GDP_PerCapita',
]].rename(columns={
    'INCIDENT_COUNT':          'Incidents',
    'POPULATION':              'Population',
    'GDP_PerCapita':           'GDP per Capita',
})

cols = df_corr.columns.tolist()
n    = len(cols)
r_mat = np.zeros((n, n))
p_mat = np.ones((n, n))

for i, c1 in enumerate(cols):
    for j, c2 in enumerate(cols):
        if i == j:
            r_mat[i, j] = 1.0
            p_mat[i, j] = 0.0
        else:
            r, p = stats.pearsonr(df_corr[c1].dropna(), df_corr[c2].dropna())
            r_mat[i, j] = r
            p_mat[i, j] = p

cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_rdbu",
    ["#1a6fa8", "#2e86c1", "#aed6f1", "#f5f5f5", "#f1948a", "#e74c3c", "#a93226"],
    N=256
)
im = ax_heat.imshow(r_mat, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

# grid lines
for i in range(n + 1):
    ax_heat.axhline(i - 0.5, color='#0d1117', lw=2)
    ax_heat.axvline(i - 0.5, color='#0d1117', lw=2)

# cell values
for i in range(n):
    for j in range(n):
        r_val = r_mat[i, j]
        text_color = "white" if abs(r_val) > 0.5 else "#1a1a2e"
        label  = "1.00" if i == j else f"{r_val:.2f}"
        weight = 'bold' if i == j else 'normal'
        ax_heat.text(j, i, label,
                     ha='center', va='center',
                     fontsize=10, fontweight=weight,
                     color=text_color)

# axis labels
ax_heat.set_xticks(range(n))
ax_heat.set_yticks(range(n))
ax_heat.set_xticklabels(cols, rotation=0, ha='center', fontsize=9, color=WHITE)
ax_heat.set_yticklabels(cols, fontsize=9, color=WHITE)

# colorbar
cbar = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02)
cbar.ax.tick_params(colors='#8b949e', labelsize=8)
cbar.set_label("Pearson r", color='#8b949e', fontsize=8)
cbar.outline.set_edgecolor('#30363d')

ax_heat.set_title("Correlation Matrix — State-Level Mass Shooting Factors",
                  fontsize=13, fontweight='bold', pad=12, color=WHITE)


# ── LEFT: Scatter — Population vs Incident Count ──────────────────────────────
slope, intercept, r, p, _ = stats.linregress(df1['POPULATION'], df1['INCIDENT_COUNT'])
x_line = np.linspace(df1['POPULATION'].min(), df1['POPULATION'].max(), 200)
ax_scatter.plot(x_line / 1e6, slope * x_line + intercept,
                color=BLUE, lw=1.8, ls='--', alpha=0.7,
                label=f"Regression", zorder=2)

for _, row in df1.iterrows():
    if row['STATE'] not in HIGHLIGHT:
        ax_scatter.scatter(row['POPULATION'] / 1e6, row['INCIDENT_COUNT'],
                           color='r', s=50, alpha=0.5, zorder=3, edgecolors='none')

for state, color in HIGHLIGHT.items():
    row = df1[df1['STATE'] == state]
    if len(row) == 0:
        continue
    row = row.iloc[0]
    ax_scatter.scatter(row['POPULATION'] / 1e6, row['INCIDENT_COUNT'],
                       color=color, s=130, alpha=1.0, zorder=5,
                       edgecolors='white', linewidths=1.2)

label_offsets = {
    "california": (-2.5,  0.1),
    "texas":      (-1.0,  1.0),
    "florida":    ( 0.4, -1.7),
}
for state, (dx, dy) in label_offsets.items():
    row = df1[df1['STATE'] == state]
    if len(row) == 0:
        continue
    row = row.iloc[0]
    ax_scatter.text(row['POPULATION'] / 1e6 + dx,
                    row['INCIDENT_COUNT'] + dy,
                    state.title(),
                    color=HIGHLIGHT[state], fontsize=9, fontweight='bold',
                    ha='center', va='bottom')

ax_scatter.set_title("Population vs Number of Mass Shooting Incidents by State",
                     fontsize=13, fontweight='bold', pad=12)
ax_scatter.set_xlabel("Average State Population (Millions)")
ax_scatter.set_ylabel("Total Incidents (1998–2024)")
ax_scatter.legend(facecolor="#161b22", edgecolor="#30363d", fontsize=9)
ax_scatter.grid(zorder=0)
xlim = ax_scatter.get_xlim()
ax_scatter.set_xlim(xlim[0], xlim[1] * 1.05)


#TODO: save
plt.tight_layout()
plt.savefig("images/population_gdp.png", dpi=150,
            bbox_inches='tight', facecolor="#0d1117")
plt.close()
print("Saved: images/population_gdp.png")
