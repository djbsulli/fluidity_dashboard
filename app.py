"""
Positional Fluidity Dashboard — 2015-16 Premier League
Built from clean_analysis notebook.

Data folder structure:
    data/avg_team_fluidity.csv
    data/team_fluidity_full.csv
    data/season_player_stats.csv
    data/valid_touches.csv

Export from notebook with:
    import os; os.makedirs('data', exist_ok=True)
    avg_team_fluidity.to_csv('data/avg_team_fluidity.csv', index=False)
    team_fluidity_full.to_csv('data/team_fluidity_full.csv', index=False)
    season_player_stats.to_csv('data/season_player_stats.csv', index=False)
    valid_touches.to_csv('data/valid_touches.csv', index=False)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from mplsoccer import Pitch

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="PL Fluidity 2015-16",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="collapsedControl"] {{ display: block !important; }}
.dash-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.6rem; letter-spacing: 5px;
    color: #1a1a2e; line-height: 1; margin-bottom: 0;
}
.dash-sub {
    font-size: 0.72rem; color: #888;
    letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.section-hdr {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.15rem; letter-spacing: 3px; color: #1a1a2e;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 0.3rem; margin: 1.2rem 0 0.7rem;
}
.metric-row { display:flex; gap:0.7rem; flex-wrap:wrap; margin-bottom:0.5rem; }
.metric-card {
    background:#fff; border:1px solid #e0e0e0; border-radius:8px;
    padding:0.8rem 1.1rem; min-width:140px; flex:1;
    box-shadow:0 1px 3px rgba(0,0,0,0.06);
}
.metric-label {
    font-size:0.63rem; color:#999; letter-spacing:2px;
    text-transform:uppercase; margin-bottom:0.2rem;
}
.metric-value {
    font-family:'Bebas Neue',sans-serif;
    font-size:1.6rem; line-height:1;
}
.mv-pos  { color:#16a34a; }
.mv-neg  { color:#dc2626; }
.mv-blue { color:#2563eb; }
.team-bar  { width:5px; height:44px; border-radius:3px; display:inline-block; }
.team-name { font-family:'Bebas Neue',sans-serif; font-size:1.8rem; letter-spacing:3px; }
.qbadge {
    display:inline-block; padding:0.18rem 0.6rem; border-radius:4px;
    font-size:0.68rem; font-weight:600; letter-spacing:1px;
    text-transform:uppercase; margin-left:0.5rem;
}
.zone-card {
    background:#f8f9fa; border:1px solid #e0e0e0; border-radius:8px;
    padding:0.7rem 1rem; text-align:center;
}
.zone-label { font-size:0.7rem; color:#777; letter-spacing:2px; text-transform:uppercase; }
.zone-val   { font-family:'Bebas Neue',sans-serif; font-size:1.4rem; }
.pcard {
    background:#fff; border:1px solid #e0e0e0;
    border-left:3px solid #2563eb; border-radius:7px;
    padding:0.6rem 0.9rem; margin-bottom:0.4rem;
    box-shadow:0 1px 3px rgba(0,0,0,0.04);
}
.pcard-name { font-size:0.86rem; font-weight:600; color:#1a1a2e; }
.pcard-meta { font-size:0.7rem; color:#888; margin-top:0.1rem; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:1.2rem; padding-bottom:2rem; }
div[data-testid="stHorizontalBlock"] {{ gap: 1rem; }}
</style>
""", unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════
# CONSTANTS  (from notebook cell 40)
# ══════════════════════════════════════════════════════════════
TEAM_COLOURS = {
    'Leicester City': '#003090',    'Crystal Palace': '#1B458F',
    'Sunderland': '#EB172B',        'West Ham United': '#7A263A',
    'Norwich City': '#FFF200',      'West Bromwich Albion': '#122F67',
    'Newcastle United': '#241F20',  'AFC Bournemouth': '#DA291C',
    'Chelsea': '#034694',           'Watford': '#FBEE23',
    'Everton': '#003399',           'Southampton': '#D71920',
    'Tottenham Hotspur': '#F5F5F0', 'Stoke City': '#E03A3E',
    'Arsenal': '#EF0107',           'Aston Villa': '#4E0A28',
    'Liverpool': '#C8102E',         'Manchester United': '#DA291C',
    'Swansea City': '#8A9BA8',      'Manchester City': '#6CABDD',
}

TEAM_ABBR = {
    'Leicester City': 'LEI',    'Crystal Palace': 'CRY',  'Sunderland': 'SUN',
    'West Ham United': 'WHU',   'Norwich City': 'NOR',    'West Bromwich Albion': 'WBA',
    'Newcastle United': 'NEW',  'AFC Bournemouth': 'BOU', 'Chelsea': 'CHE',
    'Watford': 'WAT',           'Everton': 'EVE',         'Southampton': 'SOU',
    'Tottenham Hotspur': 'TOT', 'Stoke City': 'STK',      'Arsenal': 'ARS',
    'Aston Villa': 'AVL',       'Liverpool': 'LIV',       'Manchester United': 'MUN',
    'Swansea City': 'SWA',      'Manchester City': 'MCI',
}

# label offsets for dense quadrant cluster (from notebook cell 40)
LABEL_OFFSETS = {
    'Arsenal':     (5,   7),
    'Stoke City':  (-10, -12),
    'Liverpool':   (10,  -12),
    'Southampton': (14,  4),
    'Chelsea':     (-14, 4),
}

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    atf  = pd.read_csv('avg_team_fluidity.csv')
    tff  = pd.read_csv('team_fluidity_full.csv')
    sps  = pd.read_csv('season_player_stats.csv')
    vt   = pd.read_csv('valid_touches.csv')
    return atf, tff, sps, vt

avg_team_fluidity, team_fluidity_full, season_player_stats, valid_touches = load_data()

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def quadrant_label(mean_z, std_z):
    if   mean_z >= 0 and std_z <= 0: return "Consistently Fluid",   "#16a34a"
    elif mean_z >= 0 and std_z >  0: return "Inconsistently Fluid", "#d97706"
    elif mean_z <  0 and std_z <= 0: return "Consistently Rigid",   "#2563eb"
    else:                             return "Inconsistently Rigid",  "#dc2626"

def short_name(full):
    parts = full.strip().split()
    return parts[0] if len(parts) == 1 else f"{parts[0]} {parts[-1]}"

def fluidity_rank(team):
    ranked = avg_team_fluidity.sort_values('season_overall_z', ascending=False).reset_index(drop=True)
    return int(ranked[ranked['team'] == team].index[0]) + 1

def mv_class(val):
    return 'mv-pos' if val > 0 else 'mv-neg'

# ══════════════════════════════════════════════════════════════
# HTML BLOCKS
# ══════════════════════════════════════════════════════════════
def team_header_html(team, row):
    colour  = TEAM_COLOURS.get(team, '#2563eb')
    ql, qc  = quadrant_label(row['season_overall_z'], row['std_z_score'])
    mz, sz  = row['season_overall_z'], row['std_z_score']
    nm, lr  = int(row['matches_included']), fluidity_rank(team)
    zc      = mv_class(mz)
    return f"""
    <div style='display:flex;align-items:center;gap:0.9rem;margin-bottom:0.6rem;'>
      <div class='team-bar' style='background:{colour};'></div>
      <div>
        <div class='team-name'>{team.upper()}</div>
        <span class='qbadge' style='background:{qc}18;color:{qc};border:1px solid {qc}44;'>{ql}</span>
      </div>
    </div>
    <div class='metric-row'>
      <div class='metric-card'>
        <div class='metric-label'>Season Fluidity Z</div>
        <div class='metric-value {zc}'>{mz:+.3f}</div>
      </div>
      <div class='metric-card'>
        <div class='metric-label'>Fluidity Rank</div>
        <div class='metric-value mv-blue'>{lr} / 20</div>
      </div>
      <div class='metric-card'>
        <div class='metric-label'>Consistency STD Z</div>
        <div class='metric-value mv-blue'>{sz:+.3f}</div>
      </div>
      <div class='metric-card'>
        <div class='metric-label'>Matches Included</div>
        <div class='metric-value mv-blue'>{nm}</div>
      </div>
    </div>
    """

def zone_stats_html(row):
    dz = row['avg_defensive_z']
    mz = row['avg_midfield_z']
    fz = row['avg_forward_z']
    def zc(v): return mv_class(v)
    return f"""
    <div class='metric-row'>
      <div class='zone-card' style='flex:1'>
        <div class='zone-label'>Defensive Z</div>
        <div class='zone-val' style='color:{"#16a34a" if dz>0 else "#dc2626"}'>{dz:+.3f}</div>
      </div>
      <div class='zone-card' style='flex:1'>
        <div class='zone-label'>Midfield Z</div>
        <div class='zone-val' style='color:{"#16a34a" if mz>0 else "#dc2626"}'>{mz:+.3f}</div>
      </div>
      <div class='zone-card' style='flex:1'>
        <div class='zone-label'>Forward Z</div>
        <div class='zone-val' style='color:{"#16a34a" if fz>0 else "#dc2626"}'>{fz:+.3f}</div>
      </div>
    </div>
    """

def player_stats_html(row):
    pz   = row['season_z']
    pos  = row['position_cat']
    rank = int(row['fluidity_rank'])
    zc   = mv_class(pz)
    return f"""
    <div class='metric-row'>
      <div class='metric-card'>
        <div class='metric-label'>Fluidity Z (Season)</div>
        <div class='metric-value {zc}'>{pz:+.3f}</div>
      </div>
      <div class='metric-card'>
        <div class='metric-label'>Position Category</div>
        <div class='metric-value mv-blue' style='font-size:1rem;line-height:1.4'>{pos}</div>
      </div>
      <div class='metric-card'>
        <div class='metric-label'>Position Rank</div>
        <div class='metric-value mv-blue'>{rank}</div>
      </div>
    </div>
    """

def closest_players_html(player_name, player_z, position_cat):
    same = season_player_stats[
        (season_player_stats['position_cat'] == position_cat) &
        (season_player_stats['player']       != player_name)
    ].copy()
    same['z_dist'] = (same['season_z'] - player_z).abs()
    closest = same.sort_values('z_dist').head(3)
    cards = ""
    for _, r in closest.iterrows():
        c = '#16a34a' if r['z_dist'] < 0.1 else '#2563eb'
        cards += (
            f"<div class='pcard' style='border-left-color:{c}'>"
            f"<div class='pcard-name'>{short_name(r['player'])}</div>"
            f"<div class='pcard-meta'>{r['team']} &nbsp;&middot;&nbsp; "
            f"Z: {r['season_z']:+.3f} &nbsp;&middot;&nbsp; "
            f"&Delta; {r['z_dist']:.3f}</div></div>"
        )
    header = f"<div class='section-hdr'>Players With Most Similar Fluidity Profiles in {position_cat.upper()}</div>"
    return f"<div style='margin-top:0.5rem'>{header}{cards}</div>"
# ══════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════
def plot_quadrant(selected_team):
    """
    Quadrant plot from notebook cell 40.
    Selected team highlighted and labelled; all others shown unlabelled.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # quadrant shading (same colours as notebook)
    ax.axhspan(0,  2, xmin=0.5, xmax=1, alpha=0.03, color='orange')
    ax.axhspan(0,  2, xmin=0,   xmax=0.5, alpha=0.03, color='blue')
    ax.axhspan(-2, 0, xmin=0,   xmax=0.5, alpha=0.03, color='red')
    ax.axhspan(-2, 0, xmin=0.5, xmax=1,   alpha=0.03, color='green')

    x_max = avg_team_fluidity['std_z_score'].abs().max()
    y_max = avg_team_fluidity['season_overall_z'].abs().max()

    for _, row in avg_team_fluidity.iterrows():
        team   = row['team']
        is_sel = (team == selected_team)
        x      = row['std_z_score']
        y      = row['season_overall_z']
        colour = TEAM_COLOURS.get(team, '#333333')

        ax.scatter(x, y,
                   s=220 if is_sel else 100,
                   color=colour,
                   edgecolors='black',
                   linewidth=2.5 if is_sel else 1.2,
                   zorder=5 if is_sel else 3)

        # only label selected team
        if is_sel:
            offset_x, offset_y = LABEL_OFFSETS.get(team, (0, 8))
            ax.annotate(
                TEAM_ABBR.get(team, team),
                xy=(x, y),
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                fontsize=10, fontweight='bold',
                ha='center', zorder=6
            )

    ax.axvline(0, color='black', linestyle='--', alpha=0.2, linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.2, linewidth=2)
    ax.set_xlim(-x_max * 1.15, x_max * 1.15)
    ax.set_ylim(-y_max * 1.15, y_max * 1.15)

    ax.set_xlabel('Tactical Consistency (Z-Score)\n← More Consistent | Less Consistent →',
                  fontsize=10, fontweight='bold')
    ax.set_ylabel('Positional Fluidity (Z-Score)\n← More Rigid | More Fluid →',
                  fontsize=10, fontweight='bold')
    ax.set_title('Team Tactical Profiles: 2015-16 Premier League',
                 fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.2, linestyle=':', zorder=0)

    for txt, tx, ty in [
        ('Inconsistently Fluid',  x_max*.68,  y_max*.88),
        ('Consistently Fluid',   -x_max*.68,  y_max*.88),
        ('Inconsistently Rigid',  x_max*.68,  -y_max*.88),
        ('Consistently Rigid',   -x_max*.68,  -y_max*.88),
    ]:
        ax.text(tx, ty, txt, fontsize=8, style='italic', alpha=0.55, ha='center')

    plt.tight_layout()
    return fig


def plot_line(selected_team):
    """
    Match-by-match fluidity line chart from notebook cell 41 style.
    """
    team_data = (team_fluidity_full[team_fluidity_full['team'] == selected_team]
                 .sort_values('match_id').reset_index(drop=True))

    colour      = 'steelblue'
    season_mean = team_data['season_overall_z'].iloc[0]
    scores      = team_data['overall_z'].values
     x = np.arange(1, len(scores) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, scores, color=colour, linewidth=1.8, marker='o', markersize=4, zorder=3)
    ax.axhline(season_mean, color='red',   linewidth=1.5, linestyle='--', label='Team Average', zorder=2)
    ax.axhline(0,           color='black', linewidth=1.5, linestyle='--', alpha=0.4, label='League Average', zorder=1)

    ax.set_xlabel('Match Number', fontsize=10, fontweight='bold')
    ax.set_ylabel('Fluidity Z-Score', fontsize=10, fontweight='bold')
    ax.set_title('Match-by-Match Fluidity Z-Scores', fontsize=11, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9, frameon=False)

    ax.set_ylim(-2, 2)
    ax.set_xlim(1,38)
    plt.tight_layout()
    return fig

def plot_zone_bar(selected_team):
    league_avg = {
        'Defensive': avg_team_fluidity['avg_defensive_z'].mean(),
        'Midfield':  avg_team_fluidity['avg_midfield_z'].mean(),
        'Forward':   avg_team_fluidity['avg_forward_z'].mean(),
    }
    team_row = avg_team_fluidity[avg_team_fluidity['team'] == selected_team].iloc[0]
    team_vals = {
        'Defensive': team_row['avg_defensive_z'],
        'Midfield':  team_row['avg_midfield_z'],
        'Forward':   team_row['avg_forward_z'],
    }

    zones  = ['Defensive', 'Midfield', 'Forward']
    colour = TEAM_COLOURS.get(selected_team, '#2563eb')
    abbr   = TEAM_ABBR.get(selected_team, selected_team)
    x      = np.arange(len(zones))

    fig, ax = plt.subplots(figsize=(7, 5))

    # bar colour: green if above league avg, red if below
    bar_colours = [
        '#16a34a' if team_vals[z] >= league_avg[z] else '#dc2626'
        for z in zones
    ]

    bars = ax.bar(x, [team_vals[z] for z in zones],
                  width=0.5, color=bar_colours,
                  edgecolor='black', linewidth=0.8, alpha=0.85,
                  label=abbr, zorder=3)

    # league average as a dot/marker per zone
    for i, zone in enumerate(zones):
        ax.plot(i, league_avg[zone], marker='D', markersize=9,
                color='black', zorder=5,
                label='Zonal League Average' if i == 0 else '')
        ax.plot([i - 0.28, i + 0.28], [league_avg[zone], league_avg[zone]],
                color='black', linewidth=2, zorder=4)

    # value labels on bars
    for bar, zone in zip(bars, zones):
        h = bar.get_height()
        # position label above bar if positive, below if negative
        y_pos = h + 0.003 if h >= 0 else h - 0.003
        va = 'bottom' if h >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2,
                y_pos, f'{h:+.2f}',
                ha='center', va=va,
                fontsize=9, fontweight='bold')

    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(zones, fontsize=10, fontweight='bold')
    ax.set_ylabel('Average Z-Score', fontsize=10, fontweight='bold')
    ax.set_title(' Team Zonal Fluidity vs League Averages', fontsize=11, fontweight='bold', pad=8)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    plt.tight_layout()
    return fig
 

def plot_touch_map(player_id, player_name, position_cat):
    """
    Touch map from notebook cell 27 style:
    white pitch, black lines, Reds KDE, title 'Name | z = x.xx'
    """
    player_locs = valid_touches[
        (valid_touches['player_id']    == player_id) &
        (valid_touches['position_cat'] == position_cat)
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    pitch = Pitch(pitch_type='statsbomb', line_color='black')
    pitch.draw(ax=ax)

    if len(player_locs) > 0:
        pitch.kdeplot(
            player_locs['x'], player_locs['y'],
            ax=ax, cmap='Reds', fill=True,
            bw_method=0.3, levels=20, thresh=0.1, alpha=0.8
        )

    player_z = season_player_stats[
        (season_player_stats['player_id']    == player_id) &
        (season_player_stats['position_cat'] == position_cat)
    ]['season_z'].values[0]

    ax.set_title(f'{short_name(player_name)} | z = {player_z:.3f}',
                 fontsize=12, fontweight='bold', pad=4)
    plt.tight_layout()
    return fig


def plot_swarm(selected_player, position_cat):
    """
    Swarm plot from notebook cell 28 style, filtered to selected position_cat.
    No player labels. Selected player highlighted as gold dot.
    """
    pos_data = season_player_stats[
        season_player_stats['position_cat'] == position_cat
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.swarmplot(
        data=pos_data, x='position_cat', y='season_z',
        hue='position_cat', size=8, alpha=0.7, ax=ax
    )

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.25, alpha=0.4)

    # highlight selected player as gold dot — no label
    sel_row = pos_data[pos_data['player'] == selected_player]
    if len(sel_row) > 0:
        sel_z = sel_row['season_z'].values[0]
        ax.scatter(0, sel_z, s=200, color='gold',
                   edgecolors='black', linewidth=2, zorder=10)


    ax.set_ylim(-1.5,1.5)

    ax.set_ylabel('Fluidity Z-Score',  fontsize=11, fontweight='bold')
    ax.set_title(f'{position_cat} — Fluidity Z Scores',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.4)

    legend = ax.get_legend()
    if legend:
        legend.remove()

    ax.set_xlabel('')
    ax.set_xticklabels([])

    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div class='dash-title'>FLUIDITY</div>", unsafe_allow_html=True)
    st.markdown("<div class='dash-sub'>2015-16 Premier League</div>", unsafe_allow_html=True)
    st.markdown("---")

    selected_team = st.selectbox(
        "SELECT Team",
        options=sorted(avg_team_fluidity['team'].unique())
    )

    # players for this team meeting threshold
    team_players = (season_player_stats[
        season_player_stats['team'] == selected_team
    ].sort_values(['player', 'season_z'], ascending=[True, False]))

    player_options = [
        (f"{r['player']}  ({r['position_cat']})", r['player'], r['position_cat'])
        for _, r in team_players.iterrows()
    ]
    player_labels = [p[0] for p in player_options]

    st.markdown("---")
    selected_label = st.selectbox("SELECT PLAYER", options=player_labels)
    selected_idx      = player_labels.index(selected_label)
    selected_player   = player_options[selected_idx][1]
    selected_position = player_options[selected_idx][2]

# ══════════════════════════════════════════════════════════════
# MAIN — TITLE
# ══════════════════════════════════════════════════════════════
st.markdown("<div class='dash-title'>POSITIONAL FLUIDITY DASHBOARD- PLAYER AND TEAM LEVEL METRICS</div>", unsafe_allow_html=True)
st.markdown("<div class='dash-sub'>2015-2016 PREMIER LEAGUE SEASON </div>",
            unsafe_allow_html=True)

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    selected_team = st.selectbox(
        "SELECT TEAM",
        options=sorted(avg_team_fluidity['team'].unique())
    )

team_players = (season_player_stats[
    season_player_stats['team'] == selected_team
].sort_values(['player', 'season_z'], ascending=[True, False]))

player_options = [
    (f"{r['player']}  ({r['position_cat']})", r['player'], r['position_cat'])
    for _, r in team_players.iterrows()
]
player_labels = [p[0] for p in player_options]

with col_sel2:
    selected_label = st.selectbox("SELECT PLAYER", options=player_labels)

selected_idx      = player_labels.index(selected_label)
selected_player   = player_options[selected_idx][1]
selected_position = player_options[selected_idx][2]
# ══════════════════════════════════════════════════════════════
# SECTION 1 — TEAM VIEW
# ══════════════════════════════════════════════════════════════
st.markdown("<div class='section-hdr'>TEAM OVERVIEW</div>", unsafe_allow_html=True)

team_row = avg_team_fluidity[avg_team_fluidity['team'] == selected_team].iloc[0]
st.markdown(team_header_html(selected_team, team_row), unsafe_allow_html=True)

# quadrant + line chart
col_q, col_l = st.columns(2)
with col_q:
    st.pyplot(plot_quadrant(selected_team))
    plt.close('all')
with col_l:
    st.pyplot(plot_line(selected_team))
    plt.close('all')

# ── zone stats ────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>ZONE FLUIDITY</div>", unsafe_allow_html=True)
st.markdown(zone_stats_html(team_row), unsafe_allow_html=True)

col_bar, col_space = st.columns([1.4, 1])
with col_bar:
    st.pyplot(plot_zone_bar(selected_team))
    plt.close('all')
with col_space:
    st.markdown("""
    <div style='padding:1.2rem 1.4rem;background:#f8f9fa;border:1px solid #e0e0e0;
                border-radius:8px;font-size:1.25rem;color:#555;line-height:2.25;
                margin-top:0.5rem;'>
    <b>Zonal Interpretation</b><br>
    Positive Z = More positional variance than league average.<br>
    Negative Z = Less positional fluidity than league average.<br><br>
    <b>Defensive</b> — Centre backs &amp; full-backs<br>
    <b>Midfield</b> — Central, attacking, defensive &amp; wide midfielders<br>
    <b>Forward</b> — Strikers &amp; wide forwards
    </div>
    """, unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════
# SECTION 2 — PLAYER VIEW
# ══════════════════════════════════════════════════════════════
st.markdown("<div class='section-hdr'>PLAYER ANALYSIS</div>", unsafe_allow_html=True)

player_row = season_player_stats[
    (season_player_stats['player']       == selected_player) &
    (season_player_stats['position_cat'] == selected_position)
].iloc[0]

st.markdown(player_stats_html(player_row), unsafe_allow_html=True)

# touch map + swarm
col_t, col_s = st.columns(2)
with col_t:
    st.markdown("<div class='section-hdr'>TOUCH MAP</div>", unsafe_allow_html=True)
    st.pyplot(plot_touch_map(player_row['player_id'], selected_player, selected_position))
    plt.close('all')

with col_s:
    st.markdown("<div class='section-hdr'>Ranking within Position Category</div>", unsafe_allow_html=True)
    st.pyplot(plot_swarm(selected_player, selected_position))
    plt.close('all')

# 3 closest players
st.markdown(
    closest_players_html(selected_player, player_row['season_z'], selected_position),
    unsafe_allow_html=True
)
