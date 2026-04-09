
### Title: "GAE process" ####
### Author: Patrón-Rivero, C. ####
### Date: 02/March/2026 ###
### Project: "Graph autoencoders and community detection algorithms to improve polymorphic identification" ###


# ------------------------------------------------------------------------------------------------ #

# Libraries #

# ------------------------------------------------------------------------------------------------ #

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import roc_auc_score
from sklearn.manifold import trustworthiness
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------------------------------------------ #

# Inputs #

# ------------------------------------------------------------------------------------------------ #

RUTA_ARCHIVO = r"D:\1_centrality\data\morfo\P_mor.csv"

RASGOS = [
    'TL', 'SVL', 'TAL', 'HL', 'HW', 'HH', 'HE', 'HP', 'HN',
    'MBW', 'MBH', 'CW', 'CH', 'Ven', 'SC', 'MID', 'PO',
    'SO', 'SL', 'IL', 'IOS'
]

N_NEIGHBORS_GRAPH = 4
N_NEIGHBORS_IMPUTER = 3
EPOCHS = 800
LR = 0.008
STEP_SIZE = 250
GAMMA = 0.5
LATENT_DIMS = list(range(1, 22))

# ------------------------------------------------------------------------------------------------ #

# Data Preparation #

# ------------------------------------------------------------------------------------------------ #

df = pd.read_csv(RUTA_ARCHIVO)

X_imputed = KNNImputer(n_neighbors=N_NEIGHBORS_IMPUTER).fit_transform(df[RASGOS].values)
X_scaled = StandardScaler().fit_transform(X_imputed)

A_sparse = kneighbors_graph(
    X_scaled,
    n_neighbors=N_NEIGHBORS_GRAPH,
    mode='connectivity',
    include_self=True
)

A = np.maximum(A_sparse.toarray(), A_sparse.toarray().T)

x_tensor = torch.FloatTensor(X_scaled)
adj_tensor = torch.FloatTensor(A)

pos_weight = torch.tensor(
    [(A.shape[0]**2 - A.sum()) / A.sum()],
    dtype=torch.float32
)

dist_orig = pdist(X_scaled)

# ------------------------------------------------------------------------------------------------ #

# Model Definition #

# ------------------------------------------------------------------------------------------------ #

class GraphConv(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)

    def forward(self, x, adj):
        return self.linear(torch.matmul(adj, x))


class SuperDeepGAE(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.gc1 = GraphConv(in_dim, 128)
        self.gc2 = GraphConv(128, 64)
        self.gc3 = GraphConv(64, latent_dim)
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.15)

    def encode(self, x, adj):
        h = self.dropout(F.elu(self.ln1(self.gc1(x, adj))))
        h = self.dropout(F.elu(self.ln2(self.gc2(h, adj))))
        return self.gc3(h, adj)

    def decode(self, z):
        return torch.matmul(z, z.t())

    def forward(self, x, adj):
        z = self.encode(x, adj)
        return z, self.decode(z)

# ------------------------------------------------------------------------------------------------ #

# Integrated Dimensional Analysis #

# ------------------------------------------------------------------------------------------------ #

results = []

for d in LATENT_DIMS:

    model = SuperDeepGAE(len(RASGOS), d)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        z, logits = model(x_tensor, adj_tensor)
        loss = criterion(logits.view(-1), adj_tensor.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        z_np = model.encode(x_tensor, adj_tensor).numpy()
        probs = torch.sigmoid(model.decode(torch.tensor(z_np))).numpy().flatten()

    auc = roc_auc_score(A.flatten(), probs)
    trust = trustworthiness(X_scaled, z_np, n_neighbors=5)
    dist_corr, _ = pearsonr(dist_orig, pdist(z_np))

    results.append({
        'Dim': d,
        'AUC-ROC': auc,
        'Trust': trust,
        'Dist_Corr': dist_corr,
        'Loss': loss.item()
    })

# ------------------------------------------------------------------------------------------------ #

# Final Results Table #

# ------------------------------------------------------------------------------------------------ #

res_df = pd.DataFrame(results)

print(res_df.to_string(
    index=False,
    formatters={
        'AUC-ROC': '{:.4f}'.format,
        'Trust': '{:.4f}'.format,
        'Dist_Corr': '{:.4f}'.format,
        'Loss': '{:.4f}'.format
    }
))

# ------------------------------------------------------------------------------------------------ #

# Graph Topological Analysis #

# ------------------------------------------------------------------------------------------------ #

import networkx as nx

with torch.no_grad():
    z_final = modelo.encode(x_tensor, adj_tensor)
    A_reconstructed = torch.sigmoid(modelo.decode(z_final)).numpy()

A_reconstructed_binary = (A_reconstructed > 0.5).astype(int)
np.fill_diagonal(A_reconstructed_binary, 0)

G = nx.from_numpy_array(A_reconstructed_binary)
G.remove_edges_from(nx.selfloop_edges(G))

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
avg_degree = sum(dict(G.degree()).values()) / num_nodes
density = nx.density(G)
connected_components = nx.number_connected_components(G)

print("--- RESULTADOS TOPOLÓGICOS ---")
print(f"Nodes: {num_nodes}")
print(f"Edges: {num_edges}")
print(f"Average degree: {avg_degree:.2f}")
print(f"Graph density: {density:.4f}")
print(f"Connected components: {connected_components}")

# ------------------------------------------------------------------------------------------------ #

# Optimal GAE – Final Training and Visualization #

# ------------------------------------------------------------------------------------------------ #

OPTIMAL_DIM = 8
RUTA_ARCHIVO = r"D:\1_centrality\data\morfo\P_mor.csv"
RUTA_FIGS = r"D:\1_met_commun\Figs"

os.makedirs(RUTA_FIGS, exist_ok=True)

RASGOS = [
    'TL', 'SVL', 'TAL', 'HL', 'HW', 'HH', 'HE', 'HP', 'HN',
    'MBW', 'MBH', 'CW', 'CH', 'Ven', 'SC', 'MID', 'PO',
    'SO', 'SL', 'IL', 'IOS'
]

# ------------------------------------------------------------------------------------------------ #

# Data Preparation #

# ------------------------------------------------------------------------------------------------ #

df = pd.read_csv(RUTA_ARCHIVO)

X_raw = df[RASGOS].values
X_imputed = KNNImputer(n_neighbors=3).fit_transform(X_raw)
X_scaled = StandardScaler().fit_transform(X_imputed)

A_sparse = kneighbors_graph(
    X_scaled,
    n_neighbors=4,
    mode='connectivity',
    include_self=True
)

A = np.maximum(A_sparse.toarray(), A_sparse.toarray().T)

x_tensor = torch.FloatTensor(X_scaled)
adj_tensor = torch.FloatTensor(A)

pos_weight = torch.tensor(
    [(A.shape[0]**2 - A.sum()) / A.sum()],
    dtype=torch.float32
)

# ------------------------------------------------------------------------------------------------ #

# Model Definition #

# ------------------------------------------------------------------------------------------------ #

class GraphConv(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)

    def forward(self, x, adj):
        return self.linear(torch.matmul(adj, x))


class SuperDeepGAE(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.gc1 = GraphConv(in_dim, 128)
        self.gc2 = GraphConv(128, 64)
        self.gc3 = GraphConv(64, latent_dim)
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.15)

    def encode(self, x, adj):
        h = F.elu(self.ln1(self.gc1(x, adj)))
        h = self.dropout(h)
        h = F.elu(self.ln2(self.gc2(h, adj)))
        h = self.dropout(h)
        return self.gc3(h, adj)

    def decode(self, z):
        return torch.matmul(z, z.t())

    def forward(self, x, adj):
        z = self.encode(x, adj)
        return z, self.decode(z)

# ------------------------------------------------------------------------------------------------ #

# Final Training #

# ------------------------------------------------------------------------------------------------ #

modelo = SuperDeepGAE(len(RASGOS), OPTIMAL_DIM)

optimizer = torch.optim.Adam(modelo.parameters(), lr=0.008)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

modelo.train()
for _ in range(1000):
    optimizer.zero_grad()
    z, logits = modelo(x_tensor, adj_tensor)
    loss = criterion(logits.view(-1), adj_tensor.view(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()

modelo.eval()
with torch.no_grad():
    z_final = modelo.encode(x_tensor, adj_tensor).numpy()

# ------------------------------------------------------------------------------------------------ #

# Paired Visualization (2x2 Grid) #

# ------------------------------------------------------------------------------------------------ #

especies = df['Species'].unique()
colors = plt.cm.get_cmap('tab10', len(especies))

fig, axs = plt.subplots(2, 2, figsize=(16/2.54, 14/2.54), dpi=600)
axs = axs.flatten()

pares = [(0, 1), (2, 3), (4, 5), (6, 7)]

for i, (d1, d2) in enumerate(pares):
    ax = axs[i]

    for j, esp in enumerate(especies):
        idx = df['Species'] == esp
        ax.scatter(
            z_final[idx, d1],
            z_final[idx, d2],
            label=esp,
            color=colors(j),
            alpha=0.7,
            s=12,
            edgecolors='none'
        )

    ax.set_xlabel(f"Latent dimension {d1+1}", fontsize=8)
    ax.set_ylabel(f"Latent dimension {d2+1}", fontsize=8)
    ax.set_title("")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(labelsize=7)

    if i == 1:
        ax.legend(
            loc='upper right',
            fontsize=6,
            frameon=True,
            title="Species",
            title_fontsize=7,
            markerscale=0.8
        )

plt.tight_layout()

plt.savefig(
    os.path.join(RUTA_FIGS, "Fig_Latent_Space.jpg"),
    bbox_inches='tight',
    dpi=600
)

plt.savefig(
    os.path.join(RUTA_FIGS, "Fig_Latent_Space.pdf"),
    bbox_inches='tight'
)

plt.show()

# ------------------------------------------------------------------------------------------------ #

# Community Detection and Network Visualization (Louvain) #

# ------------------------------------------------------------------------------------------------ #

def draw_cluster_blob(ax, posiciones, nodos, color):
    puntos = np.array([posiciones[n] for n in nodos])

    if len(puntos) >= 3:
        hull = ConvexHull(puntos)
        poligono = Polygon(
            puntos[hull.vertices],
            closed=True,
            fill=True,
            facecolor=color,
            alpha=0.2,
            edgecolor=color,
            lw=2
        )
        ax.add_patch(poligono)

    elif len(puntos) == 2:
        ax.plot(
            puntos[:, 0],
            puntos[:, 1],
            color=color,
            alpha=0.2,
            lw=20,
            solid_capstyle='round'
        )

    elif len(puntos) == 1:
        ax.scatter(
            puntos[:, 0],
            puntos[:, 1],
            color=color,
            alpha=0.2,
            s=1500,
            edgecolors='none'
        )

# ------------------------------------------------------------------------------------------------ #

# Graph Preparation #

# ------------------------------------------------------------------------------------------------ #

G = nx.from_numpy_array(A)
G.remove_edges_from(nx.selfloop_edges(G))

# ------------------------------------------------------------------------------------------------ #

# Louvain Community Detection #

# ------------------------------------------------------------------------------------------------ #

partition = community_louvain.best_partition(G, random_state=42)
df['Community'] = df.index.map(partition)

# ------------------------------------------------------------------------------------------------ #

# Network Visualization #

# ------------------------------------------------------------------------------------------------ #

fig, ax = plt.subplots(figsize=(10, 8), dpi=600)

pos = nx.spring_layout(G, seed=42, k=0.15)

comunidades_unicas = sorted(df['Community'].unique())
colores_com = sns.color_palette("Set3", len(comunidades_unicas))

for i, com in enumerate(comunidades_unicas):

    nodos_com = [n for n in G.nodes() if partition[n] == com]

    draw_cluster_blob(ax, pos, nodos_com, colores_com[i])

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodos_com,
        node_color=[colores_com[i]],
        node_size=30,
        alpha=0.8,
        label=f"Cluster {com}",
        ax=ax
    )

nx.draw_networkx_edges(
    G,
    pos,
    alpha=0.8,
    edge_color='gray',
    ax=ax
)

ax.legend(
    loc='center left',
    bbox_to_anchor=(1, 0.5),
    fontsize=10,
    frameon=False,
    ncol=1,
    title=None
)

ax.set_axis_off()
plt.tight_layout()

plt.savefig(
    r"D:\1_met_commun\Figs\Network_Clusters.jpg",
    dpi=600,
    bbox_inches='tight'
)

plt.savefig(
    r"D:\1_met_commun\Figs\Network_Clusters.pdf",
    dpi=600,
    bbox_inches='tight'
)

plt.show()

# ------------------------------------------------------------------------------------------------ #

# Modularity Summary #

# ------------------------------------------------------------------------------------------------ #

q_val = community_louvain.modularity(partition, G)

print(f"Clusters detectados: {len(comunidades_unicas)}")
print(f"Modularidad (Q): {q_val:.4f}")

# ------------------------------------------------------------------------------------------------ #

# Contingency Matrix – Morphotipe Validation #

# ------------------------------------------------------------------------------------------------ #

plt.figure(figsize=(12, 8))

contingency = pd.crosstab(
    df['Species'],
    df['Community'],
    normalize='index'
)

sns.heatmap(
    contingency,
    annot=True,
    cmap="YlGnBu",
    fmt=".2f",
    cbar_kws={'label': 'Proportion of Specimens'}
)

plt.title("")
plt.xlabel("Louvain Morphological Cluster")
plt.ylabel("")

plt.tight_layout()

plt.savefig(
    r"D:\1_met_commun\Figs\Contingency_Matrix_GAE.jpg",
    dpi=600,
    bbox_inches='tight'
)

plt.savefig(
    r"D:\1_met_commun\Figs\Contingency_Matrix_GAE.pdf",
    dpi=600,
    bbox_inches='tight'
)

plt.show()

print("--- RESULTADOS DEL ANÁLISIS ---")
print(f"Modularidad (Q): {q_value:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")

# ------------------------------------------------------------------------------------------------ #

# Confidence Ellipse Function #

# ------------------------------------------------------------------------------------------------ #

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)

# ------------------------------------------------------------------------------------------------ #

# Inputs #

# ------------------------------------------------------------------------------------------------ #

metric_cols = ['SVL', 'TAL', 'MBW', 'MBH', 'CW', 'CH', 'HL', 'HW', 'HH', 'HE', 'HP', 'HN']
meristic_cols = ['Ven', 'SC', 'MID', 'IOS', 'SO', 'PO', 'SL', 'IL']
tl_col = 'TL'

# ------------------------------------------------------------------------------------------------ #

# Imputation #

# ------------------------------------------------------------------------------------------------ #

all_cols = metric_cols + meristic_cols + [tl_col]

imputer = KNNImputer(n_neighbors=5, weights="distance")
df_imputed = pd.DataFrame(
    imputer.fit_transform(df[all_cols]),
    columns=all_cols
)

# ------------------------------------------------------------------------------------------------ #

# Allometric Correction #

# ------------------------------------------------------------------------------------------------ #

df_transformed = df_imputed.copy()

for col in metric_cols:
    df_transformed[col] = np.log(df_transformed[col] / df_transformed[tl_col])

for col in meristic_cols:
    df_transformed[col] = np.log(df_transformed[col] + 1)

X_final = df_transformed[metric_cols + meristic_cols]

# ------------------------------------------------------------------------------------------------ #

# Standardization #

# ------------------------------------------------------------------------------------------------ #

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# ------------------------------------------------------------------------------------------------ #

# PCA #

# ------------------------------------------------------------------------------------------------ #

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Principal components retained after allometric correction (95% variance): {pca.n_components_}")

# ------------------------------------------------------------------------------------------------ #

# Optimal K-Means #

# ------------------------------------------------------------------------------------------------ #

best_k = 2
best_score = -1

for k in range(2, 16):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    score = silhouette_score(X_pca, kmeans.fit_predict(X_pca))
    if score > best_score:
        best_score = score
        best_k = k

print(f"Optimal number of clusters: {best_k} (Silhouette Score: {best_score:.4f})")

final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X_pca)

# ------------------------------------------------------------------------------------------------ #

# PCA Visualization with Ellipses #

# ------------------------------------------------------------------------------------------------ #

fig, ax = plt.subplots(figsize=(9, 7))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for cluster in range(best_k):
    idx = cluster_labels == cluster
    x_cluster = X_pca[idx, 0]
    y_cluster = X_pca[idx, 1]

    color = colors[cluster % len(colors)]

    ax.scatter(
        x_cluster,
        y_cluster,
        c=color,
        label=f'Cluster {cluster}',
        alpha=0.7,
        edgecolors='w',
        s=50
    )

    confidence_ellipse(
        x_cluster,
        y_cluster,
        ax,
        n_std=2.0,
        edgecolor=color,
        facecolor=color,
        alpha=0.15
    )

var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100

ax.set_title("")
ax.set_xlabel(f'PC1 ({var_pc1:.1f}%)')
ax.set_ylabel(f'PC2 ({var_pc2:.1f}%)')
ax.legend(title=f"K-Means (k={best_k})", loc='best')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig(r"D:\1_met_commun\Figs\PCA.jpg", dpi=600, bbox_inches='tight')
plt.savefig(r"D:\1_met_commun\Figs\PCA.pdf", dpi=600, bbox_inches='tight')

plt.show()

# ------------------------------------------------------------------------------------------------ #

### End Not Run
