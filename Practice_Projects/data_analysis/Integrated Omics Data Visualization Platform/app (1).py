import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from scipy.stats import pearsonr

# Page styling matrix configuration
st.set_page_config(page_title="Multi-Omics Integration Platform", layout="wide")

st.title("🧬 Integrated Multi-Omics Data Visualization Platform")
st.markdown("Connect transcriptomic expressions, proteomic abundances, and biological pathway networks dynamically.")
st.markdown("---")

# Synthetic data generator for rapid environment testing
@st.cache_data
def generate_mock_omics_data():
    np.random.seed(42)
    genes = [f"GENE_{i:03d}" for i in range(1, 51)]
    samples = [f"Tumor_{i:02d}" for i in range(1, 6)] + [f"Normal_{i:02d}" for i in range(1, 6)]

    # Transcriptomics (Log2 CPM)
    rna_data = np.random.normal(loc=8, scale=2, size=(len(genes), len(samples)))
    rna_data[:10, :5] += 2.5 # Simulate dynamic upregulation variations in tumor tissue
    df_rna = pd.DataFrame(rna_data, index=genes, columns=samples)

    # Proteomics (Mass Spec Int)
    prot_data = rna_data * 0.7 + np.random.normal(loc=2, scale=1, size=(len(genes), len(samples)))
    df_prot = pd.DataFrame(prot_data, index=genes, columns=samples)

    # Target pathway configurations
    pathways = {
        "MAPK Signaling": genes[:15],
        "PI3K-Akt Pathway": genes[12:30],
        "Apoptosis": genes[28:40],
        "Cell Cycle": genes[35:50]
    }
    return df_rna, df_prot, pathways

df_rna, df_prot, pathway_dict = generate_mock_omics_data()

# Workspace interactive control handles
st.sidebar.header("⚙️ Workspace Controls")
selected_pathway = st.sidebar.selectbox("Select Functional Pathway Module", list(pathway_dict.keys()))
correlation_threshold = st.sidebar.slider("Cross-Omics Correlation Cutoff (r)", 0.0, 1.0, 0.5, 0.05)

pathway_genes = pathway_dict[selected_pathway]
filtered_rna = df_rna.loc[df_rna.index.isin(pathway_genes)]
filtered_prot = df_prot.loc[df_prot.index.isin(pathway_genes)]
st.sidebar.success(f"Loaded {len(pathway_genes)} target components within {selected_pathway}.")

# MODULE 1: HEATMAP EXPRESSION LAYOUTS
st.header("📊 1. Parallel Expression Profiles (Transcript vs. Protein)")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transcriptomics (mRNA Expression)")
    fig_rna = px.imshow(filtered_rna, labels=dict(x="Samples", y="Genes", color="Log2 CPM"), color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_rna, use_container_width=True)

with col2:
    st.subheader("Proteomics (Protein Abundance)")
    fig_prot = px.imshow(filtered_prot, labels=dict(x="Samples", y="Genes", color="Intensity"), color_continuous_scale="Viridis")
    st.plotly_chart(fig_prot, use_container_width=True)

# MODULE 2: CORRELATION CONCORDANCE AUDIT
st.header("📈 2. Cross-Omics Feature Concordance")
correlation_results = []
for gene in pathway_genes:
    if gene in df_rna.index and gene in df_prot.index:
        r_val, _ = pearsonr(df_rna.loc[gene], df_prot.loc[gene])
        correlation_results.append({"Gene": gene, "Pearson_r": r_val})

df_corr = pd.DataFrame(correlation_results).sort_values(by="Pearson_r", ascending=False)
col_chart, col_data = st.columns([2, 1])

with col_chart:
    fig_bar = px.bar(df_corr, x="Gene", y="Pearson_r", color="Pearson_r", color_continuous_scale="Tropic")
    st.plotly_chart(fig_bar, use_container_width=True)
with col_data:
    st.subheader("Audit Records")
    st.dataframe(df_corr, use_container_width=True, height=280)

# MODULE 3: INTERACTIVE NETWORK VISUALIZATION
st.header("🕸️ 3. Multi-Omics Pathway Topology Network")
G = nx.Graph()
for gene in pathway_genes:
    G.add_node(gene, rna=df_rna.loc[gene].mean(), prot=df_prot.loc[gene].mean())

for idx, row in df_corr.iterrows():
    if abs(row["Pearson_r"]) >= correlation_threshold:
        current_gene = row["Gene"]
        other_genes = df_corr[df_corr["Gene"] != current_gene]["Gene"].head(2).values
        for target in other_genes:
            G.add_edge(current_gene, target, weight=row["Pearson_r"])

pos = nx.spring_layout(G, seed=42)
edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#888'), hoverinfo='none', mode='lines')
node_x, node_y, node_hover, node_color = [], [], [], []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_color.append(G.nodes[node]['rna'])
    node_hover.append(f"🧬 Symbol: {node}<br>Avg RNA: {G.nodes[node]['rna']:.2f}<br>Avg Prot: {G.nodes[node]['prot']:.2f}")

node_trace = go.Scatter(
    x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), textposition="top center", hoverinfo='text', hovertext=node_hover,
    marker=dict(showscale=True, colorscale='YlGnBu', color=node_color, size=24, line_width=2, colorbar=dict(title="Avg RNA Log2 CPM"))
)

fig_network = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
st.plotly_chart(fig_network, use_container_width=True)
