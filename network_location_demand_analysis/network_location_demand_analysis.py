import os
import pandas as pd
from clickhouse_connect import get_client
import networkx as nx

# -----------------------------
# ClickHouse Configuration
# -----------------------------
CLICKHOUSE_CONFIG = {
    "host": "localhost",
    "port": 8123,
    "username": "ziadashraf98765",
    "password": "x5x6x7x8",
    "database": "data_warehouse",
}

OUTPUT_DIR = "network_location_demand_analysis/network_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------
def load_trip_data():
    print("🔄 Loading trip data from ClickHouse...")
    client = get_client(**CLICKHOUSE_CONFIG)
    query = """
        SELECT *
        FROM trip_location_network_metrics
    """
    df = client.query_df(query)
    print(f"✅ Loaded {len(df)} records from ClickHouse.")
    return df


# -----------------------------
# Build Graph
# -----------------------------
def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    print("🔄 Building graph from DataFrame...")
    G = nx.from_pandas_edgelist(
        df,
        source="source_location",
        target="target_location",
        edge_attr=[
            "trip_count",
            "avg_distance",
            "sum_total_amounts",
            "avg_duration_minutes",
        ],
        create_using=nx.DiGraph(),
    )
    print(f"✅ Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


# -----------------------------
# Network Analysis Models
# -----------------------------
def analyze_network(G: nx.DiGraph) -> dict:
    """Compute different centrality and network metrics."""
    results = {}

    print("🔄 Calculating Degree Centrality...")
    in_centrality = nx.in_degree_centrality(G)
    out_centrality = nx.out_degree_centrality(G)
    centrality_df = pd.DataFrame({
        "in_degree": in_centrality,
        "out_degree": out_centrality
    })
    centrality_df["asymmetry"] = centrality_df["in_degree"] - centrality_df["out_degree"]
    results["degree_centrality"] = centrality_df.sort_values("in_degree", ascending=False)

    print("🔄 Calculating Betweenness Centrality...")
    betweenness = nx.betweenness_centrality(G, weight="avg_distance")
    results["betweenness"] = pd.DataFrame.from_dict(
        betweenness, orient="index", columns=["betweenness"]
    ).sort_values("betweenness", ascending=False)

    print("🔄 Calculating Closeness Centrality...")
    closeness = nx.closeness_centrality(G, distance="avg_distance")
    results["closeness"] = pd.DataFrame.from_dict(
        closeness, orient="index", columns=["closeness"]
    ).sort_values("closeness", ascending=False)

    print("🔄 Calculating PageRank...")
    pagerank = nx.pagerank(G, weight="trip_count")
    results["pagerank"] = pd.DataFrame.from_dict(
        pagerank, orient="index", columns=["pagerank"]
    ).sort_values("pagerank", ascending=False)

    print("✅ Network analysis complete.\n")
    return results


# -----------------------------
# Save Results
# -----------------------------
def save_results(results: dict):
    print("💾 Saving results to CSV...")
    for name, df in results.items():
        filename = os.path.join(OUTPUT_DIR, f"{name}.csv")
        df.to_csv(filename, index=True)
        print(f"   ✔ Saved {name} results ({len(df)} rows) → {filename}")
    print("✅ All results saved.\n")


# -----------------------------
# Reporting
# -----------------------------
def report_results(results: dict):
    print("=== Top 10 Inbound Hubs (Trip Attractors) ===")
    print(results["degree_centrality"].sort_values("in_degree", ascending=False).head(10), "\n")

    print("=== Top 10 Outbound Hubs (Trip Generators) ===")
    print(results["degree_centrality"].sort_values("out_degree", ascending=False).head(10), "\n")

    print("=== Top 10 Net Attractors (Inbound - Outbound) ===")
    print(results["degree_centrality"].sort_values("asymmetry", ascending=False).head(10), "\n")

    print("=== Top 10 Bottleneck Locations (Betweenness Centrality) ===")
    print(results["betweenness"].head(10), "\n")

    print("=== Top 10 Accessible Locations (Closeness Centrality) ===")
    print(results["closeness"].head(10), "\n")

    print("=== Top 10 Influential Locations (PageRank by Trip Count) ===")
    print(results["pagerank"].head(10), "\n")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    df = load_trip_data()
    G = build_graph(df)

    results = analyze_network(G)

    # Print top results to console
    report_results(results)

    # Save full results as CSVs
    save_results(results)
