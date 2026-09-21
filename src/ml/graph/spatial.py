# src/ml/graph/spatial.py
"""
Spatial Network Analyzer (NetworkX Graph Analytics)
Builds directed transit graphs from ClickHouse route metrics to discover
transit hubs, corridor bottlenecks, and flow asymmetry across NYC taxi zones.
"""
import os
import time
from typing import Dict, Any, Optional
import pandas as pd
import networkx as nx
from ml.graph.base import BaseGraphAnalyzer
from ml.data.base import BaseNetworkRepository
from ml.data.clickhouse import ClickHouseFeatureRepository
from core.logging import get_logger

logger = get_logger(__name__)


class SpatialNetworkAnalyzer(BaseGraphAnalyzer):
    """
    Constructs and analyzes directed spatial transit networks using NetworkX.
    Computes PageRank, betweenness centrality, degree centrality, and route flow metrics.
    """

    def __init__(self, feature_repo: Optional[BaseNetworkRepository] = None):
        self.repo = feature_repo or ClickHouseFeatureRepository()
        self.graph: Optional[nx.DiGraph] = None
        self.metrics: Dict[str, Any] = {}
        self.summary_df: Optional[pd.DataFrame] = None

    def build_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Construct a directed weighted graph from origin-destination DataFrame.
        """
        logger.info(f"Constructing transit graph from {len(df):,} route corridors...")
        start_time = time.perf_counter()

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
        self.graph = G
        duration = time.perf_counter() - start_time
        logger.info(
            f"Graph constructed in {duration:.3f}s: {G.number_of_nodes()} zones (nodes), "
            f"{G.number_of_edges()} active routes (edges)."
        )
        return G

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive network centrality metrics:
        - PageRank: Transit hub influence weighted by trip volume.
        - In-Degree / Out-Degree: Flow attraction vs generation.
        - Flow Asymmetry: Net attractor score (Inbound - Outbound).
        - Betweenness: Strategic bridging corridors and choke points.
        - Closeness: Zone accessibility.
        """
        if self.graph is None:
            raise RuntimeError("Graph has not been built yet. Call build_graph() first.")

        G = self.graph
        logger.info("Computing network centralities and flow dynamics...")
        start_time = time.perf_counter()

        in_degree = nx.in_degree_centrality(G)
        out_degree = nx.out_degree_centrality(G)
        pagerank = nx.pagerank(G, weight="trip_count", max_iter=200)
        betweenness = nx.betweenness_centrality(G, weight="avg_distance")
        closeness = nx.closeness_centrality(G, distance="avg_distance")

        all_nodes = list(G.nodes())
        records = []
        for node in all_nodes:
            in_d = in_degree.get(node, 0.0)
            out_d = out_degree.get(node, 0.0)
            records.append({
                "zone": node,
                "pagerank": round(pagerank.get(node, 0.0), 6),
                "in_degree_centrality": round(in_d, 4),
                "out_degree_centrality": round(out_d, 4),
                "asymmetry_score": round(in_d - out_d, 4),
                "betweenness_centrality": round(betweenness.get(node, 0.0), 6),
                "closeness_centrality": round(closeness.get(node, 0.0), 4),
            })

        summary = pd.DataFrame(records).sort_values("pagerank", ascending=False).reset_index(drop=True)
        self.summary_df = summary
        duration = time.perf_counter() - start_time

        self.metrics = {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "top_hubs": summary.head(5)[["zone", "pagerank"]].to_dict(orient="records"),
            "top_bottlenecks": summary.sort_values("betweenness_centrality", ascending=False)
            .head(5)[["zone", "betweenness_centrality"]]
            .to_dict(orient="records"),
            "computation_time_seconds": round(duration, 3),
        }

        logger.info(
            f"Network metrics computed in {duration:.2f}s. "
            f"Top Transit Hub: {summary.iloc[0]['zone']} (PageRank: {summary.iloc[0]['pagerank']})"
        )
        return self.metrics

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Return the consolidated zone metrics DataFrame."""
        if self.summary_df is None:
            self.compute_metrics()
        return self.summary_df

    def run(self, pickup_month: Optional[str] = None, output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Execute end-to-end network analysis: query ClickHouse, construct graph,
        compute centralities, and export tabular results.
        """
        df_routes = self.repo.get_network_metrics(pickup_month=pickup_month)
        if df_routes.empty and pickup_month:
            logger.warning(
                f"No route data found for specified month '{pickup_month}'. "
                f"Falling back to overall historical route metrics across all available periods..."
            )
            df_routes = self.repo.get_network_metrics(pickup_month=None)

        if df_routes.empty:
            logger.warning("No route data retrieved from ClickHouse. Graph analysis skipped.")
            return pd.DataFrame()

        self.build_graph(df_routes)
        summary = self.get_summary_dataframe()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            suffix = f"_{pickup_month}" if pickup_month else ""
            out_file = os.path.join(output_dir, f"network_metrics{suffix}.csv")
            summary.to_csv(out_file, index=False)
            logger.info(f"Exported network analysis summary to {out_file}")

        return summary
