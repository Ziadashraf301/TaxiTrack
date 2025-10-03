# 🚖 Taxi Network Location Demand Analysis

## 📌 Overview

This script analyzes **NYC taxi trip flows between locations** by modeling them as a **directed graph**.

* **Nodes** represent pickup/drop-off locations (e.g., JFK Airport, Times Square).
* **Edges** represent taxi trips between locations, enriched with metrics like **trip count, distance, duration, and total fares**.

The analysis applies **network science techniques** (degree centrality, betweenness, closeness, PageRank) to uncover **demand hubs, bottlenecks, and influential areas**.

---

## ⚙️ Steps Performed

1. **Data Loading**

   * Pulled **61,782 taxi trip records** from ClickHouse.
   * Built a graph with **262 nodes** and **61,782 edges**.

2. **Graph Construction**

   * Each node = location.
   * Each directed edge = trip from pickup → drop-off.

3. **Network Analysis Metrics**

   * **Degree Centrality** → inbound vs outbound trip hubs.
   * **Asymmetry** → net attractors vs net generators.
   * **Betweenness** → bottleneck locations critical for flow.
   * **Closeness** → accessibility of locations within the network.
   * **PageRank** → influential areas weighted by trip counts.

4. **Reporting & Saving**

   * Printed **Top 10 locations** for each metric.
   * Exported full results to CSV for further BI/ML usage.

---

## 🔍 Key Insights

### 1. **Inbound Hubs (Trip Attractors)**

Locations attracting the most inbound trips:

* **JFK Airport**, Times Square, TriBeCa, and Kips Bay.
  ➡️ These are **high-demand destinations** where passengers frequently arrive.

### 2. **Outbound Hubs (Trip Generators)**

Locations generating the most outbound trips:

* **JFK Airport**, Midtown South, Times Square, and Union Square.
  ➡️ These are **high-supply pickup points**, important for driver positioning.

### 3. **Net Attractors (Inbound > Outbound)**

Top net attractors include:

* **Newark Airport, Staten Island neighborhoods, Broad Channel**.
  ➡️ These are areas where **demand inflow exceeds supply outflow**, highlighting **driver shortages**.

### 4. **Bottlenecks (Betweenness Centrality)**

* **Governor’s Island, Great Kills, Astoria Park** appear as structural bottlenecks.
  ➡️ These locations **connect different regions**; disruptions here may affect large portions of the network.

### 5. **Accessible Locations (Closeness Centrality)**

* **NV (unclassified zone), Jamaica Bay, Battery Park, Midtown East** are central in terms of accessibility.
  ➡️ Trips to/from these locations are **efficiently reachable**, making them strategic for **dispatch optimization**.

### 6. **Influential Locations (PageRank by Trip Count)**

* **Upper East Side North & South, Midtown Center, Murray Hill** dominate by weighted trip volume.
  ➡️ These areas are **highly influential for overall trip flow** and **revenue concentration zones**.

---

## 💼 Business Benefits

1. **Driver Allocation & Supply Planning**

   * Place more drivers at **outbound hubs** (e.g., Times Square, JFK) to **reduce passenger wait times**.
   * Redirect drivers toward **net attractors** (e.g., Newark Airport) to **cover underserved demand**.

2. **Pricing & Surge Optimization**

   * Use centrality metrics to **dynamically adjust fares** in bottleneck or highly influential areas.
   * Anticipate **peak demand** in inbound hubs (tourist/business districts).

3. **Urban Mobility Planning**

   * Identify **congested or bottleneck locations** for city planners.
   * Improve **infrastructure & traffic flow** around central hubs like airports and Times Square.

4. **Revenue Maximization**

   * Focus marketing/promotions on **high PageRank zones** where trips are most frequent and valuable.
   * Increase efficiency by **minimizing empty rides** between hubs.

---

✅ This analysis bridges **data science, network theory, and business strategy**, enabling taxi companies and city planners to **optimize demand forecasting, driver supply, and customer experience**.

---