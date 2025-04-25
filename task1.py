import os
import re
import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from difflib import get_close_matches

def title_match(title, all_titles):
    best_match = get_close_matches(title, all_titles.values(), n=1, cutoff=0.6)
    print(f"Best match for {title}: {best_match}")
    if best_match:
        for node_id, node_title in all_titles.items():
            if node_title == best_match[0]:
                return node_id
    return None

def add_edges(G, paper_id, citations, all_titles):
    for citation in citations:
        cited_paper_id = citation["key"]
        cited_paper_title = citation["title"].upper()
        if cited_paper_id in G.nodes:
            G.add_edge(paper_id, cited_paper_id)
            continue
        best_match_id = title_match(cited_paper_title, all_titles)
        if best_match_id:
            G.add_edge(paper_id, best_match_id)
            continue
        else:
            print(f"Warning: No match found for {cited_paper_title} in {paper_id}.")
            G.add_node(cited_paper_id, title=cited_paper_title)
            G.add_edge(paper_id, cited_paper_id)
            all_titles[cited_paper_id] = cited_paper_title

def visualize_graph(G, output_path):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight="bold")
    plt.title("Citation Network")
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    dataset_path = "dataset_papers/dataset_papers"
    json_path = "parsed_citations.json"
    nodes = [os.path.splitext(file)[0] for file in os.listdir(dataset_path)]
    G = nx.DiGraph()

    for paper in os.listdir(dataset_path):
        paper_path = os.path.join(dataset_path, paper)
        title, abstract = None, None
        with open(os.path.join(paper_path, "title.txt"), "r", encoding="utf-8") as f:
            title = f.read().strip()
        with open(os.path.join(paper_path, "abstract.txt"), "r", encoding="utf-8") as f:
            abstract = f.read().strip()
        G.add_node(paper, title=title, abstract=abstract)

    all_titles = {node[0]: node[1]["title"].upper() for node in G.nodes(data=True)}
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        for paper in json_data:
            paper_id = paper["paper_id"]
            citations = paper["citations"]
            add_edges(G, paper_id, citations, all_titles)

    visualize_graph(G, "citation_network.png")




