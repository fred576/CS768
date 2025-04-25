import os
import re
import json

def parse_bbl(file_path):
    citations = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            bibitems = content.split("\\bibitem")
            for bibitem in bibitems[1:]:
                matches = re.findall(r'\[(.*?),\s*(.*?)\]\[.*?\]\{(.*?)\}', bibitem)
                if matches:
                    for author, year, key in matches:
                        title_match = re.search(r'\\newblock\s+(.*?)(?:\\newblock|$)', bibitem, re.DOTALL)
                        title = title_match.group(1).strip() if title_match else "Title not found"
                        cleaned_title = title.replace("\\", "").replace("{", "").replace("}", "").upper()
                        citations.append({
                            "key": key.strip(),
                            "author": author.strip(),
                            "year": year.strip(),
                            "title": cleaned_title,
                            "source": "bbl"
                        })
                else:
                    matches = re.findall(r'\[(.*?)\]\{(.*?)\}', bibitem)
                    if matches:
                        for authors_year, key in matches:
                            year_match = re.search(r'\((\d{4})\)', authors_year)
                            authors = authors_year if not year_match else authors_year[:authors_year.find(year_match.group(0))].strip()
                            year = year_match.group(1) if year_match else "Year not found"
                            title_match = re.search(r'\\newblock\s+(.*?)(?:\\newblock|$)', bibitem, re.DOTALL)
                            title = title_match.group(1).strip() if title_match else "Title not found"
                            cleaned_title = title.replace("\\", "").replace("{", "").replace("}", "").upper()
                            citations.append({
                                "key": key.strip(),
                                "author": authors.strip(),
                                "year": year.strip(),
                                "title": cleaned_title,
                                "source": "bbl"
                            })
                    else:
                        title_match = re.search(r'\\newblock\s+(.*?)(?:\\newblock|$)', bibitem, re.DOTALL)
                        title = title_match.group(1).strip() if title_match else "Title not found"
                        cleaned_title = title.replace("\\", "").replace("{", "").replace("}", "").upper()
                        key = bibitem.split("{")[1].split("}")[0]
                        author = bibitem.split("\\newblock")[0].split("}")[-1].strip()
                        year = bibitem.split("\\newblock")[-1].split(", ")[-1].strip()
                        citations.append({
                            "key": key.strip(),
                            "author": author.strip(),
                            "year": year.strip(),
                            "title": cleaned_title,
                            "source": "bbl"
                        })
    except Exception as e:
        print(f"Error parsing .bbl file {file_path}: {e}")
    return citations

def parse_bib(file_path):
    citations = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            bibitems = content.split("@")
            for bibitem in bibitems[1:]:
                key = bibitem.split("{")[1].split(",")[0].strip()
                author_match = re.search(r'author\s*=\s*{(.*?)}', bibitem, re.DOTALL)
                year_match = re.search(r'year\s*=\s*{(.*?)}', bibitem, re.DOTALL)
                title_match = re.search(r'title\s*=\s*{(.*?)}', bibitem, re.DOTALL)
                doi_match = re.search(r'doi\s*=\s*{(.*?)}', bibitem, re.DOTALL)
                author = author_match.group(1).strip() if author_match else "Author not found"
                year = year_match.group(1).strip() if year_match else "Year not found"
                title = title_match.group(1).strip() if title_match else "Title not found"
                doi = doi_match.group(1).strip() if doi_match else "DOI not found"
                cleaned_title = title.replace("\\", "").replace("{", "").replace("}", "").upper()
                citations.append({
                    "key": key.strip(),
                    "author": author.strip(),
                    "year": year.strip(),
                    "doi": doi.strip(),
                    "title": cleaned_title,
                    "source": "bib"
                })
    except Exception as e:
        # print(bibitem)
        print(f"Error parsing .bib file {file_path}: {e}")
    return citations

def process_dataset(dataset_path, output_json_path):
    all_data = []

    for paper_id in os.listdir(dataset_path):
        paper_path = os.path.join(dataset_path, paper_id)
        if not os.path.isdir(paper_path):
            continue
        print(f"Processing {paper_id}...")
        citation_list = []
        for file in os.listdir(paper_path):
            file_path = os.path.join(paper_path, file)
            if file.endswith(".bbl"):
                citation = parse_bbl(file_path)
                if len(citation) == 0:
                    print(f"Warning: No citations found in {file_path}")
                else:
                    citation_list.extend(citation)
            elif file.endswith(".bib"):
                citation = parse_bib(file_path)
                if len(citation) == 0:
                    print(f"Warning: No citations found in {file_path}")
                else:
                    citation_list.extend(citation)

        if citation_list:
            all_data.append({
                "paper_id": paper_id,
                "citations": citation_list
            })

    with open(output_json_path, "w", encoding="utf-8") as out_file:
        json.dump(all_data, out_file, indent=2)
    print(f"Saved {len(all_data)} papers with citations to {output_json_path}")

if __name__ == "__main__":
    dataset_folder = "dataset_papers/dataset_papers/"  # 👈 Replace with your actual dataset path
    output_file = "parsed_citations.json"
    process_dataset(dataset_folder, output_file)
