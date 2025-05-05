import argparse

################################################
#               IMPORTANT                      #
################################################
# 1. Do not print anything other than the ranked list of papers.
# 2. Do not forget to remove all the debug prints while submitting.




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-paper-title", type=str, required=True)
    parser.add_argument("--test-paper-abstract", type=str, required=True)
    args = parser.parse_args()

    # print(args)

    ################################################
    #               YOUR CODE START                #
    ################################################

    from sentence_transformers import SentenceTransformer
    feature_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Encode the title and abstract
    text = args.test_paper_title + " " + args.test_paper_abstract
    embedding = feature_model.encode(text)  # This returns numpy array instead of tensor

    # Load saved files
    import torch
    import pickle
    with open("feature_tensor.pkl", "rb") as f:
        features_array = pickle.load(f)
        features_tensor = torch.from_numpy(features_array)

    with open("edge_index.pkl", "rb") as f:
        edge_index_array = pickle.load(f)
        edge_index = torch.from_numpy(edge_index_array)

    with open("idx_to_node.pkl", "rb") as f:
        idx_to_node = pickle.load(f)
    
    with open('titles.pkl', 'rb') as f:
        titles = pickle.load(f)
    
    # Load the model
    from model import GraphSAGE
    model = GraphSAGE(in_channels=768, hidden_channels=128, out_channels=64, dropout=0.5)
    
    # Determine device and ensure all tensors are on the same device
    device = torch.device('cpu')
    model.load_state_dict(torch.load('graph_sage_model.pth', map_location=device))
    model = model.to(device)
    features_tensor = features_tensor.to(device)
    edge_index = edge_index.to(device)
    
    # Convert embedding to tensor and add to features
    embedding_tensor = torch.from_numpy(embedding).unsqueeze(0).to(device)
    features_tensor = torch.cat([features_tensor, embedding_tensor], dim=0)
    # ensure the last embedding is the query paper
    features_tensor[-1] = embedding_tensor.squeeze(0)
    # print(features_tensor.shape)
    # Add query paper to idx_to_node mapping
    idx_to_node[len(idx_to_node)] = "query_paper"
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        h_all = model(features_tensor, edge_index)
        query_feature = h_all[-1].unsqueeze(0)
        query_feature = query_feature.repeat(h_all.size(0), 1)
        all_embeddings = torch.cat([query_feature, h_all], dim=1)
        scores = model.link_pred(all_embeddings).squeeze()
        
        # Make a sorted list of papers based on the scores
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_papers = [idx_to_node[idx.item()] for idx in sorted_indices]
    
    # Get the top 100 recommended papers
    result = sorted_papers
    


    # prepare a ranked list of papers like this:
    # result = ['paper1', 'paper2', 'paper3', 'paperK']  # Replace with your actual ranked list


    ################################################
    #               YOUR CODE END                  #
    ################################################


    
    ################################################
    #               DO NOT CHANGE                  #
    ################################################
    print('\n'.join(result))

if __name__ == "__main__":
    main()