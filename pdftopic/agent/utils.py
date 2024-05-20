from llama_index.core.schema import TextNode


def add_topics_to_nodes(nodes, labels, probs):

    labelled_splits = [s for s in nodes]
    for i, s in enumerate(labelled_splits):
        s.metadata["topic_label"] = labels[i]
        s.metadata["topic_probs"] = str(probs[i])
    return labelled_splits


def add_metadata_to_nodes(nodes, metadata_dict):

    labelled_splits = [s for s in nodes]
    for i, s in enumerate(labelled_splits):
        # add the metadata information to each node with the given id
        if s.metadata["file_id"] in metadata_dict:
            for k, v in metadata_dict[s.metadata["file_id"]].items():
                s.metadata[k] = v
    return labelled_splits


def get_number_pages(nodes):

    npages = 0
    for node in nodes:
        try:
            page_number = int(node.metadata["page_label"])
        except:
            page_number = -float("inf")
        npages = max(npages, page_number)
    return npages


def get_metadata_node(metadata_to_add):

    text = "Here we describe the metadata for this document, which can be used to answer questions about the document's topics, title, authers or history"
    for k, v in metadata_to_add.items():
        text += "\n" + k + ": " + str(v)

    node_metadata = metadata_to_add.copy()
    node_metadata["page_label"] = "undefined"
    node = TextNode(text=text)
    node.metadata = node_metadata
    return node
