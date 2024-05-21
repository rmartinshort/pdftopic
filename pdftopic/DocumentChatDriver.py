from pdftopic.topic_modelling.DocumentTopicSummarizer import DocumentTopicSummarizer
from pdftopic.agent.DocumentAgent import DocumentAgent
from pdftopic.agent.utils import (
    add_metadata_to_nodes,
    add_topics_to_nodes,
    get_number_pages,
    get_metadata_node,
)
from typing import List
from collections import defaultdict
from llama_index.core.schema import TextNode


class DocumentChatDriver:

    def __init__(
        self,
        api_key: str,
        chat_model_name: str = "gpt-4o",
        chat_temperature: int = 0,
        persist: bool = False,
    ) -> None:

        self.chat_model_name = chat_model_name
        self.api_key = api_key
        self.chat_temperature = chat_temperature
        self.summarizer = DocumentTopicSummarizer(openai_api_key=api_key)
        self.persist = persist
        self.chat_model = None

    def search_for_files(self) -> None:

        return self.summarizer.search_for_files()

    def generate_splits(self, files_list: List[dict]) -> List:

        all_splits = []
        auto_metadata = {}
        for file_details in files_list:

            file_id = file_details["id"]
            file_name = file_details["name"]

            loaded_text = self.summarizer.load_text_from_document(file_id)
            split_text = self.summarizer.split_text(loaded_text)
            # add autometadata for all files
            auto_metadata[file_id] = {
                "name": file_name,
                "number_of_pages": get_number_pages(split_text),
            }
            # add file_id to all nodes
            for split in split_text:
                split.metadata["file_id"] = file_id
                split.metadata["file_name"] = file_name
            all_splits += split_text

        return all_splits, auto_metadata

    def generate_topics(
        self, splits: List, metadata: dict, project_name=None, save_project=True
    ) -> dict:

        if save_project:
            if not project_name:
                raise ValueError(
                    "Please enter a project name to save the BertTopic model"
                )

        embeddings, reduced_embeddings = self.summarizer.embed_text(splits)
        topics, probs, labels = self.summarizer.generate_topics_probs(
            splits, embeddings
        )
        self.summarizer.save_topic_model(project_name)

        labelled_splits = [s for s in splits]
        labelled_splits = add_topics_to_nodes(labelled_splits, labels, probs)

        return {
            "reduced_embeddings": reduced_embeddings,
            "topic_probs": probs,
            "topic_names": topics,
            "split_labels": labels,
            "labelled_splits": labelled_splits,
            "metadata": metadata,
        }

    def vizualize_topics(self, docs: dict, add_document_names=True) -> None:

        self.summarizer.vizualize(
            docs["reduced_embeddings"],
            docs["split_labels"],
            docs["labelled_splits"],
            add_document_names=add_document_names,
        )
        return None

    def add_metadata_to_all_nodes(self, nodes: List, metadata: dict) -> List:

        nodes = add_metadata_to_nodes(nodes, metadata)
        return nodes

    def get_metadata_nodes(self, metadata: dict, topics: dict = {}) -> List:

        metadata_nodes = []
        for file_id, details in metadata.items():
            details.update({"topic_labels": str(topics.get(file_id, "None"))})
            metadata_nodes.append(get_metadata_node(details))
        return metadata_nodes

    def get_topics_set(self, nodes: List[TextNode]) -> dict:

        file_ids = defaultdict(set)
        for node in nodes:
            file_ids[node.metadata["file_id"]].add(node.metadata["topic_label"])
        return file_ids

    def set_up_chat_agent(self, topic_dict: dict) -> None:

        # add metadata to all the nodes
        nodes = self.add_metadata_to_all_nodes(
            topic_dict["labelled_splits"], topic_dict["metadata"]
        )
        # create additional nodes containing just metadaat
        metadata_nodes = self.get_metadata_nodes(topic_dict["metadata"])
        # make a set of the represented topics
        topics_set = self.get_topics_set(topic_dict["labelled_splits"])

        doc_splits = nodes
        reduced_embeddings = topic_dict["reduced_embeddings"]

        # add the metadata nodes if we have any
        doc_dict = {
            "document_chunks": doc_splits,
            "reduced_embeddings": reduced_embeddings,
            "metadata_node": metadata_nodes,
        }
        self.chat_model = DocumentAgent(
            doc_dict, api_key=self.api_key, persist_indices=self.persist
        )
        return None

    def query(self, query: str, plot: bool = True) -> str:

        if isinstance(self.chat_model, type(None)):
            raise ValueError("Must run set_up_chat_agent to create chatbot")

        response = self.chat_model.query(query, plot)

        # response starts with "assistant: ", so we remove that
        return response[10:].strip()

    def chat(self, query: str, plot: bool = True) -> str:

        if isinstance(self.chat_model, type(None)):
            raise ValueError("Must run set_up_chat_agent to create chatbot")

        response = self.chat_model.chat(query, plot)

        # response starts with "assistant: ", so we remove that
        return response[10:].strip()
