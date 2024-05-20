import openai
from bertopic.backend import OpenAIBackend
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from tempfile import NamedTemporaryFile
from llama_index.core import SimpleDirectoryReader
from pdftopic.google_drive.GoogleDriveService import GoogleDriveService
from pdftopic.google_drive.GoogleDriveLoader import GoogleDriveLoader
from pdftopic.topic_modelling.TopicComponentFactory import TopicComponentFactory
import re
import datamapplot
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class DocumentTopicSummarizer:

    def __init__(self, openai_api_key):

        self.gdrive_service = GoogleDriveService().build()
        self.gdrive_loader = GoogleDriveLoader(self.gdrive_service)
        self.topic_components = TopicComponentFactory()

        dimension_reduction_model = self.topic_components.get_component(
            "dimensionality_reduction_model"
        ).model
        self.viz_dimension_reduction_model = dimension_reduction_model["viz"]
        self.splitter = self.topic_components.get_component("text_splitter").model
        self.embed_model = self.topic_components.get_component(
            "embedding_model",
            topic_component_model=self._set_up_embedding_model(openai_api_key),
        ).model
        self.representation_model = {
            "KeyBERT": KeyBERTInspired(),
            "LLM": self.topic_components.get_component(
                "topic_representation", kwargs={"api_key": openai_api_key}
            ).model,
        }

        self.topic_extractor = BERTopic(
            language="english",
            embedding_model=self.embed_model,
            calculate_probabilities=True,
            umap_model=dimension_reduction_model["pipeline"],
            hdbscan_model=self.topic_components.get_component("clustering_model").model,
            vectorizer_model=self.topic_components.get_component(
                "topic_vectorizer"
            ).model,
            ctfidf_model=self.topic_components.get_component("topic_model").model,
            representation_model=self.representation_model,
            top_n_words=20,
            verbose=True,
        )

    def _set_up_embedding_model(self, api_key):
        client = openai.OpenAI(api_key=api_key)
        embedding_model = OpenAIBackend(client, "text-embedding-ada-002")
        return embedding_model

    def search_for_files(self):

        return self.gdrive_loader.search_for_files()

    def load_text_from_document(self, document_id):

        pdf_bytes = self.gdrive_loader.download_file(document_id)
        with NamedTemporaryFile(suffix=".pdf") as temp_file:
            with open(temp_file.name, "wb") as f:
                f.write(pdf_bytes)
            # load the pdf data
            parsed_pdf = SimpleDirectoryReader(input_files=[temp_file.name]).load_data()

        return parsed_pdf

    def split_text(self, loaded_data):
        splits = self.splitter.get_nodes_from_documents(loaded_data)
        return splits

    def embed_text(self, splits):
        split_texts = [x.text for x in splits]
        all_embeddings = self.embed_model.embed(split_texts)
        reduced_embeddings = self.viz_dimension_reduction_model.fit_transform(
            all_embeddings
        )
        return all_embeddings, reduced_embeddings

    def generate_topics_probs(self, splits, embeddings):

        split_texts = [x.text for x in splits]
        topics, probs = self.topic_extractor.fit_transform(split_texts, embeddings)

        llm_labels = [
            re.sub(r"\W+", " ", label[0][0].split("\n")[0].replace('"', ""))
            for label in self.topic_extractor.get_topics(full=True)["LLM"].values()
        ]
        llm_labels = [label if label else "Unlabelled" for label in llm_labels]
        all_labels = [
            (
                llm_labels[topic + self.topic_extractor._outliers]
                if topic != -1
                else "Unlabelled"
            )
            for topic in topics
        ]
        self.topic_extractor.set_topic_labels(llm_labels)
        return topics, probs, all_labels

    def vizualize(self, reduced_embeddings, labels, splits, add_document_names=False):

        # Run the visualization
        fig, ax = datamapplot.create_plot(
            reduced_embeddings,
            labels,
            noise_label="No Topic",
            force_matplotlib=True,
            label_font_size=11,
            label_wrap_width=20,
            font_family="Urbanist",
            figsize=(8, 8),
        )
        ax.set_title("Topic distribution from BERTopic")

        if add_document_names:
            # add on the names of the documents, using markers
            marker_options = Line2D.filled_markers
            doc_names = [n.metadata.get("name", "unnamed document") for n in splits]
            doc_labels = defaultdict(list)
            for i, label in enumerate(doc_names):
                doc_labels[label].append(i)

            marker_id = 0
            for doc, indices in doc_labels.items():

                if marker_id >= len(marker_options):
                    marker_id = 0

                marker = marker_options[marker_id]
                ax.scatter(
                    reduced_embeddings[indices, 0],
                    reduced_embeddings[indices, 1],
                    label=doc,
                    alpha=0.25,
                    marker=marker,
                    c="black",
                )
                marker_id += 1
            ax.legend()
        plt.show()
