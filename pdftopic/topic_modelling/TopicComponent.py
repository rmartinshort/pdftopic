from abc import ABC, abstractmethod
import openai
from bertopic.representation import OpenAI as BertopicOpenAI
from bertopic.backend import OpenAIBackend
from llama_index.core.node_parser import SentenceSplitter
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer


class TopicComponent(ABC):

    def set_model(self, model=None):

        if isinstance(model, type(None)):
            self.model = self._default_model()
        else:
            self.model = model

    @abstractmethod
    def _default_model(self):
        raise NotImplementedError


class TextSplitterComponent(TopicComponent):

    def __init__(self, model=None):
        self.component_name = "text_splitter_model"
        self.set_model(model)

    def _default_model(self):
        return SentenceSplitter(chunk_size=512, chunk_overlap=25, include_metadata=True)


class EmbeddingComponent(TopicComponent):

    def __init__(self, model=None):
        self.component_name = "embedding_model"
        self.set_model(model)

    def _default_model(self):
        return SentenceTransformer("all-MiniLM-L6-v2")

    def open_ai_model(self, api_key):
        client = openai.OpenAI(api_key=api_key)
        embedding_model = OpenAIBackend(client, "text-embedding-ada-002")
        return embedding_model


class DimensionReductionComponent(TopicComponent):

    def __init__(self, model=None):
        self.component_name = "dimensionality_reduction_model"
        self.set_model(model)

    def _default_model(self):
        return {
            "pipeline": UMAP(
                n_neighbors=5, n_components=5, min_dist=0.0, metric="cosine"
            ),
            "viz": UMAP(n_neighbors=5, n_components=2, min_dist=0.0, metric="cosine"),
        }


class ClusterComponent(TopicComponent):

    def __init__(self, model=None):
        self.component_name = "clustering_model"
        self.set_model(model)

    def _default_model(self):
        return HDBSCAN(
            min_cluster_size=10,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )


class TopicVectorizerComponent(TopicComponent):

    def __init__(self, model=None):
        self.component_name = "topic_vectorizer"
        self.set_model(model)

    def _default_model(self):
        return CountVectorizer(
            stop_words="english", ngram_range=(1, 3), min_df=1, max_df=1.0
        )


class TopicModelComponent(TopicComponent):

    def __init__(self, model=None):
        self.component_name = "topic_model"
        self.set_model(model)

    def _default_model(self):
        return ClassTfidfTransformer(bm25_weighting=True)


class TopicRepresentationComponent(TopicComponent):

    def __init__(self, api_key, model=None):
        self.component_name = "topic_representation"
        self.api_key = api_key
        self.set_model(model)

    def _default_model(self):
        summarization_prompt = """
        I have a topic that is described by the following keywords: [KEYWORDS]
        In this topic, the following documents are a small but representative subset of all documents in the topic:
        [DOCUMENTS]

        Based on the information above, please provide a single descriptive label for the topic that is less than 5 words long.
        If you're not sure, make a best guess. Your output topic label MUST be less than 5 words in length.
        topic: <descriptive label>
        """

        client = openai.OpenAI(api_key=self.api_key)
        llm_caller = BertopicOpenAI(
            client,
            model="gpt-3.5-turbo",
            chat=True,
            prompt=summarization_prompt,
            nr_docs=20,
            delay_in_seconds=3,
        )
        return llm_caller
