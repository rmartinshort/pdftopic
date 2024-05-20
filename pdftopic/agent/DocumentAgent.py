from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
import datamapplot
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from llama_index.core import StorageContext, load_index_from_storage
from pdftopic.agent.prompts import AgentPrompt
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class DocumentAgent:

    INDEX_PERSIST_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "datasets", "persisted_indices"
    )

    def __init__(
        self,
        docs,
        api_key,
        embedding_model="text-embedding-ada-002",
        llm_model="gpt-3.5-turbo",
        temperature=0,
        persist_indices=True,
    ):

        if not os.path.exists(self.INDEX_PERSIST_DIR):
            os.mkdir(self.INDEX_PERSIST_DIR)

        self.persist_index = persist_indices
        self.embedding_model = OpenAIEmbedding(model=embedding_model, api_key=api_key)
        self.llm = OpenAI(model=llm_model, temperature=temperature, api_key=api_key)
        self.agent_prompt = AgentPrompt()
        self.docs = docs["document_chunks"]
        self.reduced_embeddings = docs["reduced_embeddings"]
        self.metadata_nodes = docs["metadata_node"]

        # build the agent
        self.agent = self.set_up_agent()

    def set_up_agent(self):

        all_tools = self.generate_tools_from_split_documents()
        agent_worker = FunctionCallingAgentWorker.from_tools(
            all_tools,
            llm=self.llm,
            system_prompt=self.agent_prompt.system_prompt,
            verbose=True,
        )
        # the agent runner is going to decide what steps to use to answer the question
        # then the agent worker is actually going to do the task
        agent = AgentRunner(agent_worker)
        return agent

    def query(self, query, plot=True):

        response = self.agent.query(query)

        if plot:
            self.vizualize_chunks(response)

        return response.response

    def chat(self, query, plot=True):

        response = self.agent.chat(query)

        if plot:
            self.vizualize_chunks(response)

        return response.response

    def vizualize_chunks(self, response):

        node_ids = set([node.node.id_ for node in response.source_nodes])
        chosen_node_ids = np.array(
            [i for i, n in enumerate(self.docs) if n.id_ in node_ids]
        )

        logging.info("Generating plot showing topics and extracted nodes")

        # Run the visualization
        fig, ax = datamapplot.create_plot(
            self.reduced_embeddings,
            self.topic_labels_list,
            noise_label="No Topic",
            label_font_size=11,
            label_wrap_width=20,
            use_medoids=True,
            figsize=(6, 6),
            arrowprops={
                "arrowstyle": "wedge,tail_width=0.5",
                "connectionstyle": "arc3,rad=0.05",
                "linewidth": 0,
                "fc": "#33333377",
            },
        )
        # Add detail for chosen nodes
        if len(chosen_node_ids) > 0:
            ax.plot(
                self.reduced_embeddings[chosen_node_ids, 0],
                self.reduced_embeddings[chosen_node_ids, 1],
                "ro",
                label="chosen",
                alpha=0.5,
            )
            ax.legend()
        plt.show()

    @staticmethod
    def build_summary_index_tool_from_nodes(nodes, topic_name=None):

        if topic_name:

            summary_tool_description = """
                Use ONLY IF the user asks for a summary of parts of the corpus that are associated
                with the topic label {}

                Do NOT use for specific questions, even if they are related to the topic label
            """.format(
                topic_name
            )
            # tool name should be less than 64 chars
            tool_name = topic_name.replace(" ", "_")[:45]

        else:

            summary_tool_description = """
                Use ONLY IF you want to get a holistic summary of the entire document. 
                Do NOT use if you have specific questions over the document.
            """
            tool_name = "holistic"

        this_summary_index = SummaryIndex(nodes)

        summary_query_engine = this_summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )

        summary_query_tool = QueryEngineTool.from_defaults(
            name=f"summary_tool_{tool_name}",
            query_engine=summary_query_engine,
            description=(summary_tool_description),
        )

        return summary_query_tool

    def build_vector_index_tool_from_nodes(
        self, nodes, metadata, document_name, persist=True
    ):

        vector_tool_description = """
            Use to answer specific questions over the document called {}.
        """.format(
            document_name
        )

        document_tool_name = os.path.splitext(document_name)[0]
        logging.info("Generating a vector tool called {}".format(document_tool_name))

        # documents vector index
        # can be costly to generate since we vectorize all the chunks
        if persist:
            logging.info("Looking to load from saved index")
            if os.path.isdir(
                os.path.join(DocumentAgent.INDEX_PERSIST_DIR, document_name)
            ):
                logging.info(
                    "Found saved index at {}".format(DocumentAgent.INDEX_PERSIST_DIR)
                )
                # first, check if we can load a pre-existing vector store
                storage_context = StorageContext.from_defaults(
                    persist_dir=os.path.join(
                        DocumentAgent.INDEX_PERSIST_DIR, document_name
                    )
                )
                vector_index = load_index_from_storage(storage_context)
            else:
                # generate the index and store it, include the metadata nodes
                logging.info(
                    "No saved index found, will generate and save one at {}".format(
                        DocumentAgent.INDEX_PERSIST_DIR
                    )
                )
                vector_index = VectorStoreIndex(
                    nodes + metadata, embed_model=self.embedding_model
                )
                vector_index.storage_context.persist(
                    persist_dir=os.path.join(
                        DocumentAgent.INDEX_PERSIST_DIR, document_name
                    )
                )

        else:
            # just generate the index, include the metadata nodes
            vector_index = VectorStoreIndex(
                nodes + metadata, embed_model=self.embedding_model
            )

        def vector_query(
            query: str,
            page_numbers: Optional[List[str]] = None,
            topic_labels: Optional[List[str]] = None,
        ) -> str:
            """
            Useful if you have specific questions over a document
            Always leave page_numbers as None UNLESS there is a specific page you want to search for.
            Always leave topics as None UNLESS there are specific topics you want to search for.
            Args:
                query (str): the string query to be embedded.
                page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE
                    if we want to perform a vector search
                    over all pages. Otherwise, filter by the set of specified pages.
                topic_labels (Optional[List[str]]): Filter by set of topic names. Leave as NONE
                    if we want to perform a vector search
                    over all topics. Otherwise, filter by the set of specified topics.
            """

            # .format(str(self.topic_labels_set))

            # Note that the only available topic labels are {}

            page_numbers = page_numbers or []
            topic_labels = topic_labels or []
            pages = [{"key": "page_label", "value": p} for p in page_numbers]
            topics = [{"key": "topic_label", "value": l} for l in topic_labels]

            query_engine = vector_index.as_query_engine(
                similarity_top_k=5,
                filters=MetadataFilters.from_dicts(
                    pages + topics, condition=FilterCondition.OR
                ),
            )
            response = query_engine.query(query)
            return response

        vector_query_tool = FunctionTool.from_defaults(
            name=f"vector_tool_{document_tool_name}",
            fn=vector_query,
            description=(vector_tool_description),
        )
        return vector_query_tool

    def generate_tools_from_split_documents(self):

        # get set of all topic names and the titles of all the documents
        self.topic_labels_set = list(
            set([x.metadata.get("topic_label", "Unlabelled") for x in self.docs])
        )
        self.topic_labels_list = [
            x.metadata.get("topic_label", "Unlabelled") for x in self.docs
        ]
        self.titles_set = list(
            set([x.metadata.get("name", "Untitled") for x in self.docs])
        )
        doc_tools = {}
        for label in self.topic_labels_set:
            # get just the nodes with that label
            chosen_nodes = [
                node for node in self.docs if node.metadata["topic_label"] == label
            ]
            node_summary_tool = self.build_summary_index_tool_from_nodes(
                chosen_nodes, topic_name=label
            )
            doc_tools[label] = node_summary_tool

        for document_title in self.titles_set:
            chosen_nodes = [
                node for node in self.docs if node.metadata["name"] == document_title
            ]
            overall_summary_tool = self.build_summary_index_tool_from_nodes(
                chosen_nodes, topic_name=None
            )
            vector_search_tool = self.build_vector_index_tool_from_nodes(
                chosen_nodes, self.metadata_nodes, document_title, self.persist_index
            )
            doc_tools["{}_overall_summary".format(document_title)] = (
                overall_summary_tool
            )
            doc_tools["{}_vector_search".format(document_title)] = vector_search_tool

        self.doc_tools = doc_tools
        # get a list of all the tools
        return [v for k, v in doc_tools.items()]
