from pdftopic.topic_modelling.TopicComponent import (
    TextSplitterComponent,
    TopicRepresentationComponent,
    TopicModelComponent,
    TopicVectorizerComponent,
    ClusterComponent,
    DimensionReductionComponent,
    EmbeddingComponent
)
class TopicComponentFactory:
    components_map = {
        "text_splitter": TextSplitterComponent,
        "topic_representation": TopicRepresentationComponent,
        "topic_model": TopicModelComponent,
        "topic_vectorizer": TopicVectorizerComponent,
        "clustering_model": ClusterComponent,
        "dimensionality_reduction_model": DimensionReductionComponent,
        "embedding_model": EmbeddingComponent
    }

    @staticmethod
    def get_component(topic_component_name, topic_component_model=None, kwargs={}):
        if topic_component_name not in TopicComponentFactory.components_map:
            raise ValueError("{} not a recognized topic component".format(topic_component_name))

        return TopicComponentFactory.components_map[topic_component_name](model=topic_component_model, **kwargs)
