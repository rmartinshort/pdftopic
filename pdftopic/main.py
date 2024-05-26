from pdftopic.DocumentChatDriver import DocumentChatDriver
from pdftopic.utils import load_secrets


def main():

    secrets = load_secrets()
    chatter = DocumentChatDriver(api_key=secrets["OPENAI_API_KEY"])

    # look for files
    files_list = chatter.search_for_files()

    # split the data
    splits, metadata = chatter.generate_splits([files_list[-1]])

    # generate topics
    topic_splits = chatter.generate_topics(
        splits, metadata, project_name="LLM2VEC", save_project=True
    )

    # look at the topics that have been generated
    chatter.vizualize_topics(topic_splits)

    # set up agent
    chatter.set_up_chat_agent(topic_splits)

    # ask a question, with memory
    res = chatter.chat("summarize whats being discussed on page 10?")

    print(res)


if __name__ == "__main__":
    main()
