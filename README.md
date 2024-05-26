# pdftopic
Use BertTopic and Agentic RAG to build QA on pdfs 

## Quickstart and test 

- Make a new virtual environment with Python 3.9+ and install the requirements
- Try the following code 

```
from pdftopic.utils import load_secrets
from pdftopic.DocumentChatDriver import DocumentChatDriver

secrets = load_secrets()
chatter = DocumentChatDriver(api_key=secrets["OPENAI_API_KEY"])

# look for files
files_list = chatter.search_for_files()

# split the data
splits, metadata = chatter.generate_splits([files_list[-1]])

# generate topics
topic_splits = chatter.generate_topics(
    splits, metadata, project_name="TEST", save_project=True
)

# look at the topics that have been generated
chatter.vizualize_topics(topic_splits)

# set up agent
chatter.set_up_chat_agent(topic_splits)

# ask a question, with memory
res = chatter.chat("summarize whats being discussed on page 10?")

print(res)
```

Note that you first need to have an OpenAI API key stored in .env at the top level of
of this package. You also need to set up a service account connection to a Google Drive folder
where you can store your documents. The code looks for a file called `topicspdf_gdrive_key.json` 
that contains the details needed to connect to google drive. 