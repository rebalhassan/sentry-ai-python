# Sentry AI - a local tool to let you analyze log files and find root cause of an error much fast and securely(since everything is on your machine)

We are building this in pure python. 
We are using pydantic to create data models that allow us to seamlessly input data to LLMs and other embedding models without worrying about how the data will be structred.
We are then using Sql-lite for database and sentence transformer to create an embedding service to be used in RAG.