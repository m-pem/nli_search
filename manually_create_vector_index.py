# This script is used to create a vector index for the NLI search on creation of the docker container.

from functions.database_utils import DatabaseUtil

databaseUtil = DatabaseUtil()

databaseUtil.create_vector_index(
    name="nli-search",
    dimension=512
)