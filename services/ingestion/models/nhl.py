import os
import logging

log = logging.getLogger(__name__)

WEB_URL = "https://api-web.nhle.com/v1/"
STATS_URL = "https://api.nhle.com/stats/rest/en"



# I have pydantic which defines how the data looks like, then I have httpx which is the one that fetches it
# calls the API and returns those models
