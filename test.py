import os
from functions import get_sentiment, obtain_proportion

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
sentiments = get_sentiment('Uk28ec4W4sA')
proportion = obtain_proportion(sentiments)
print(proportion)
