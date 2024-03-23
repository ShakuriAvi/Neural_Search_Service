from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import spacy
from qdrant_client.models import Filter


class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # initialize Qdrant client
        self.qdrant_client = QdrantClient("http://localhost:6333")
        self.context = {"city": None}
        self.nlp = spacy.load("en_core_web_sm")
    def search(self, text: str):
        # Convert text query into vector
        vector = self.model.encode(text).tolist()
        city_of_interest = self.extract_city(text)
        if city_of_interest:
            self.set_context(city_of_interest )
        city_filter = None
        if self.context["city"]:
            city_filter = Filter(must=[{
                "key": "city",
                "match": {
                    "value": city_of_interest
                }
            }])
            # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=city_filter,  # If you don't want any filters for now
            limit=5,  # 5 the most closest results is enough
        )
        # `search_result` contains found vector ids with similarity scores along with the stored payload
        # In this function you are interested in payload only
        payloads = [hit.payload for hit in search_result]
        return payloads

    def set_context(self, city: str):
        self.context["city"] = city

    def extract_city(self, text: str):
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG" or ent.label_ == "GPE": # GPE label indicates geographic/political entities (e.g., cities)
                # Check if the entity is an organization
                # Perform further checks if needed to determine if the organization represents a city
                # For example, you could check if the organization name contains city names
                # or if it's a known organization related to cities
                # If it matches a city, return it
                return ent.text
        return None
