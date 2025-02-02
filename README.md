# Neural Search Service:

1) Clone the Neural Search Service Project:
* Clone the Neural Search Service project from its GitHub repository.

2) Install Dependencies:
* Execute pip install -r requirements.txt to install the necessary dependencies.
* Execute python -m spacy download en_core_web_sm


3) Pull and Run Qdrant Docker Container:
* Pull the Qdrant Docker image by running docker pull qdrant/qdrant. Then, initiate the container with docker run -p 6333:6333 qdrant/qdrant.

4) Load Data into Qdrant:
* Once the Qdrant container is running, execute load_data.py to load the data into the container.

5) Run the Neural Search Service:
* Launch service.py to start the Neural Search Service, which will be accessible through port 8000.

# Project Details:
Created a Simple Neural Search Service using the Qdrant open-source library and extended its functionality to include a feature that filters search results based on the city mentioned in the user's input text.

Here's a summary of the setup:

Neural Search Microservice (Task 1):

I deployed the Qdrant library as a neural search service.
I added a new feature to the service that allows it to extract the city mentioned in the user's input text to filter search results accordingly.
Calling the Neural Search Service from Another Microservice (Task 2):

In my second microservice (Task 2), I make HTTP requests to the neural search service created in Task 1.
I provide a CURL command as an example of how to call the neural search service, specifying the query text and parameters in the URL.

curl --location 'http://localhost:8000/api/search?q=Are%20there%20startups%20about%20wine%3F'


![image](https://github.com/ShakuriAvi/Ai-Chatbot/assets/65177459/d52abea1-024d-410d-bc9b-5d5ebbf0ebed)

  # Ai-Chatbot
  https://github.com/ShakuriAvi/Ai-Chatbot
