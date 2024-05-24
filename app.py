from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
import requests, os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from textwrap import dedent
from PyPDF2 import PdfReader
from typing import List, Dict, Any
from unstructured.partition.pdf import partition_pdf
import unstructured
import pdfplumber
from langchain_core.documents import Document
import base64
from PIL import Image
from openai import OpenAI


embedding_function = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Hardcoded paths
pdf_path = "data/sample_rule.pdf"
image_paths = ["data/car_sample1.jpeg", "data/car_sample3.jpeg"]
text_description = "The mileage of the vehicle is 30,071. The market price is around 29000."
vin = ""

# Tool 1 Store the PDF guide in Chroma vector database
class StoreRulesInChroma:
    @tool("Store Rules Tool")
    def store_rules(pdf_path: str):
        """Extracts text from a PDF and stores it in a Chroma vector database."""
        print(f"Received PDF path: {pdf_path}")
        documents = StoreRulesInChroma.read_and_partition_pdf(pdf_path)
        if not documents:
            return "Failed to read and process the PDF."

        StoreRulesInChroma.store_documents_in_chroma(documents)
        return "Rules stored successfully in Chroma vector database."

    @staticmethod
    def read_and_partition_pdf(file_path: str) -> List[Document]:
        """Reads and partitions a PDF file into document chunks."""
        documents = []
        with pdfplumber.open(file_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(full_text)

        documents = [Document(page_content=chunk, metadata={}) for chunk in chunks]
        return documents

    @staticmethod
    def store_documents_in_chroma(documents: List[Document]):
        """Stores documents in the Chroma vector database."""
        vectorstore = Chroma.from_documents(documents, embedding=embedding_function, persist_directory="./chroma_db")


# Tool 2: Access Vehicle Information API
class AccessVehicleAPI:
    @tool("Access Vehicle API Tool")
    def access_api(vin: str):
        """Fetches vehicle information using VIN and stores it in a Chroma vector database."""
        vehicle_info = AccessVehicleAPI.fetch_vehicle_info(vin)
        if not vehicle_info:
            return "Failed to fetch vehicle information."

        AccessVehicleAPI.store_vehicle_info_in_chroma(vehicle_info)
        return "Vehicle information stored successfully in Chroma vector database."

    @staticmethod
    def fetch_vehicle_info(vin: str) -> Dict[str, Any]:
        """Fetches vehicle information using the VIN."""
        url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/{vin}?format=json"
        response = requests.get(url)
        if response.status_code != 200:
            return None
        data = response.json()
        vehicle_info = {
            "VIN": vin,
            "Trim": next((item["Value"] for item in data["Results"] if item["Variable"] == "Trim"), "N/A"),
            "Make": next((item["Value"] for item in data["Results"] if item["Variable"] == "Make"), "N/A"),
            "ModelYear": next((item["Value"] for item in data["Results"] if item["Variable"] == "Model Year"), "N/A"),
            # Add more if needed
        }
        print(vehicle_info)
        return vehicle_info

    @staticmethod
    def store_vehicle_info_in_chroma(vehicle_info: Dict[str, Any]):
        """Stores vehicle information in the Chroma vector database."""
        document = Document(page_content=str(vehicle_info), metadata={"VIN": vehicle_info["VIN"]})
        vectorstore = Chroma.from_documents([document], embedding=embedding_function, persist_directory="./chroma_db")


# Tool 3: Process and Describe Images
class ProcessImages:
    @tool("Process Images Tool")
    def process_images(image_paths: List[str]):
        """Processes vehicle images, describes them, and stores the descriptions in a Chroma vector database."""
        descriptions = ProcessImages.process_and_describe_images(image_paths)
        if not descriptions:
            return "Failed to process and describe the images."

        ProcessImages.store_descriptions_in_chroma(descriptions)
        return "Image descriptions stored successfully in Chroma vector database."

    @staticmethod
    def process_and_describe_images(image_paths: List[str]) -> List[str]:
        """Processes and describes images using GPT-4o model."""
        descriptions = []
        for image_path in image_paths:
            encoded_image = ProcessImages.encode_image(image_path)
            description = ProcessImages.get_image_description_from_gpt4o(encoded_image)
            if description:
                descriptions.append(description)
        return descriptions

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encodes an image file as a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def get_image_description_from_gpt4o(encoded_image: str) -> str:
        """Uses GPT-4o API to get image description."""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that describes vehicle images."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe the vehicle in this image, make sure you mention any defects, any upgrades and important details that might affect the evaluation of the price of the vehicle:"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"}
                    }
                ]}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content

    @staticmethod
    def store_descriptions_in_chroma(descriptions: List[str]):
        """Stores image descriptions in the Chroma vector database."""
        documents = [Document(page_content=description, metadata={}) for description in descriptions]
        vectorstore = Chroma.from_documents(documents, embedding=embedding_function, persist_directory="./chroma_db")


# Tool 4: Get the vehicle information from database
class GenerateReport:
    @tool("Generate Report Tool")
    def generate_report(file_path: str = "vehicle_report.txt"):
        """Generates a report by accessing the vector database using similarity search, applying the rules, and following a specific format. Saves the report to a text file."""
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

        # Fetch relevant information from Chroma
        rules = vectorstore.similarity_search("rules") or []
        vehicle_info = vectorstore.similarity_search("vehicle information") or []
        image_descriptions = vectorstore.similarity_search("image descriptions") or []

        # Combine the fetched information into a report
        report = GenerateReport.format_report(rules, vehicle_info, image_descriptions)

        # Save the report to a text file
        with open(file_path, "w") as file:
            file.write(report)
        
        return f"Report generated and saved to {file_path}"

    @staticmethod
    def format_report(rules: List[Document], vehicle_info: List[Document], image_descriptions: List[Document]) -> str:
        """Formats the fetched information into a comprehensive report."""
        report = "Vehicle Evaluation Report\n\n"
        report += "Rules:\n"
        for rule in rules:
            report += f"- {rule.page_content}\n"
        
        report += "\nVehicle Information:\n"
        for info in vehicle_info:
            report += f"- {info.page_content}\n"
        
        report += "\nImage Descriptions:\n"
        for desc in image_descriptions:
            report += f"- {desc.page_content}\n"
        
        return report


# Agent1: PDF Rules Agent
pdf_rules_agent = Agent(
    role='PDF Rules Agent',
    goal='Extract text from a PDF of rules and store it in a Chroma vector database.',
    backstory='Responsible for processing rule PDFs.',
    tools=[StoreRulesInChroma().store_rules],
    allow_delegation=True,
    verbose=True,
    llm=llm
)

# Agent2: Vehicle API Agent
vehicle_info_agent = Agent(
    role='Vehicle Information Agent',
    goal='Fetch vehicle information using VIN and store it in a Chroma vector database.',
    backstory='Responsible for retrieving and storing vehicle information.',
    tools=[AccessVehicleAPI().access_api],
    allow_delegation=True,
    verbose=True,
    llm=llm
)

# Agent3: Image Processing Agent
image_processing_agent = Agent(
    role='Image Processing Agent',
    goal='Process vehicle images, describe them, and store the descriptions in a Chroma vector database.',
    backstory='Responsible for processing and describing vehicle images.',
    tools=[ProcessImages().process_images],
    allow_delegation=True,
    verbose=True,
    llm=llm
)

# Agent4: Report Generation Agent
report_generation_agent = Agent(
    role='Report Generation Agent',
    goal='Generate a comprehensive report by accessing the vector database and applying the rules. The rules, description of images of the vehicle, and vehicle information are all stored in the vector database.',
    backstory='Responsible for generating detailed reports based on stored information.',
    tools=[GenerateReport().generate_report],
    allow_delegation=False,
    verbose=True,
    llm=llm
)


# Task1: Store Rules Task
def store_pdf_rules_task(agent, pdf_path):
    return Task(
        description=dedent(f"""\
            Extract text from the PDF of rules located at: {pdf_path}.
            Store the extracted text in the Chroma vector database.
            """),
        agent=agent,
        expected_output='Rules stored successfully in Chroma vector database.',
    )

# Task2: Access Vehicle Information Task
def access_vehicle_info_task(agent, vin):
    return Task(
        description=dedent(f"""\
            Fetch vehicle information using VIN: {vin}.
            Store the fetched information in the Chroma vector database.
            """),
        agent=agent,
        expected_output='Vehicle information stored successfully in Chroma vector database.',
    )

# Task3: Process Images Task
def process_images_task(agent, image_paths):
    return Task(
        description=dedent(f"""\
            Process vehicle images located at: {', '.join(image_paths)}.
            Describe them and store the descriptions in the Chroma vector database.
            """),
        agent=agent,
        expected_output='Image descriptions stored successfully in Chroma vector database.',
    )


# Task4: Generate Report Task
def generate_report_task(agent, file_path="vehicle_report.txt", text_description=text_description):
    return Task(
        description=dedent(f"""\
            Generate a comprehensive report by accessing the vector database, additional text description, and applying the rules.
            Additional text description: {text_description}.
            Make sure that you referenced the trim, year, and make of the vehicle in the report correctly, which is stored in the vector database by Vehicle Information Agent.
            Save the report to the specified file path: {file_path}.
            """),
        agent=agent,
        expected_output=f'Report generated and saved to {file_path}.',
    )

# Define the crew
rag_crew = Crew(
    agents=[pdf_rules_agent, vehicle_info_agent, image_processing_agent, report_generation_agent],
    tasks=[
        store_pdf_rules_task(pdf_rules_agent, pdf_path),
        access_vehicle_info_task(vehicle_info_agent, vin),
        process_images_task(image_processing_agent, image_paths),
        generate_report_task(report_generation_agent, "vehicle_report.txt")
    ],
    process=Process.sequential,
    manager_llm=llm
)



# Execute the crew
if __name__ == "__main__":    
    # Print if the file exists
    pdf_path = "data/sample_rule.pdf"
    if os.path.exists(pdf_path):
        print(f"The file {pdf_path} exists.")
    else:
        print(f"The file {pdf_path} does not exist.")
        
    result = rag_crew.kickoff()
    print(result)