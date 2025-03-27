# Mirial - AI-Powered Document Scraping and RAG System

Mirial is a modular and scalable system for web scraping, document processing, and intelligent querying using Retrieval-Augmented Generation (RAG). The system consists of several microservices that work together to scrape web content, process and vectorize it, and provide a powerful RAG interface.

## System Architecture

The system consists of the following microservices:

1. **API Service**: The main interface for clients, handling query requests and orchestrating interactions between services.
2. **Scraper Service**: Responsible for crawling websites and extracting content from web pages.
3. **Vectorizer Service**: Processes raw text data, creates embeddings, and manages the vector store.
4. **Ollama-RAG Service**: Runs Mistral, a language model, for generating responses based on retrieved content.

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for Mistral service)
- PostgreSQL database

## PostgreSQL Setup

1. Install PostgreSQL on your system or use a hosted version.
2. Create a new database for the application:

```sql
CREATE DATABASE mirialdbdev;
```

3. Ensure your PostgreSQL server is accessible from Docker containers (adjust pg_hba.conf and postgresql.conf as needed).

## Configuration

### Database Configuration

Update the PostgreSQL connection details in the `docker-compose.yml` file. Find the environment variables for each service and modify them according to your PostgreSQL setup:

```yaml
environment:
  - DB_HOST=host.docker.internal # Change to your PostgreSQL server address
  - DB_PORT=5432 # Change if your PostgreSQL uses a different port
  - DB_USER=postgres # Change to your PostgreSQL username
  - DB_PASSWORD=yourpassword # Change to your PostgreSQL password
  - DB_NAME=mirialdbdev # Change to your database name
```

### Running on CPU Instead of CUDA

By default, the Mistral service is configured to run on NVIDIA CUDA GPUs. If you want to run it on CPU instead:

1. Modify the `mistral/Dockerfile` to use CPU-based images and dependencies:

   - Change the base image to a regular Python image instead of NVIDIA CUDA
   - Update the torch installation to use CPU version
   - Remove CUDA-specific environment variables

2. Remove the NVIDIA-specific configurations from the `docker-compose.yml` file for the `ollama-rag` service:
   ```yaml
   # Remove or comment out these sections
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   environment:
     - NVIDIA_VISIBLE_DEVICES=all
     - NVIDIA_DRIVER_CAPABILITIES=all
   ```

## Getting Started

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd mirial
   ```

2. Build and start the services:

   ```bash
   docker-compose up -d
   ```

3. The services will be available at the following endpoints:
   - API Service: http://localhost:8005
   - Scraper Service: http://localhost:8001
   - Vectorizer Service: http://localhost:8002
   - Ollama-RAG Service: http://localhost:8006

## Data Flow

1. **Domain Registration**: Register websites to be scraped through the API service
2. **Web Scraping**: The scraper service crawls the registered domains and extracts content
3. **Text Processing**: Raw text is processed and stored in the database
4. **Vectorization**: Text is converted into embeddings and stored in the vector database
5. **Querying**: Users can ask questions through the API, which uses RAG to generate contextually relevant answers

## Database Schema

The main data models include:

- **Domain**: Represents a website to be scraped
- **ScrapedDocument**: Contains the raw content extracted from web pages

## Health Checks

All services include health checks to ensure they're running properly. You can verify the status by accessing:

- API: http://localhost:8005/health
- Scraper: http://localhost:8001/health
- Vectorizer: http://localhost:8002/health
- Ollama-RAG: http://localhost:8006/

## Development

For development purposes, the volumes are mounted to allow real-time code changes without rebuilding the containers:

- Vectorizer service uses local code directories
- Mistral service uses local code directories

## Troubleshooting

- **Database Connection Issues**: Ensure your PostgreSQL server is running and accessible from Docker containers.
- **GPU Not Detected**: Verify your NVIDIA drivers and CUDA installation are working correctly.
- **Service Crashes**: Check the logs with `docker-compose logs <service_name>` for detailed error messages.

This work belongs to Sridhar Narayan Kashyap. Please reach out to me at sridharkashyap04@gmail.com for any questions or feedback.
