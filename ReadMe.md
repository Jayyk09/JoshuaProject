# Joshua Project RAG Based Bot

## Overview

The Joshua Project RAG Based Bot is designed to interact with a comprehensive database of people groups worldwide, providing insights and data-driven answers to queries about these groups. The bot leverages OpenAI's capabilities to convert natural language questions into SQL queries, facilitating easy access to the database.

## Key Features

- **Database Interaction**: Utilizes SQLAlchemy to connect and interact with a MySQL database containing detailed information about people groups.
- **Entity Caching**: Caches common entities such as people groups, countries, languages, and religions for efficient fuzzy matching.
- **Fuzzy Matching**: Implements simple fuzzy matching to handle variations in entity names, improving query accuracy.
- **Schema Description**: Automatically generates a description of the database schema to assist in query formulation.
- **Preprocessing**: Preprocesses user questions to identify potential entities and improve query generation.
- **OpenAI Integration**: Uses OpenAI to generate SQL queries from user questions, leveraging conversation history for context.

## How It Works

1. **Initialization**: The bot initializes by connecting to the database and caching common entities for fuzzy matching.
2. **Schema Description**: Retrieves and formats the database schema to provide context for query generation.
3. **Question Processing**: Preprocesses user questions to identify potential entities and uses OpenAI to generate SQL queries.
4. **Query Execution**: Executes the generated SQL queries against the database to retrieve and return relevant data.

## Usage

To use the Joshua Project RAG Based Bot, ensure you have the necessary environment variables set up, including the OpenAI API key. The bot can be run from the command line, where it will prompt for questions and return data-driven answers based on the database content.

## Requirements

- Python 3.x
- Pandas
- SQLAlchemy
- OpenAI Python Client
- dotenv

## Setup

1. Clone the repository.
2. Install the required Python packages.
3. Set up the `.env` file with your OpenAI API key and database connection details.
4. Run the `main.py` script to start interacting with the bot.

## Datasets

The bot interacts with several datasets, each containing specific information about people groups, languages, countries, and more. These datasets are stored in CSV files and are loaded into the database for querying.

- **AllLanguageListing.csv**: Contains language-related data.
- **PeopleCtryLangListing.csv**: Details people groups by country and language.
- **jp-cppi-cross-reference.csv**: Cross-references various identifiers and attributes.
- **AllCountriesListing.csv**: Provides country-specific data.
- **AllPeoplesAcrossCountries.csv**: Lists people groups across different countries.
- **FieldDefinitions.csv**: Describes the fields in the database.
- **UnreachedPeoplesByCountry.csv**: Focuses on unreached people groups by country.
- **AllPeoplesInCountry.csv**: Details all people groups within a country.

For more detailed information about each dataset, refer to the database schema and the CSV files provided in the repository.