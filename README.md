# IndeedRecommender

IndeedRecommender is an intelligent job recommendation system designed to provide detailed job requirements for specific positions. By leveraging data from various educational and career resources, IndeedRecommender helps job seekers understand the requirements for their desired jobs and assists employers in defining job requirements for open positions.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Workflow](#project-workflow)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Job Requirement Recommendation**: Provides specific job requirements for various positions based on collected data.
- **Data Collection**: Retrieves study fields from the Université Laval website and explores potential career paths.
- **Job Data Retrieval**: Scrapes job data from Indeed for use in training machine learning models.
- **Metrics-Based Responses**: Delivers accurate responses to user queries based on defined metrics.
- **Employer Assistance**: Helps employers fill job requirements for specific positions.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/IndeedRecommender.git
    cd IndeedRecommender
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install ChromeDriver for Selenium:
    - Download ChromeDriver from [here](https://sites.google.com/a/chromium.org/chromedriver/downloads).
    - Ensure the `chromedriver` executable is in your system's PATH or specify its location in the Selenium configuration.

## Usage

### Data Collection

1. Retrieve study fields from the Université Laval website:
    ```python
    python scripts/retrieve_study_fields.py
    ```

2. Explore career paths for the retrieved study fields:
    ```python
    python scripts/explore_career_paths.py
    ```

3. Scrape job data from Indeed:
    ```python
    python scripts/scrape_indeed_jobs.py
    ```

### Job Requirement Recommendation

To get job requirements for a specific position:
```python
python scripts/recommend_job_requirements.py --job_title "Data Scientist"
