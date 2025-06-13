import requests
import backoff
import json
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class SansadAPIClient:
    def __init__(self):
        self.base_url = "https://sansad.in/api_ls/question/qetFilteredQuestionsAns"
        self.session = requests.Session()

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException),
        max_tries=3
    )
    def fetch_questions(self, ministry: str, date_range: Dict) -> List[Dict]:
        """Fetch questions from Sansad API with retry logic"""
        try:
            params = {
                "ministry": ministry,
                "fromDate": date_range["start"],
                "toDate": date_range["end"]
            }
            
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise

    def process_questions(self, raw_data: List[Dict]) -> List[Dict]:
        """Process and format raw API data for vector storage"""
        processed_data = []
        
        for item in raw_data:
            processed_item = {
                "id": item["questionID"],
                "text": f"""
                Question: {item["question"]}
                Answer: {item["answer"]}
                """,
                "metadata": {
                    "ministry": item["ministry"],
                    "member": item["memberName"],
                    "date": item["date"],
                    "question_type": item["questionType"]
                }
            }
            processed_data.append(processed_item)
            
        return processed_data