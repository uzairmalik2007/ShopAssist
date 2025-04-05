import requests
import json

BASE_URL = "http://127.0.0.1:9000"

def test_chat_endpoint():
    test_cases = [
        {
            "message": "Show me laptops under 50000",
            "session_id": "test123"
        },
        {
            "message": "I need a gaming laptop with good graphics",
            "session_id": "test124"
        },
        {
            "message": "Compare Dell XPS and MacBook Pro",
            "session_id": "test125"
        },
        {
            "message": "What are the specs of Legion 5?",
            "session_id": "test126"
        }

    ]

    for test_case in test_cases:
        print(f"\nTesting: {test_case['message']}")
        response = requests.post(
            f"{BASE_URL}/chat",
            json=test_case
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Intent: {result['intent']}")
            print(f"Response: {result['response'][:200]}...")
            if result.get('recommendations'):
                print(f"Recommendations: {len(result['recommendations'])} found")
                print(result.get('recommendations'))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    test_chat_endpoint()