"""
ViSense API client for PagaLava laundromat
"""
import os
import requests
from dotenv import load_dotenv

class ViSenseAPI:
    def __init__(self):
        """Initialize the ViSense API client."""
        load_dotenv()
        self.api_url = os.getenv("VISENSE_API_URL", "https://visense2-func-py.azurewebsites.net/api")
        self.api_code = os.getenv("VISENSE_API_CODE")
        if not self.api_code:
            raise ValueError("VISENSE_API_CODE environment variable is not set")

    def reboot_machine(self, laundry_id, machine_number):
        """
        Reboot a specific machine in a laundry.
        
        Args:
            laundry_id (str): Laundry identifier (L1 or L2)
            machine_number (int): Machine number to reboot
            
        Returns:
            dict: Response from the API
        """
        endpoint = f"{self.api_url}/reboot_machine"
        params = {
            "code": self.api_code,
            "laundry_id": laundry_id,
            "machine_number": int(machine_number)
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"status": "error", "message": f"API request failed: {str(e)}"}

    def check_presence(self, laundry_id, check_streams=1, check_sensors=1, check_logs=1):
        """
        Check presence in a specific laundry.
        
        Args:
            laundry_id (str): Laundry identifier (L1 or L2)
            check_streams (int): Whether to check camera streams
            check_sensors (int): Whether to check sensors
            check_logs (int): Whether to check purchase logs
            
        Returns:
            dict: Response from the API with presence information
        """
        endpoint = f"{self.api_url}/check_presence"
        params = {
            "code": self.api_code,
            "laundry_id": laundry_id,
            "streams": check_streams,
            "sensors": check_sensors,
            "logs": check_logs
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"status": "error", "message": f"API request failed: {str(e)}"}

# Example usage
if __name__ == "__main__":
    api = ViSenseAPI()
    # Test the API with a ping (not actually rebooting)
    print("API initialized")