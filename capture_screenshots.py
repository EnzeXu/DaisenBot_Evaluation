#!/usr/bin/env python3
"""
Daisen Screenshot Capture Tool

This script captures screenshots of Daisen visualizations and converts them to base64 format.

Requirements:
    - Python packages: selenium, requests (installed in venv)
    - Firefox and geckodriver installed on the system

Setup:
    This script should be run within the venv virtual environment.
    
    To activate the virtual environment:
        source venv/bin/activate
    
Usage:
    python capture_screenshots.py <json_filename>
    
Example:
    source venv/bin/activate
    python capture_screenshots.py spmv.json
    
    This will:
    1. Read view_record/spmv.json
    2. Launch Daisen with the corresponding data file
    3. Capture screenshots of each URL
    4. Save results to spmv_screenshots.json in the DaisenBot_Evaluation directory
"""

import json
import os
import subprocess
import time
import base64
import tempfile
import requests
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


class DaisenScreenshotCapture:
    """
    Captures screenshots of Daisen visualizations and converts them to base64 format.
    """
    
    def __init__(self, view_record_dir="/home/Velocity/Desktop/sarch_lab/DaisenBot_Dataset/view_record",
                 data_dir="/home/Velocity/Desktop/sarch_lab/DaisenBot_Dataset/data",
                 daisen_dir="/home/Velocity/Desktop/sarch_lab/akita/daisen",
                 output_dir=None):
        self.view_record_dir = Path(view_record_dir)
        self.data_dir = Path(data_dir)
        self.daisen_dir = Path(daisen_dir)
        # If no output_dir specified, use the directory where this script is located
        if output_dir is None:
            self.output_dir = Path(__file__).parent
        else:
            self.output_dir = Path(output_dir)
        self.daisen_process = None
        self.driver = None
        
    def launch_daisen(self, data_id):
        """
        Launch Daisen server with the specified data file.
        
        Args:
            data_id: The data ID (e.g., "D0600000")
            
        Returns:
            subprocess.Popen: The Daisen process handle
        """
        sqlite_path = self.data_dir / f"{data_id}.sqlite3"
        
        if not sqlite_path.exists():
            raise FileNotFoundError(f"Data file not found: {sqlite_path}")
        
        daisen_executable = self.daisen_dir / "daisen"
        if not daisen_executable.exists():
            raise FileNotFoundError(f"Daisen executable not found: {daisen_executable}")
        
        print(f"Launching Daisen with data file: {sqlite_path}")
        
        # Launch Daisen in the background
        process = subprocess.Popen(
            [str(daisen_executable), "-sqlite", str(sqlite_path)],
            cwd=str(self.daisen_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        self.daisen_process = process
        return process
    
    def check_daisen_ready(self, url="http://localhost:3001", timeout=10, interval=0.5):
        """
        Check if Daisen server is ready by pinging it.
        
        Args:
            url: The base URL to ping
            timeout: Maximum time to wait in seconds
            interval: Time between ping attempts in seconds
            
        Returns:
            bool: True if server is ready, False otherwise
        """
        print(f"Waiting for Daisen to be ready at {url}...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    print("Daisen is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(interval)
        
        print(f"Daisen did not respond within {timeout} seconds")
        return False
    
    def setup_driver(self):
        """
        Set up Firefox Selenium WebDriver in headless mode.
        
        Returns:
            webdriver.Firefox: Configured Firefox driver
        """
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--width=1920")
        options.add_argument("--height=1080")
        
        print("Setting up Firefox WebDriver in headless mode...")
        driver = webdriver.Firefox(options=options)
        driver.set_window_size(1920, 1080)
        
        self.driver = driver
        return driver
    
    def wait_for_page_load(self, driver, timeout=30):
        """
        Wait for the page to fully load.
        
        Args:
            driver: Selenium WebDriver instance
            timeout: Maximum time to wait in seconds
        """
        # Wait for document ready state
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        
        # Additional buffer for JavaScript rendering
        time.sleep(2)
    
    def capture_screenshot(self, url, driver):
        """
        Capture a screenshot of the given URL and convert to base64.
        
        Args:
            url: The URL to capture
            driver: Selenium WebDriver instance
            
        Returns:
            str: Base64-encoded JPEG image with data URI prefix
        """
        print(f"Capturing screenshot of: {url}")
        
        try:
            # Navigate to URL
            driver.get(url)
            
            # Wait for page to load
            self.wait_for_page_load(driver)
            
            # Create temporary file for screenshot
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Take screenshot
            driver.save_screenshot(tmp_path)
            
            # Read and convert to base64
            with open(tmp_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_string = base64.b64encode(image_data).decode('utf-8')
            
            # Delete temporary file
            os.remove(tmp_path)
            
            # Return with data URI prefix
            return f"data:image/jpeg;base64,{base64_string}"
            
        except TimeoutException:
            raise Exception(f"Timeout while loading page: {url}")
        except Exception as e:
            raise Exception(f"Error capturing screenshot for {url}: {str(e)}")
    
    def process_json(self, json_filename):
        """
        Process a single JSON file: launch Daisen, capture screenshots, save results.
        
        Args:
            json_filename: Name of the JSON file in view_record directory (e.g., "spmv.json")
            
        Returns:
            str: Path to the output JSON file
        """
        # Load input JSON
        input_path = self.view_record_dir / json_filename
        if not input_path.exists():
            raise FileNotFoundError(f"Input JSON not found: {input_path}")
        
        print(f"\nProcessing: {json_filename}")
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        if not data:
            raise ValueError(f"Empty JSON file: {json_filename}")
        
        # Extract data_id (should be same for all entries)
        data_id = data[0]['data_id']
        print(f"Data ID: {data_id}")
        
        # Launch Daisen
        self.launch_daisen(data_id)
        
        # Wait for Daisen to be ready
        if not self.check_daisen_ready():
            self.cleanup()
            raise Exception("Daisen failed to start")
        
        # Setup Selenium driver
        driver = self.setup_driver()
        
        # Process each URL
        results = []
        try:
            for i, entry in enumerate(data, 1):
                print(f"\nProcessing entry {i}/{len(data)}")
                view_id = entry['id']
                view_data_id = entry['data_id']
                view_url = entry['view_url']
                
                # Capture screenshot and convert to base64
                base64_image = self.capture_screenshot(view_url, driver)
                
                # Create result entry
                result = {
                    "view_id": view_id,
                    "data_id": view_data_id,
                    "url": base64_image
                }
                results.append(result)
                
                print(f"✓ Successfully captured screenshot for {view_id}")
        
        except Exception as e:
            print(f"\n✗ Error occurred: {str(e)}")
            self.cleanup()
            raise
        
        # Save results to output JSON
        output_filename = json_filename.replace('.json', '_screenshots.json')
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✓ Results saved to: {output_path}")
        
        # Cleanup
        self.cleanup()
        
        return str(output_path)
    
    def cleanup(self):
        """
        Clean up resources: close driver and terminate Daisen process.
        """
        print("\nCleaning up...")
        
        if self.driver:
            try:
                self.driver.quit()
                print("✓ Selenium driver closed")
            except Exception as e:
                print(f"Warning: Error closing driver: {e}")
            self.driver = None
        
        if self.daisen_process:
            try:
                self.daisen_process.terminate()
                self.daisen_process.wait(timeout=5)
                print("✓ Daisen process terminated")
            except Exception as e:
                print(f"Warning: Error terminating Daisen: {e}")
                try:
                    self.daisen_process.kill()
                except:
                    pass
            self.daisen_process = None


def main():
    """
    Main function to run the screenshot capture process.
    
    Usage:
        python capture_screenshots.py <json_filename>
        
    Example:
        python capture_screenshots.py spmv.json
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python capture_screenshots.py <json_filename>")
        print("Example: python capture_screenshots.py spmv.json")
        sys.exit(1)
    
    json_filename = sys.argv[1]
    
    try:
        capture = DaisenScreenshotCapture()
        output_path = capture.process_json(json_filename)
        print(f"\n{'='*60}")
        print(f"SUCCESS! Screenshots saved to:")
        print(f"{output_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
