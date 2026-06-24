#!/usr/bin/env python3
"""CLI tool to verify and demo the asynchronous 5-stage receipt processing pipeline."""
import io
import time
import sys
import httpx

API_URL = "http://localhost:8000/api/v1/pipeline"

def print_header(title: str):
    print("=" * 60)
    print(f" {title:^58} ")
    print("=" * 60)

def main():
    print_header("Asynchronous 5-Stage Receipt Processing Pipeline Demo")
    
    # 1. Prepare dummy receipt image bytes
    print("[*] Preparing dummy receipt image...")
    dummy_file = io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08")
    
    # 2. Submit job to the pipeline
    files = {"file": ("demo_receipt.jpg", dummy_file, "image/jpeg")}
    print(f"[*] Submitting job to: {API_URL}/process")
    
    try:
        response = httpx.post(f"{API_URL}/process", files=files, timeout=10.0)
    except httpx.ConnectError:
        print("[!] Error: Could not connect to AI service. Make sure it is running on http://localhost:8000")
        sys.exit(1)
        
    if response.status_code != 200:
        print(f"[!] Submission failed with status {response.status_code}: {response.text}")
        sys.exit(1)
        
    job_info = response.json()
    job_id = job_info["job_id"]
    print(f"[+] Job successfully submitted! Job ID: {job_id}")
    
    # 3. Poll the job status and track progress stages
    print("\n[*] Tracking pipeline stages:")
    last_stage = None
    last_progress = -1
    
    start_time = time.time()
    max_duration = 30.0  # 30 seconds max timeout
    
    while time.time() - start_time < max_duration:
        try:
            status_resp = httpx.get(f"{API_URL}/jobs/{job_id}")
            if status_resp.status_code != 200:
                print(f"\n[!] Failed to get job status: {status_resp.text}")
                sys.exit(1)
                
            job_data = status_resp.json()
            status = job_data["status"]
            stage = job_data["current_stage"]
            progress = job_data["progress"]
            
            if stage != last_stage or progress != last_progress:
                print(f"    --> Stage: {stage:<25} | Progress: {progress:3d}% | Status: {status}")
                last_stage = stage
                last_progress = progress
                
            if status == "completed":
                print("\n[+] Job completed successfully!")
                print("=" * 60)
                print("Result Suggestion:")
                result = job_data["result"]
                extracted = result.get("extracted", {})
                ocr = result.get("ocr", {})
                print(f"    Merchant/Note: {extracted.get('note')}")
                print(f"    Amount:        {extracted.get('amount')} {ocr.get('suggestion', {}).get('currency', 'VND')}")
                print(f"    Category:      {extracted.get('category')}")
                print(f"    Confidence:    {extracted.get('confidence')}")
                print(f"    Backend:       {ocr.get('backend')}")
                print("=" * 60)
                sys.exit(0)
            elif status == "failed":
                print(f"\n[!] Job failed: {job_data.get('error')}")
                sys.exit(1)
                
        except Exception as e:
            print(f"\n[!] Error during status check: {e}")
            sys.exit(1)
            
        time.sleep(0.1)
        
    print(f"\n[!] Timeout: Job did not finish within {max_duration} seconds.")
    sys.exit(1)

if __name__ == "__main__":
    main()
