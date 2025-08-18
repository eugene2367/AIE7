"""
Diagnostic script to check A2A server endpoints and configuration.

This script helps diagnose issues with the A2A server by checking:
1. Available endpoints
2. Assistant information
3. Server status
"""

import asyncio
import httpx
import json
from typing import Dict, Any


async def check_server_status(base_url: str = "http://localhost:10000"):
    """Check basic server status."""
    print(f"🔍 Checking A2A server at {base_url}")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        # Check root endpoint
        try:
            response = await client.get(f"{base_url}/", timeout=5)
            print(f"Root endpoint (/): {response.status_code} - {response.reason_phrase}")
            if response.status_code == 405:
                print("  → This is expected - root endpoint doesn't support GET")
        except Exception as e:
            print(f"Root endpoint (/): Error - {e}")
        
        # Check available endpoints
        endpoints_to_check = [
            "/v1/assistants",
            "/v1/tasks",
            "/health",
            "/docs",
            "/openapi.json"
        ]
        
        print("\n📋 Checking available endpoints:")
        for endpoint in endpoints_to_check:
            try:
                response = await client.get(f"{base_url}{endpoint}", timeout=5)
                print(f"  {endpoint}: {response.status_code} - {response.reason_phrase}")
                
                if response.status_code == 200 and endpoint == "/v1/assistants":
                    try:
                        data = response.json()
                        print(f"    Response: {json.dumps(data, indent=2)}")
                    except:
                        print(f"    Response: {response.text[:200]}...")
                        
            except Exception as e:
                print(f"  {endpoint}: Error - {e}")


async def check_assistant_info(base_url: str = "http://localhost:10000"):
    """Check available assistants using JSON-RPC."""
    print(f"\n🤖 Checking available assistants at {base_url}")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        try:
            # Use JSON-RPC format to get agent card
            payload = {
                "jsonrpc": "2.0",
                "method": "agent/getAuthenticatedExtendedCard",
                "params": {},
                "id": 1
            }
            
            response = await client.post(f"{base_url}/", json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Agent info accessible")
                print(f"Response: {json.dumps(data, indent=2)}")
                
                # Extract assistant ID if available
                if "result" in data:
                    result = data["result"]
                    if isinstance(result, dict) and 'id' in result:
                        print(f"\n🎯 Suggested assistant_id: {result['id']}")
                        return result['id']
                    elif isinstance(result, dict) and 'assistant_id' in result:
                        print(f"\n🎯 Suggested assistant_id: {result['assistant_id']}")
                        return result['assistant_id']
                elif "error" in data:
                    print(f"❌ A2A server error: {data['error']}")
                else:
                    print(f"❌ Unexpected response format")
                    
            else:
                print(f"❌ Agent info request returned {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error checking agent info: {e}")
    
    return None


async def test_task_creation(base_url: str = "http://localhost:10000", assistant_id: str = None):
    """Test creating a task with the A2A server."""
    print(f"\n🧪 Testing task creation at {base_url}")
    print("=" * 50)
    
    if not assistant_id:
        print("❌ No assistant ID available, skipping task creation test")
        return
    
    async with httpx.AsyncClient() as client:
        # Test payload using JSON-RPC format
        import uuid
        message_id = str(uuid.uuid4())
        
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "assistant_id": assistant_id,
                "message": {
                    "messageId": message_id,
                    "role": "user",
                    "parts": [{
                        "type": "text",
                        "text": "Hello, this is a test message"
                    }]
                }
            },
            "id": 1
        }
        
        print(f"Creating task with assistant_id: {assistant_id}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = await client.post(
                f"{base_url}/",
                json=payload,
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            if response.status_code in [200, 201]:
                try:
                    data = response.json()
                    print(f"✅ Task created successfully!")
                    print(f"Response: {json.dumps(data, indent=2)}")
                    
                    # Extract task ID for further testing
                    if "result" in data:
                        result = data["result"]
                        task_id = result.get('id')
                        if task_id:
                            print(f"Task ID: {task_id}")
                            await test_task_status(client, base_url, task_id)
                        
                except Exception as e:
                    print(f"Response parsing error: {e}")
                    print(f"Raw response: {response.text}")
            else:
                print(f"❌ Task creation failed")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error creating task: {e}")


async def test_task_status(client: httpx.AsyncClient, base_url: str, task_id: str):
    """Test getting task status using JSON-RPC."""
    print(f"\n📊 Testing task status for task: {task_id}")
    print("-" * 40)
    
    try:
        # Use JSON-RPC format to get task status
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {
                "task_id": task_id
            },
            "id": 1
        }
        
        response = await client.post(f"{base_url}/", json=payload, timeout=10)
        
        print(f"Status response: {response.status_code}")
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ Task status retrieved!")
                print(f"Response: {json.dumps(data, indent=2)}")
            except Exception as e:
                print(f"Status parsing error: {e}")
                print(f"Raw response: {response.text}")
        else:
            print(f"❌ Status retrieval failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error getting task status: {e}")


async def main():
    """Run all diagnostic checks."""
    print("🚀 A2A Server Diagnostic Tool")
    print("=" * 60)
    
    base_url = "http://localhost:10000"
    
    # Check server status
    await check_server_status(base_url)
    
    # Check assistant information
    assistant_id = await check_assistant_info(base_url)
    
    # Test task creation with the default assistant_id "agent"
    print("\n🧪 Testing task creation with default assistant_id 'agent'")
    await test_task_creation(base_url, "agent")
    
    print("\n🔍 Diagnostic complete!")
    print("\nNext steps:")
    print("1. If you found an assistant_id, update the test scripts")
    print("2. Check server logs for any error messages")
    print("3. Verify the A2A server configuration")


if __name__ == "__main__":
    asyncio.run(main()) 