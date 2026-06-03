"""Query OSF API for HCEye project files."""
import requests
import json

headers = {'Accept': 'application/json'}

print("Querying OSF API for HCEye project (x8p9b)...")
try:
    r = requests.get('https://api.osf.io/v2/nodes/x8p9b/', headers=headers, timeout=15)
    print(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        print(f"Project: {data['data']['attributes']['title']}")
        print(f"Description: {data['data']['attributes']['description'][:200]}")
        
        # Get children/components
        children_url = data['data']['relationships']['children']['links']['related']['href']
        r2 = requests.get(children_url, headers=headers, timeout=15)
        print(f"\nChildren status: {r2.status_code}")
        
        if r2.status_code == 200:
            children = r2.json()
            print(f"Number of components: {len(children['data'])}")
            
            for child in children['data']:
                cid = child['id']
                title = child['attributes']['title']
                public = child['attributes']['public']
                print(f"\n--- Component: {title} (id={cid}, public={public}) ---")
                
                # Get files for this component
                files_url = child['relationships']['files']['links']['related']['href']
                r3 = requests.get(files_url, headers=headers, timeout=15)
                if r3.status_code == 200:
                    providers = r3.json()['data']
                    for prov in providers:
                        prov_name = prov['attributes']['name']
                        prov_files_url = prov['relationships']['files']['links']['related']['href']
                        print(f"  Provider: {prov_name}")
                        r4 = requests.get(prov_files_url, headers=headers, timeout=15)
                        if r4.status_code == 200:
                            items = r4.json()['data']
                            print(f"  Files count: {len(items)}")
                            for item in items[:20]:
                                name = item['attributes']['name']
                                kind = item['attributes']['kind']
                                size = item['attributes'].get('size', '?')
                                print(f"    [{kind}] {name} ({size} bytes)")
                        else:
                            print(f"  Files request failed: {r4.status_code}")
                else:
                    print(f"  Storage request failed: {r3.status_code}")
    else:
        print(f"Failed: {r.text[:500]}")
        
except requests.exceptions.Timeout:
    print("Request timed out - network issue")
except Exception as e:
    print(f"Error: {e}")
