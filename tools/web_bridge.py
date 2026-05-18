import sys
import trafilatura
import json

def fetch_knowledge(url):
    print(f"  [Web Bridge] Fetching: {url}", file=sys.stderr)
    try:
        downloaded = trafilatura.fetch_url(url)
        content = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
        if content:
            return {"url": url, "content": content}
        return {"error": "No content extracted"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No URL provided"}))
        sys.exit(1)
    
    url = sys.argv[1]
    result = fetch_knowledge(url)
    print(json.dumps(result))
