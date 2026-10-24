"""Upload a transcript to a running ActionGraph server and mark the first item complete.

actiongraph serve --db demo.db      # in another terminal
python examples/api_client.py data/samples/atlas_sync_2026-03-09.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"


def main(path: str) -> None:
    p = Path(path)
    with httpx.Client(base_url=BASE, timeout=60) as client:
        r = client.post(
            "/api/meetings",
            files={"file": (p.name, p.read_bytes())},
            data={"series": "atlas-checkout"},
        )
        r.raise_for_status()
        meeting = r.json()
        items = meeting["analysis"]["action_items"]
        print(f"{meeting['meeting_id']}: {len(items)} action items")
        for it in items:
            print(f"  - {it['description']} ({it['owner']}, due {it['deadline']})")
        if items:
            client.patch(
                f"/api/items/{items[0]['id']}", json={"status": "completed"}
            ).raise_for_status()
            print(f"Marked complete: {items[0]['description']}")
        ics = client.get(f"/api/meetings/{meeting['meeting_id']}/export/ics").text
        print(f"ICS export: {ics.count('BEGIN:VEVENT')} calendar events")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/samples/atlas_sync_2026-03-09.json")
