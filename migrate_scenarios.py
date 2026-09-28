import json
from pathlib import Path

def migrate(path):
    p = Path(path)
    with p.open('r') as f:
        data = json.load(f)
    
    if "floor_plan" not in data:
        data["floor_plan"] = {
            "grid": data.pop("grid", {"rows": 10, "cols": 10}),
            "walls": data.pop("walls", [])
        }
    if "exits" in data:
        data["exit_configuration"] = {
            "configuration_id": "reference",
            "exits": data.pop("exits")
        }
    if "agents" in data:
        data["occupants"] = {
            "agents": data.pop("agents")
        }
        
    with p.open('w') as f:
        json.dump(data, f, indent=4)
    print(f'Migrated {p.name}')

for f in Path('scenarios').glob('*.json'):
    migrate(f)
