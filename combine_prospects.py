#!/usr/bin/env python3
import json

INPUT_FILE = "combined_prospects.jsonl"
OUTPUT_FILE = "cleaned_prospects.jsonl"

# Full hitting + pitching stat fields
HITTING_STATS = [
    "G","AB","PA","H","1B","2B","3B","HR","R","RBI","BB","IBB","SO","HBP",
    "SF","SH","GDP","SB","CS","AVG","BB%","K%","BB/K","OBP","SLG","OPS","ISO",
    "Spd","BABIP","wSB","wRC","wRAA","wOBA","wRC+","GB/FB","LD%","GB%","FB%",
    "IFFB%","HR/FB","Pull%","Cent%","Oppo%","SwStr%","Balls","Strikes","Pitches"
]

PITCHING_STATS = [
    "W","L","ERA","G","GS","CG","ShO","SV","BS","IP","TBF","H","R","ER","HR",
    "BB","IBB","HBP","WP","BK","SO","K/9","BB/9","K/BB","HR/9","K%","BB%","K-BB%",
    "AVG","WHIP","BABIP","LOB%","ERA_y","FIP","E-F","xFIP","GB/FB","LD%","GB%","FB%",
    "IFFB%","HR/FB","Pull%","Cent%","Oppo%","SwStr%","Balls","Strikes","Pitches"
]

def has_any_stats(row, fields):
    """Check if the player has valid stats (based on 'G' not being null)."""
    g_val = row.get("G")
    return g_val is not None

def extract_stats(row, fields):
    """Extract only available stats from a row."""
    stats = {}
    for f in fields:
        val = row.get(f)
        if val is not None:
            stats[f] = val
    return stats

def clean_row(row):
    try:
        # ID and basic info
        player_id = row.get("PlayerId") or row.get("PlayerId_best")
        name = row.get("name_key") or row.get("Name") or row.get("Name_hit")
        team = row.get("Team") or row.get("Org")
        org = row.get("Org")
        level = row.get("Level") or row.get("Current Level")
        position = row.get("role") or row.get("Pos")
        report = row.get("report") or row.get("Report")

        # Bio
        age = row.get("Age") or row.get("__clean_allcols__", {}).get("Age")
        height = row.get("Ht") or row.get("__clean_allcols__", {}).get("Ht")
        weight = row.get("Wt") or row.get("__clean_allcols__", {}).get("Wt")
        bats = row.get("B") or row.get("__clean_allcols__", {}).get("B")
        throws = row.get("T") or row.get("__clean_allcols__", {}).get("T")
        sign_year = row.get("Sign Yr") or row.get("__clean_allcols__", {}).get("Sign Yr")

        # Tools
        hit = row.get("Hit")
        game_power = row.get("Game Pwr")
        raw_power = row.get("Raw Pwr")
        field = row.get("Fld")
        speed = row.get("Spd")
        bat_control = row.get("Bat Ctrl")
        pitch_selection = row.get("Pitch Sel")
        contact_style = row.get("Con Style")
        versatility = row.get("Versa")

        # Embedding
        embedding = row.get("embedding")

        # Decide hitter vs pitcher
        is_pitcher = position and any(x in position.upper() for x in ["P", "LHP", "RHP"])

        # Extract stats (only one side, not both)
        if is_pitcher and has_any_stats(row, PITCHING_STATS):
            stats = extract_stats(row, PITCHING_STATS)
        elif not is_pitcher and has_any_stats(row, HITTING_STATS):
            stats = extract_stats(row, HITTING_STATS)
        else:
            return None  # no valid stats

        # Skip rows missing embedding or stats
        if not stats or not embedding:
            return None

        clean = {
            "player_id": player_id,
            "name": name,
            "team": team,
            "org": org,
            "age": age,
            "level": level,
            "position": position,
            "height": height,
            "weight": weight,
            "bats": bats,
            "throws": throws,
            "sign_year": sign_year,
            "report": report,
            "hit": hit,
            "game_power": game_power,
            "raw_power": raw_power,
            "field": field,
            "speed": speed,
            "bat_control": bat_control,
            "pitch_selection": pitch_selection,
            "contact_style": contact_style,
            "versatility": versatility,
            "stats": stats,
            "embedding": embedding
        }
        return clean

    except Exception as e:
        print(f"Skipping row due to error: {e}")
        return None

def main():
    kept, dropped = 0, 0
    with open(INPUT_FILE, "r") as infile, open(OUTPUT_FILE, "w") as outfile:
        for line in infile:
            if not line.strip():
                continue
            row = json.loads(line)
            clean = clean_row(row)
            if clean:
                outfile.write(json.dumps(clean) + "\n")
                kept += 1
            else:
                dropped += 1

    print(f"✅ Finished. Kept {kept} players, dropped {dropped} players.")
    print(f"Cleaned file written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()