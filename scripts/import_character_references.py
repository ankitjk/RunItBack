#!/usr/bin/env python3
"""Import character reference images from downloaded assets.

This script copies character reference images from the Super Smash Bros Ultimate
Fighters download to the appropriate character_references directories.

Source structure:
    Super Smash Bros Ultimate Fighters/
        1 - Mario/
            chara_1_mario_00.png  -> loading_screen
            chara_2_mario_00.png  -> stock_icon
            chara_4_mario_00.png  -> icon
        ...

Target structure:
    data/character_references/
        icon/
            mario.png
        loading_screen/
            mario.png
        stock_icon/
            mario.png
"""

import shutil
from pathlib import Path

# Source directory
SOURCE_DIR = Path("/Users/ankit/Downloads/Super Smash Bros Ultimate Fighters")

# Target directory
TARGET_DIR = Path(__file__).parent.parent / "data" / "character_references"

# Mapping from folder name to standardized character name
# Format: "folder_name_prefix": "character_name"
FOLDER_TO_CHARACTER = {
    "1 - Mario": "mario",
    "2 - Donkey Kong": "donkey_kong",
    "3 - Link": "link",
    "4 - Samus": "samus",
    "4ε - Dark Samus": "dark_samus",
    "5 - Yoshi": "yoshi",
    "6 - Kirby": "kirby",
    "7 - Fox": "fox",
    "8 - Pikachu": "pikachu",
    "9 - Luigi": "luigi",
    "10 - Ness": "ness",
    "11 - Captain Falcon": "captain_falcon",
    "12 - Jigglypuff": "jigglypuff",
    "13 - Peach": "peach",
    "13ε - Daisy": "daisy",
    "14 - Bowser": "bowser",
    "15 - Ice Climbers": "ice_climbers",
    "16 - Sheik": "sheik",
    "17 - Zelda": "zelda",
    "18 - Dr Mario": "dr_mario",
    "19 - Pichu": "pichu",
    "20 - Falco": "falco",
    "21 - Marth": "marth",
    "21ε - Lucina": "lucina",
    "22 - Young Link": "young_link",
    "23 - Ganondorf": "ganondorf",
    "24 - Mewtwo": "mewtwo",
    "25 - Roy": "roy",
    "25ε - Chrom": "chrom",
    "26 - Mr Game and Watch": "mr_game_and_watch",
    "27 - Meta Knight": "meta_knight",
    "28 - Pit": "pit",
    "28ε - Dark Pit": "dark_pit",
    "29 - Zero Suit Samus": "zero_suit_samus",
    "30 - Wario": "wario",
    "31 - Snake": "snake",
    "32 - Ike": "ike",
    "33 - Pokémon Trainer": "pokemon_trainer",
    "36 - Diddy Kong": "diddy_kong",
    "37 - Lucas": "lucas",
    "38 - Sonic": "sonic",
    "39 - King Dedede": "king_dedede",
    "40 - Olimar (Alph)": "olimar",
    "41 - Lucario": "lucario",
    "42 - ROB": "rob",
    "43 - Toon Link": "toon_link",
    "44 - Wolf": "wolf",
    "45 - Villager": "villager",
    "46 - Mega Man": "mega_man",
    "47 - Wii Fit Trainer": "wii_fit_trainer",
    "48 - Rosalina and Luma": "rosalina_and_luma",
    "49 - Little Mac": "little_mac",
    "50 - Greninja": "greninja",
    "51-53 - Mii": "mii_brawler",  # Use for Mii Brawler
    "54 - Palutena": "palutena",
    "55 - Pac-Man": "pac_man",
    "56 - Robin": "robin",
    "57 - Shulk": "shulk",
    "58 - Bowser Jr (Koopalings)": "bowser_jr",
    "59 - Duck Hunt": "duck_hunt",
    "60 - Ryu": "ryu",
    "60ε - Ken": "ken",
    "61 - Cloud": "cloud",
    "62 - Corrin": "corrin",
    "63 - Bayonetta": "bayonetta",
    "64 - Inkling": "inkling",
    "65 - Ridley": "ridley",
    "66 - Simon": "simon",
    "66ε - Richter": "richter",
    "67 - King K Rool": "king_k_rool",
    "68 - Isabelle": "isabelle",
    "69 - Incineroar": "incineroar",
    "70 - Piranha Plant": "piranha_plant",
    "71 - Joker": "joker",
    "72 - Hero": "hero",
    "73 - Banjo & Kazooie": "banjo_and_kazooie",
    "74 - Terry Bogard": "terry",
    "75 - Byleth": "byleth",
    "76 - Min Min": "min_min",
    "77 - Steve": "steve",
    "78 - Sephiroth": "sephiroth",
    "79 - Pyra & Mythra": "pyra_mythra",
    "80 - Kazuya Mishima": "kazuya",
}

# File prefix to reference type mapping
PREFIX_TO_TYPE = {
    "chara_1_": "loading_screen",
    "chara_2_": "stock_icon",
    "chara_4_": "icon",
}




def get_character_internal_name(folder_name: str) -> str | None:
    """Extract internal character name from folder contents."""
    folder_path = SOURCE_DIR / folder_name
    if not folder_path.exists():
        return None

    # Look for any chara_1_ file to get the internal name
    for f in folder_path.glob("chara_1_*_00.png"):
        # Extract name between chara_1_ and _00.png
        name = f.stem.replace("chara_1_", "").replace("_00", "")
        return name

    return None


def import_references():
    """Import all character references including all skins."""
    # Create target directories
    for ref_type in PREFIX_TO_TYPE.values():
        (TARGET_DIR / ref_type).mkdir(parents=True, exist_ok=True)

    imported_count = 0
    skipped_count = 0

    for folder_name, char_name in FOLDER_TO_CHARACTER.items():
        folder_path = SOURCE_DIR / folder_name

        if not folder_path.exists():
            print(f"Warning: Folder not found: {folder_name}")
            skipped_count += 1
            continue

        # Get internal name for file matching
        internal_name = get_character_internal_name(folder_name)
        if not internal_name:
            print(f"Warning: Could not determine internal name for {folder_name}")
            skipped_count += 1
            continue

        # Copy each type of reference (all skins)
        for prefix, ref_type in PREFIX_TO_TYPE.items():
            # Find all skin variants for this character and type
            pattern = f"{prefix}{internal_name}_*.png"
            source_files = list(folder_path.glob(pattern))

            if not source_files:
                print(f"Warning: Missing {ref_type} for {char_name}")
                continue

            for source_file in source_files:
                # Extract skin number from filename (e.g., _00, _01, etc.)
                skin_suffix = source_file.stem.split("_")[-1]  # Gets "00", "01", etc.

                # Name format: character_skin.png (e.g., mario_00.png, mario_01.png)
                target_file = TARGET_DIR / ref_type / f"{char_name}_{skin_suffix}.png"
                shutil.copy2(source_file, target_file)
                imported_count += 1

            print(f"Copied: {char_name} ({ref_type}) - {len(source_files)} skins")

    print(f"\nImported {imported_count} references")
    if skipped_count:
        print(f"Skipped {skipped_count} folders")


if __name__ == "__main__":
    import_references()
