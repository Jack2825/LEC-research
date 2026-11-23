"""Export Figure 5I from the completed experimental artwork 5I-J.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from import_supplied_panel import export_panel

if __name__ == "__main__":
    export_panel("5I")
