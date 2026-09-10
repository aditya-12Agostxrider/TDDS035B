"""
Pytest configuration ensuring the project root is in sys.path.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
