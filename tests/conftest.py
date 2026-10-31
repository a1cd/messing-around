import os
import sys

# Make the repository root importable so `import drivesim` works however pytest is invoked.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
