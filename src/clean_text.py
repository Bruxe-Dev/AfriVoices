"""
show_actual_source.py
=======================
Prints the exact source code of clean_text() as Python actually sees it,
no guessing. Also prints file size and modification time to rule out a
stale/duplicate file confusion.
"""

import inspect
import os
import preprocessing

print("=" * 70)
print("File path :", preprocessing.__file__)
print("File size :", os.path.getsize(preprocessing.__file__), "bytes")
print("Modified  :", os.path.getmtime(preprocessing.__file__))
print("=" * 70)
print("ACTUAL SOURCE CODE of clean_text() as Python sees it:")
print("=" * 70)
print(inspect.getsource(preprocessing.clean_text))