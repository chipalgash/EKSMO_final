import shutil
import subprocess
from pathlib import Path

DOCS_DIR = Path("books")
WORKSPACE = Path("workspace")

for docx_path in DOCS_DIR.glob("*.docx"):
    book_id = docx_path.stem
    raw_dir = WORKSPACE / book_id / "00_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(docx_path, raw_dir / docx_path.name)
    print(f"=== Processing book_id={book_id} ===")
    subprocess.run([
        "python", "run_book.py",
        "--book-id", book_id,
        "--force"
    ], check=True)