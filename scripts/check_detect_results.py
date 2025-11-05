from pathlib import Path
import os

root = Path.cwd() / "artifacts" / "uploaded_inputs"
print("Scanning:", root)
found = []
if root.exists():
    for p in root.rglob("*"):
        if p.is_dir():
            has_report = (p / "report.parquet").exists() or (p / "report.csv").exists()
            has_imgsum = (p / "images_summary.csv").exists()
            if has_report and has_imgsum:
                found.append(str(p.resolve()))
            else:
                # ordered_paths fallback
                if has_report and (p / "artifacts" / "ordered_paths.txt").exists():
                    found.append(str(p.resolve()))
else:
    print("No uploaded_inputs directory found")

print("Found result dirs:")
for d in found:
    print(d)
