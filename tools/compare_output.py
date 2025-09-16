import os, glob, csv

out = r'c:\Users\tjddn\OneDrive\바탕 화면\answer_scan_project\answer_scan_project\output'
csvf = os.path.join(out, 'images_summary.csv')
expect = []
with open(csvf, newline='', encoding='utf-8') as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        expect.append(row[0])

out_log = os.path.join(os.path.dirname(__file__), 'compare_output_result.txt')
with open(out_log, 'w', encoding='utf-8') as log:
    log.write(f'expected count from csv: {len(expect)}\n')
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    found = []
    for e in exts:
        found += glob.glob(os.path.join(out, '**', e), recursive=True)
    log.write(f'found count by glob: {len(found)}\n')
    found_bns = [os.path.basename(p) for p in found]
    missing = [fn for fn in expect if fn not in found_bns]
    log.write(f'missing count: {len(missing)}\n')
    for m in missing:
        log.write(f'MISSING: {m}\n')
    from collections import Counter
    cnt = Counter(found_bns)
    dups = [k for k, v in cnt.items() if v > 1]
    log.write(f'duplicates count: {len(dups)}\n')
    if dups:
        log.write(f'duplicates sample: {dups[:10]}\n')
    extra = [bn for bn in found_bns if bn not in expect]
    log.write(f'extra in filesystem but not in csv: {len(extra)}\n')
    if extra:
        log.write(str(extra[:20]) + '\n')
print('Wrote results to', out_log)
