"""Execute the code cells of an original notebook (unchanged) headlessly, from its own folder.
Usage: python run_notebook.py <notebook.ipynb> [max_cell_index]"""
import json, os, sys, re
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close('all')

nb_path = os.path.abspath(sys.argv[1]); stop = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
os.chdir(os.path.dirname(nb_path))
g = {'__name__': '__main__', 'display': print}
for i, c in enumerate(json.load(open(nb_path))['cells']):
    if i > stop:
        break
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    src = '\n'.join(l for l in src.splitlines() if not l.strip().startswith(('%', '!')))
    print(f"\n######## CELL {i} ########", flush=True)
    try:
        exec(compile(src, f'cell{i}', 'exec'), g)
    except Exception as e:
        print(f"!! cell {i} raised {type(e).__name__}: {e}", flush=True)
