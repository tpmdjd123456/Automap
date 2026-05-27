\# Pipeline Inspection Commands



Useful commands to inspect the pipeline outputs at each stage.

Run all commands from the repo root with the virtual environment active.



\---



\## 0. Setup



```bash

cd C:\\Users\\dertan\\Automap

.venv\\Scripts\\activate

```



\---



\## 1. Input Data



\*\*How many tables in the data chunk:\*\*

```bash

python -c "lines=open('dev\_chunks/chunk\_1.json').readlines(); print('Number of tables:',len(lines))"

```



\*\*File sizes:\*\*

```bash

dir dev\_chunks\\

```



\---



\## 2. WP1 — PMI Coherence Filtering



\*\*How many tables and columns survived filtering:\*\*

```bash

python -c "import json; lines=open('output\_mini/filtered\_corpus.jsonl').readlines(); cols=sum(len(json.loads(l)\['relation']) for l in lines); print('Tables after filtering:',len(lines)); print('Columns after filtering:',cols)"

```



\---



\## 3. WP2 — FD Filtering



\*\*How many candidates were found:\*\*

```bash

python -c "lines=open('output\_mini/candidates.jsonl').readlines(); print('Total candidates:',len(lines))"

```



\---



\## 4. WP3 — Table Synthesis



\*\*Total synthesized mappings:\*\*

```bash

python -c "lines=open('output\_mini/synthesized\_mappings.jsonl').readlines(); print('Total mappings:',len(lines))"

```



\*\*First 10 mappings with samples:\*\*

```bash

python -c "import json; lines=open('output\_mini/synthesized\_mappings.jsonl').readlines(); print('Total mappings:',len(lines)); \[print('\\nMapping',m\['partition\_id'],':',m\['size'],'pairs from',m\['num\_source\_tables'],'tables\\n sample:',m\['pairs']\[:2]) for m in \[json.loads(l) for l in lines\[:10]]]"

```



\*\*Top 5 largest mappings:\*\*

```bash

python -c "import json; lines=open('output\_mini/synthesized\_mappings.jsonl').readlines(); data=\[json.loads(l) for l in lines]; data.sort(key=lambda x:x\['size'],reverse=True); \[print('\\nMapping',m\['partition\_id'],':',m\['size'],'pairs from',m\['num\_source\_tables'],'tables\\n sample:',m\['pairs']\[:2]) for m in data\[:5]]"

```



\---



\## 5. WP4 — Conflict Resolution



\*\*Summary of conflict resolution:\*\*

```bash

python -c "import json; lines=open('output\_mini/resolved\_mappings.jsonl').readlines(); data=\[json.loads(l) for l in lines]; print('Total mappings:',len(data)); print('Total pairs:',sum(m\['size'] for m in data)); print('Total conflicts removed:',sum(m\['num\_conflicts\_removed'] for m in data))"

```



\*\*Mappings where conflicts were removed:\*\*

```bash

python -c "import json; lines=open('output\_mini/resolved\_mappings.jsonl').readlines(); \[print('\\nMapping',m\['partition\_id'],': removed',m\['num\_conflicts\_removed'],'pairs, kept',m\['size'],'\\n sample:',m\['pairs']\[:2]) for m in \[json.loads(l) for l in lines] if m\['num\_conflicts\_removed']>0]"

```



\*\*Top 5 largest final mappings:\*\*

```bash

python -c "import json; lines=open('output\_mini/resolved\_mappings.jsonl').readlines(); data=\[json.loads(l) for l in lines]; data.sort(key=lambda x:x\['size'],reverse=True); \[print('\\nMapping',m\['partition\_id'],':',m\['size'],'pairs, removed',m\['num\_conflicts\_removed'],'conflicts\\n sample:',m\['pairs']\[:2]) for m in data\[:5]]"

```



\---



\## 6. Full Pipeline



\*\*Run the full pipeline on mini dataset:\*\*

```bash

python main.py --corpus\_path dev\_chunks/chunk\_1\_mini.json --output\_folder output\_mini/ --threshold 0.3 --theta 0.95 --index\_path output\_mini/cooccurrence\_index.pkl

```



\*\*Run the full pipeline on sample data:\*\*

```bash

python main.py --corpus\_path data/sample.json --output\_folder output/ --threshold 0.3 --theta 0.95 --index\_path output/cooccurrence\_index.pkl

```



\---



\## 7. Run Tests



\*\*Run all tests:\*\*

```bash

python -m pytest -v

```



\*\*Run only conflict resolution tests:\*\*

```bash

python -m pytest tests/test\_conflict\_resolution.py -v

```

