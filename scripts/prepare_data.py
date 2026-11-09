"""Recover byte-identical locked cohorts from the pinned official Parquet."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
URL='https://huggingface.co/datasets/pminervini/NQ-Swap/resolve/3801dbf037c6f58b0e79246f2a8073ebe5656dad/data/dev-00000-of-00001.parquet'
DIGEST='7114c7d1c8a7343f7754b12487272b3c9cf5a5ba0f5fce298c0b8d9a6130fe4c'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parquet',type=Path);a=parser.parse_args()
    path=a.parquet or ROOT/'data/raw/nqswap_fixed_v2/dev.parquet'
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(URL,timeout=120) as response:path.write_bytes(response.read())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==DIGEST,'Official data checksum mismatch'
    import duckdb
    connection=duckdb.connect();cursor=connection.execute('SELECT * FROM read_parquet(?)',[str(path)])
    columns=[c[0] for c in cursor.description];source=[dict(zip(columns,row)) for row in cursor.fetchall()]
    assert len(source)==4746
    for p in sorted((ROOT/'resources').glob('nqswap*/*.metadata.json')):
        spec=json.loads(p.read_text());rows=[]
        for metadata in spec['rows']:
            row=dict(source[metadata['source_row_idx']]);row.update(metadata);rows.append(row)
        data=''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+spec['newline'] for row in rows).encode()
        assert hashlib.sha256(data).hexdigest()==spec['sha256'],f'Reconstruction differs: {p.name}'
        out=ROOT/spec['output'];out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
        print(out.relative_to(ROOT),len(rows),'checksum verified')
if __name__=='__main__':main()
