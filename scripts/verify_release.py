"""Fail closed on unreviewed files, manuscript content, secrets, or changed research code."""
import hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'resources/public_files.json').read_text())
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    paths=[p for p in tracked if p]
    assert set(paths)<=set(manifest)|{'resources/public_files.json'},'Unreviewed tracked file'
    banned=re.compile(r'(^|/)(paper|manuscript|投稿|分析汇总)(/|$)|manuscript|cover.letter|reviewer.response|kaggle\.json',re.I)
    secret=re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{32,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----')
    for name in paths:
        assert not banned.search(name),f'Forbidden publication path: {name}'
        raw=(ROOT/name).read_bytes()
        assert not secret.search(raw),f'Credential-like content: {name}'
        if Path(name).suffix in ['.tex','.md']:
            assert b'\\documentclass' not in raw and b'\\begin{abstract}' not in raw,f'Possible manuscript: {name}'
        if name in manifest:assert hashlib.sha256(raw).hexdigest()==manifest[name],f'Unreviewed change: {name}'
    source=json.loads((ROOT/'resources/research_source_hashes.json').read_text())
    for name,digest in source.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    print(f'Public scope and secret-pattern checks passed: {len(paths)} tracked files; original code/config/assets unchanged.')
if __name__=='__main__':main()
