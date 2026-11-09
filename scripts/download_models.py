"""Fetch official pinned base-model files without redistributing them."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen3b','qwen7b'],required=True)
    p.add_argument('--metadata-only',action='store_true');a=p.parse_args()
    if a.model=='qwen3b':
        repo='Qwen/Qwen2.5-3B-Instruct';spec=json.loads((ROOT/'configs/sources.json').read_text())[repo]
        hashes=spec['sha256']
    else:
        spec=json.loads((ROOT/'configs/phase4_qwen7b_source.json').read_text());repo=spec['model']
        hashes={**spec['weight_sha256'],**spec.get('metadata_sha256',{})}
    target=ROOT/'models'/repo.rsplit('/',1)[1]
    from huggingface_hub import snapshot_download
    patterns=['config.json','generation_config.json','tokenizer*.json','special_tokens_map.json','vocab.json','merges.txt','model.safetensors.index.json','LICENSE','README.md']
    if not a.metadata_only:patterns+=['*.safetensors']
    snapshot_download(repo,revision=spec['revision'],local_dir=target,allow_patterns=patterns)
    for name,digest in hashes.items():
        if a.metadata_only and name.endswith('.safetensors'):continue
        h=hashlib.sha256()
        with (target/name).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        assert h.hexdigest()==digest,name
    print(target)
    print('Metadata only; model inference is not available.' if a.metadata_only else 'Official files downloaded; recorded checksums verified.')
if __name__=='__main__':main()
