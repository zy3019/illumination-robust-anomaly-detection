import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from anomaly.data import file_hash
from anomaly.metrics import auroc
from scripts.final_metrics import exact_pixel_auroc
from scripts.final_protocol import load_locked,environment
from scripts.evaluate_frozen_model import evaluate

class FinalEvaluationTests(unittest.TestCase):
    def test_exact_auc_matches_reference_and_pairwise_with_ties(self):
        rng=np.random.default_rng(91)
        for n in [2,17,3000]:
            labels=rng.integers(0,2,n);labels[:2]=[0,1]
            scores=rng.integers(-5,6,n).astype(np.float32)
            self.assertAlmostEqual(exact_pixel_auroc(labels,scores),auroc(labels,scores),places=14)
            if n<100:
                p=scores[labels==1]; q=scores[labels==0]
                expected=np.mean((p[:,None]>q)+.5*(p[:,None]==q))
                self.assertAlmostEqual(exact_pixel_auroc(labels,scores),expected)
        self.assertIsNone(exact_pixel_auroc([0,0],[1,2]))
        with self.assertRaises(ValueError):exact_pixel_auroc([0,1],[np.nan,1])

    def test_final_access_rejected_without_lock_before_io(self):
        with self.assertRaises(ValueError):evaluate({}, {}, 'must_not_be_created')
        self.assertFalse(Path('must_not_be_created').exists())

    def test_digest_asset_change_and_unknown_job_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[]
            for category in ['bottle','metal_nut','tile']:
                for label in [0,1]:
                    rows.append(dict(category=category,path=f'{category}/{label}.png',split='final_test',
                                     label=label,defect='bad' if label else 'good',sha256=f'{category}_{label}'))
            (root/'manifest.json').write_text(json.dumps({'rows':rows}))
            (root/'asset.txt').write_text('original')
            jobs=[]
            for category in ['bottle','metal_nut','tile']:
                for n in [5,10,25,'all']:
                    for seed in [17,42,2026]:
                        for method in (['A','B','C'] if n in [5,25] else ['A','B']):
                            jobs.append(dict(id=f'{category}_{n}_{method}_{seed}',config=dict(category=category,normal_images=n,
                                seed=seed,method=method,evaluation_split='dev'),frozen_files={}))
            value=dict(status='locked',evaluation_split='final_test',environment=environment(),source_sha256={},
                       assets_sha256={'asset.txt':file_hash(root/'asset.txt')},manifest='manifest.json',jobs=jobs,
                       final_samples={c:[r for r in rows if r['category']==c] for c in ['bottle','metal_nut','tile']})
            path=root/'protocol.json';path.write_text(json.dumps(value));digest=file_hash(path)
            load_locked(path,digest,root=root)
            with self.assertRaises(ValueError):load_locked(path,'0'*64,root=root)
            with self.assertRaises(ValueError):load_locked(path,digest,job_id='absent',root=root)
            (root/'asset.txt').write_text('changed')
            with self.assertRaises(ValueError):load_locked(path,digest,root=root)
            (root/'asset.txt').write_text('original')
            value['jobs'].pop();path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):load_locked(path,file_hash(path),root=root)

if __name__=='__main__':unittest.main()
