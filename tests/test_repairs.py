import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from anomaly.metrics import classification_metrics
from scripts.prepare_data import verify_existing
from scripts.report_integrity import verify_report_inputs
from scripts.frozen_bundle import restore_bundle, sha
import scripts.run_final_matrix as runner


class RepairTests(unittest.TestCase):
    def test_column_vectors_do_not_broadcast(self):
        expected = classification_metrics([0,1],[.1,.9],.5)
        self.assertEqual(classification_metrics([[0],[1]],[.1,.9],.5), expected)
        self.assertEqual(classification_metrics([0,1],[[.1],[.9]],.5), expected)

    def test_invalid_metric_inputs_rejected(self):
        for threshold in [np.nan, np.inf, -np.inf, [.5]]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                classification_metrics([0,1],[.1,.9],threshold)
        for labels,scores in [([0],[.1,.9]),([0,2],[.1,.9]),([0,1],[np.nan,.9])]:
            with self.assertRaises(ValueError):
                classification_metrics(labels,scores,.5)

    def test_existing_manifest_is_read_only_and_detects_mask_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=root/'image.png'; mask=root/'mask.png'
            image.write_bytes(b'image');mask.write_bytes(b'mask')
            row=dict(category='bottle',path='image.png',mask='mask.png',split='dev',defect='bad',label=1,
                     sha256=sha(image),mask_sha256=sha(mask))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(split_seed=42,rows=[row])))
            before=manifest.read_bytes()
            verify_existing(root,manifest,['bottle'],42)
            self.assertEqual(before,manifest.read_bytes())
            with self.assertRaises(ValueError):verify_existing(root,manifest,['tile'],42)
            mask.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_existing(root,manifest,['bottle'],42)

    def test_complete_attempt_adopted_and_corrupt_attempt_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'attempt01';out.mkdir();(out/'integrity.json').write_text('{}')
            entry=dict(status='running',output='attempt01')
            with patch.object(runner,'ROOT',root), patch.object(runner,'verify_result',return_value={}) as verify:
                recovered=runner.recover_completed(entry,{}, {},'digest')
                verify.assert_called_once()
                self.assertEqual(recovered['status'],'complete')
                self.assertEqual(recovered['output'],'attempt01')
                self.assertIsNone(recovered['process_wall_seconds'])
                self.assertEqual(entry['status'],'running')
            with patch.object(runner,'ROOT',root), patch.object(runner,'verify_result',side_effect=ValueError('corrupt')):
                with self.assertRaises(ValueError):runner.recover_completed(entry,{}, {},'digest')
            (out/'integrity.json').unlink()
            with patch.object(runner,'ROOT',root):
                self.assertIsNone(runner.recover_completed(entry,{}, {},'digest'))

    def test_modified_report_table_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);csv=root/'table.csv';csv.write_text('value\n1\n')
            manifest=root/'accepted.json'
            manifest.write_text(json.dumps(dict(status='accepted_inputs',files_sha256={'table.csv':sha(csv)})))
            verify_report_inputs(root,manifest)
            csv.write_text('value\n999\n')
            with self.assertRaisesRegex(ValueError,'Accepted report input changed'):
                verify_report_inputs(root,manifest)

    def test_bundle_restore_hashes_and_non_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'bundle.zip';target=root/'restored'
            data=b'frozen\r\n'
            manifest=dict(protocol_sha256='test',files_sha256={'source.py':hashlib.sha256(data).hexdigest()})
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('source.py',data);z.writestr('BUNDLE.json',json.dumps(manifest))
            with self.assertRaises(ValueError):restore_bundle(archive,'bad',target)
            restore_bundle(archive,sha(archive),target)
            self.assertEqual((target/'source.py').read_bytes(),data)
            with self.assertRaises(FileExistsError):restore_bundle(archive,sha(archive),target)

    def test_bundle_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('../escape',b'x')
                z.writestr('BUNDLE.json',json.dumps(dict(files_sha256={'../escape':hashlib.sha256(b'x').hexdigest()})))
            with self.assertRaises(ValueError):restore_bundle(archive,sha(archive),root/'target')
            self.assertFalse((root/'target').exists())
            self.assertFalse((root/'escape').exists())


if __name__ == '__main__':
    unittest.main()
