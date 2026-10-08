import unittest
from scripts.development_statistics import seed_summary, paired_differences, macro_categories, condition_groups


class DevelopmentStatisticsTests(unittest.TestCase):
    def test_paired_std_preserves_seed_correlation(self):
        rows=[]
        for seed, baseline in zip([17,42,2026],[0.1,0.4,0.7]):
            for method, shift in [('A',0),('B',0.05)]:
                rows.append(dict(category='bottle',budget='10',seed=seed,condition='clean',method=method,score=baseline+shift))
        pairs=paired_differences(rows,['score'])
        result=seed_summary(pairs,['category','budget','condition','comparison'],['score'])[0]
        self.assertAlmostEqual(result['score_mean'],0.05)
        self.assertAlmostEqual(result['score_std'],0)
        a=seed_summary([r for r in rows if r['method']=='A'],['category','budget','condition'],['score'])[0]
        self.assertAlmostEqual(a['score_std'],0.3)

    def test_duplicate_and_missing_seeds_rejected(self):
        with self.assertRaises(ValueError):
            seed_summary([dict(seed=s,score=1) for s in [17,17,2026]],[],['score'])
        with self.assertRaises(ValueError):
            seed_summary([dict(seed=s,score=1) for s in [17,42]],[],['score'])

    def test_missing_ablation_rejected(self):
        rows=[dict(category='bottle',budget='5',seed=17,condition='clean',method=m,score=1) for m in ['A','B']]
        with self.assertRaises(ValueError):
            paired_differences(rows,['score'])

    def test_macro_within_seed_before_sd(self):
        rows=[]
        for seed, values in zip([17,42,2026],[(0,1,2),(2,1,0),(1,1,1)]):
            rows.extend(dict(seed=seed,category=c,score=v) for c,v in zip(['bottle','metal_nut','tile'],values))
        macro=macro_categories(rows,['seed'],['score'])
        result=seed_summary(macro,[],['score'])[0]
        self.assertEqual(result['score_mean'],1)
        self.assertEqual(result['score_std'],0)
        with self.assertRaises(ValueError):
            macro_categories(rows[:-1],['seed'],['score'])


    def test_severity_uses_all_conditions_once(self):
        conditions=[dict(name='clean',operation='clean',factor=1)]
        conditions += [dict(name=f'{op}_{factor}',operation=op,factor=factor)
                       for op in ['brightness','contrast'] for factor in [.9,1.1,.7,1.3,.5,1.5]]
        rows=[dict(seed=17,condition=c['name'],score=0 if c['name']=='clean' else c['factor']) for c in conditions]
        groups=condition_groups(rows,['seed'],['score'],conditions)
        self.assertEqual(next(r for r in groups if r['severity']=='clean')['score'],0)
        for row in groups:
            if row['severity']!='clean': self.assertAlmostEqual(row['score'],1)
        with self.assertRaises(ValueError):
            condition_groups(rows[:-1],['seed'],['score'],conditions)
        with self.assertRaises(ValueError):
            condition_groups(rows+[rows[-1]],['seed'],['score'],conditions)


    def test_full_design_pair_counts_and_ablations(self):
        rows=[]
        for category in ['bottle','metal_nut','tile']:
            for budget in ['5','10','25','all']:
                for seed in [17,42,2026]:
                    for condition in range(13):
                        methods=['A','B','C'] if budget in ['5','25'] else ['A','B']
                        for method in methods:
                            rows.append(dict(category=category,budget=budget,seed=seed,condition=str(condition),method=method,
                                             score={'A':1,'C':2,'B':4}[method]+seed*.001))
        self.assertEqual(len(rows),1170)
        pairs=paired_differences(rows,['score'])
        self.assertEqual(len(pairs),936)
        for row in pairs:
            self.assertAlmostEqual(row['score'],{'B-A':3,'C-A':1,'B-C':2}[row['comparison']])
        summary=seed_summary(pairs,['category','budget','condition','comparison'],['score'])
        self.assertEqual(len(summary),312)
        for row in summary:
            self.assertAlmostEqual(row['score_std'],0)


if __name__=='__main__':
    unittest.main()
