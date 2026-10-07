import json,tempfile,unittest
from pathlib import Path
from .run import describe,ratio_records,evolution_stats,domainbed_scores

class StatisticsTests(unittest.TestCase):
    def test_percentiles_and_active_vs_zero_distribution(self):
        active=describe([.1,1.]);all_points=describe([0.,.1,1.])
        self.assertAlmostEqual(active['mean'],.55);self.assertAlmostEqual(active['median'],.55);self.assertAlmostEqual(active['p90'],.91)
        self.assertAlmostEqual(all_points['mean'],1.1/3);self.assertAlmostEqual(all_points['p90'],.82)

    def test_missing_active_measurement_is_unknown_not_zero(self):
        def r(active):return dict(stage='invariance',step=2150,sirm_active=active,weighted_sirm_grad_norm=None,base_residual_grad_norm=None,eligible_pair_expert_count=1,effective_lambda_sirm=.03,source_validation_accuracy=.85)
        self.assertIsNone(ratio_records([r(True)])[0]['gradient_ratio']);self.assertEqual(ratio_records([r(False)])[0]['gradient_ratio'],0.)

    def test_weight_changes_and_support_changes_are_distinct(self):
        records=[dict(step=2000,q=[[.1,0],[0,.2]]),dict(step=2050,q=[[.2,0],[0,.2]]),dict(step=2100,q=[[.2,0],[.1,.2]])]
        result=evolution_stats(records)
        self.assertEqual(result['distinct_Q_matrices'],3);self.assertEqual(result['Q_weight_changes'],2)
        self.assertEqual(result['Q_support_changes'],1);self.assertEqual(result['first_shared_support_step'],2100)

    def test_domainbed_oracle_uses_last_complete_row_not_best_or_source_refresh(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);bank=root/'banks/R1';bank.mkdir(parents=True)
            (root/'plan.json').write_text(json.dumps({'target':1}))
            def row(step,test,validation,source):
                r=dict(step=step,args={'test_envs':[1]},source_val_mean=source)
                r.update({f'env{env}_{part}_acc':.8 for env in range(4) for part in ['in','out']})
                r.update(env1_in_acc=test,env1_out_acc=validation)
                return r
            records=[row(0,.8,.9,.3),row(2500,.7,.85,.93),row(2750,.6,.75,.95),dict(step=2800,args={'test_envs':[1]},env0_out_acc=.95)]
            (bank/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            score=domainbed_scores(root)
            self.assertEqual(score['oracle_step'],2750);self.assertEqual(score['oracle']['test_acc'],.6)
            self.assertEqual(score['best_target_validation_all_steps']['step'],0)
            self.assertEqual(score['best_target_validation_stage3']['step'],2500)
            self.assertEqual(score['extra_source_only_rows'],1)

if __name__=='__main__':unittest.main()
