import unittest
from pathlib import Path
from .router_weight import configuration,invocation
from experiments.terra_10pct_factor_diagnosis.run import params

class RouterWeightControl(unittest.TestCase):
    def test_only_preregistered_config_change(self):
        for arm in ['OFF','ON']:
            bank=Path('/tmp/router-weight-test')/arm
            old=params(1,arm,bank);new=configuration(arm,bank)
            self.assertEqual(new['sirm_discover_config']['lambda_router'],1.)
            new['sirm_discover_config']['lambda_router']=old['sirm_discover_config']['lambda_router']
            new['sirm_discover_config']['protocol']=old['sirm_discover_config']['protocol']
            self.assertEqual(old,new)
    def test_paired_controls_and_schedule(self):
        bank=Path('/tmp/router-weight-test');off=configuration('OFF',bank);on=configuration('ON',bank)
        self.assertEqual(off['sirm_discover_config']['lambda_sirm'],0.)
        self.assertEqual(on['sirm_discover_config']['lambda_sirm'],1.)
        on['sirm_discover_config']['lambda_sirm']=0.
        self.assertEqual(off,on)
        cmd=invocation('ON',bank)
        self.assertEqual(cmd[cmd.index('--steps')+1],'501')
        self.assertTrue(off['diagnostic_defer_target_eval'])

if __name__=='__main__':unittest.main()
