import unittest
import torch
from domainbed.support_eda import PredictionEDA

class EDATests(unittest.TestCase):
    def test_confusion_routing_and_collapse(self):
        e = PredictionEDA(2, 2)
        y = torch.tensor([0,1])
        logits = torch.tensor([[[20.,-20.],[20.,-20.]], [[-20.,20.],[-20.,20.]]])
        router = torch.zeros(2,2)
        z = torch.tensor([[[1.,0.],[0.,1.]], [[1.,0.],[0.,1.]]])
        for i in range(2):
            e.add(y[i:i+1], logits[i:i+1].mean(1), logits[i:i+1],router[i:i+1],z[i:i+1])
        r=e.finalize()
        self.assertEqual(r['confusion_matrix'],[[1,0],[0,1]])
        self.assertEqual(r['routing_mean'],[.5,.5])
        self.assertEqual(r['feature_variance_trace'],[0.,0.])
        self.assertAlmostEqual(r['routing_effective_experts'],2.)
        self.assertEqual(r['expert_oracle_accuracy'],1.)
        self.assertLess(r['ece_10bins'],1e-6)
        self.assertLess(r['brier_score'],1e-6)
        self.assertEqual(r['expert_disagreement'],[[0.,0.],[0.,0.]])

    def test_absent_classes_report_null(self):
        e=PredictionEDA(2,3)
        e.add(torch.tensor([0]),torch.zeros(1,3),torch.zeros(1,2,3),torch.zeros(1,2),torch.ones(1,2,2))
        r=e.finalize()
        self.assertEqual(r['class_counts'],[1,0,0])
        self.assertIsNone(r['per_class'][1]['accuracy'])
        self.assertIsNone(r['per_class'][1]['routing_mean'])

if __name__=='__main__':
    unittest.main()
