# PACS support EDA — target cartoon

Status: completed; latest step 5000/5000; seed 0.

Source domains: art_painting, photo, sketch.
Class indices: 0=dog, 1=elephant, 2=giraffe, 3=guitar, 4=horse, 5=house, 6=person.

Checkpoint selection uses mean source-validation accuracy. Target statistics appear only after selection.

MMD estimates may be negative. Empirical candidate gaps are probe-specific, not population separation guarantees. Feature statistics and routing utilization are diagnostics, not new losses.

This run uses mixture CE and a support-conditional predictive hinge. The old max-CE prediction bound remains a diagnostic only. Entropy thresholds are fixed from source-training class counts; an empirical hinge of zero does not certify population mutual information.

Best source accuracy: 0.9803; selected step: 3200.
Target accuracy: 0.8306.

![training.png](training.png)

![structure.png](structure.png)

![expert_domain.png](expert_domain.png)

![class_calibration.png](class_calibration.png)

![feature_collapse.png](feature_collapse.png)

![validation_diagnostics.png](validation_diagnostics.png)

![gradients.png](gradients.png)

![structure_information.png](structure_information.png)
