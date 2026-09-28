# PACS support EDA — target art_painting

Status: running; latest step 500/5000; seed 0.

Source domains: cartoon, photo, sketch.
Class indices: 0=dog, 1=elephant, 2=giraffe, 3=guitar, 4=horse, 5=house, 6=person.

Checkpoint selection uses mean source-validation accuracy. Target statistics appear only after selection.

MMD estimates may be negative. Empirical candidate gaps are probe-specific, not population separation guarantees. Feature statistics and routing utilization are diagnostics, not new losses.

![training.png](training.png)

![structure.png](structure.png)

![expert_domain.png](expert_domain.png)

![class_calibration.png](class_calibration.png)

![feature_collapse.png](feature_collapse.png)

![validation_diagnostics.png](validation_diagnostics.png)

![gradients.png](gradients.png)

![structure_information.png](structure_information.png)
