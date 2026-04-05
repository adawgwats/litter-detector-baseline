# Distillation Strategy For Trash Models

Current as checked on `2026-03-31`.

This note answers a practical question:

- how do we use a larger model to make a smaller on-robot model better?

It is written for the current `v1.5` stack:

- training on the PC
- deployment on `Raspberry Pi 5`
- `ONNX Runtime` at inference time

## Bottom Line

The right way to use a large model is **not** to run it on the robot.

The right way is:

1. use a large offline `teacher`
2. turn its outputs into structured supervision
3. train a smaller `student` detector on that supervision plus real labels
4. quantize and benchmark the student on the `Pi 5`

For this project, the teacher should give us:

- better labels
- soft probabilities
- better hard-negative mining
- richer class grouping
- `pick / skip / flag / unknown` decisions

The student should only learn the parts that matter on-robot:

- object location
- coarse waste class
- pickup-worthiness

## A. What The Teacher Should Be

The teacher does not have to be one model.

In practice the best teacher is often a stack:

- a larger object detector
- optionally a segmentation model
- optionally a vision-language model for ambiguity review
- optional geometry or anomaly side-signals

For this project, good teacher roles are:

- high-quality litter detection on offline image batches
- pet-waste vs natural-organic disambiguation
- open-vocabulary review for unknown objects
- relabeling ambiguous false positives

The teacher lives:

- on the training PC
- or in offline batch jobs

It does **not** belong in the critical robot loop.

## B. What The Student Should Be

The student should be the thing we can actually deploy.

That means:

- compact detector
- low-latency
- stable under quantization
- robust enough for real park camera angles

The student output should be intentionally coarse:

- `cigarette_butt_cluster`
- `wrapper_film`
- `cup_lid_straw`
- `bottle`
- `can`
- `paper_service`
- `bagged_waste`
- `pet_waste_likely`
- `natural_organic`
- `bulky_non_target`
- `hazard_unknown`

And then one more derived decision layer:

- `pick`
- `skip`
- `flag`
- `unknown`

That keeps the edge model useful without asking it to solve every semantic question directly.

## C. What To Distill

Do **not** think only in terms of copying final labels.

The larger model can teach the smaller model several different things:

## 1. Soft class probabilities

This is the classic distillation path from Hinton et al.:

- teacher predicts a probability distribution
- student learns the teacher's softer view of confusing classes

Why this matters for us:

- a crushed can may look partly like litter film or cup
- mud may look partly like pet waste
- a leaf clump may look partly like trash

Those "wrong-but-plausible" probabilities contain useful structure.

## 2. Box proposals and pseudo-labels

The teacher can run over unlabeled robot video and produce:

- candidate boxes
- class scores
- uncertainty scores

We then:

- keep high-confidence boxes as pseudo-labels
- send low-confidence cases to manual review

This is how we turn lots of cheap raw field video into more training data.

## 3. Feature-level distillation

For detectors, logits alone are often not enough.

A larger detector can also teach:

- intermediate feature maps
- attention to foreground regions
- instance-level representations

This is especially relevant for object detection, where the model must learn both:

- what the object is
- where it is

## 4. Privileged information

The teacher can see signals the deployed student will not have at runtime.

Examples:

- segmentation masks
- depth estimates
- offboard anomaly scores
- text explanations from a VLM
- operator intervention outcomes
- later, smell-module evidence

The student should not copy the raw explanation text.

Instead, we convert privileged information into structured targets such as:

- refined labels
- uncertainty flags
- hazard flags
- pickup-worthiness scores

## D. Best Distillation Pattern For This Robot

The best pattern is:

### Stage 1. Train a strong teacher offline

Use:

- public datasets
- our own field images
- richer class taxonomy than the student may ultimately keep

The teacher can be slower and larger because it is not deployed.

### Stage 2. Run the teacher over unlabeled field footage

Use the teacher on:

- park footage
- dog-area footage
- event-spillover footage
- false-positive-rich scenes

Save:

- boxes
- scores
- soft class vectors
- uncertainty

### Stage 3. Human-in-the-loop triage

Do not trust pseudo-labels blindly.

Use human review on:

- high-impact classes
- hazardous candidates
- high-uncertainty cases
- repeated false positives

### Stage 4. Train a compact student

Train the student on:

- hard labels from curated data
- soft targets from the teacher
- class-balanced data
- lots of hard negatives

### Stage 5. Quantize and deploy

Export to `ONNX`.

Then:

- calibrate with representative robot images
- use static int8 quantization for the CNN detector path
- benchmark on the actual `Pi 5`

### Stage 6. Close the loop

After deployment, mine:

- misses
- false pickups
- skipped true trash
- biological false positives
- lighting / weather failures

Then retrain teacher and student again.

## E. Where Large Models Help Most

The large model is most useful for:

## 1. Hard-negative mining

This is probably the highest-value use.

The teacher can help identify confusing non-trash examples such as:

- leaves
- mulch
- dark wet patches
- pine straw
- toys
- shadows
- roots
- bark clumps

These matter more than impressive demo accuracy on easy litter.

## 2. Taxonomy compression

The teacher can learn a richer label space, for example:

- soda can
- beer can
- water bottle
- coffee cup
- napkin
- candy wrapper
- dog poop
- goose droppings
- roadkill-like

The student does not need all of those.

It can compress them into:

- `cigarette_butt_cluster`
- `wrapper_film`
- `cup_lid_straw`
- `bottle`
- `can`
- `paper_service`
- `bagged_waste`
- `pet_waste_likely`
- `natural_organic`
- `bulky_non_target`
- `hazard_unknown`

That is exactly the kind of information distillation is good at.

## 3. Unknown-object handling

A larger vision-language model or larger detector can review weird cases offline and answer questions like:

- is this likely trash?
- is this biological?
- is this animal remains?
- is this a dangerous object?

Again, the answer that reaches the student should be structured, not free-form text.

## F. What Not To Do

Do **not**:

- try to distill a chat model directly into a detector
- train the student only on pseudo-labels
- keep a huge fine-grained taxonomy on the robot
- skip real field calibration for quantization
- put the teacher into the real-time pickup loop

Those are the common ways to make the system look advanced while actually making deployment worse.

## G. Most Practical First Implementation

If we wanted the simplest version that still works, I would do this:

1. train a strong offline detector on `TACO + ScatSpotter + our field images`
2. use it to pseudo-label unlabeled park video
3. manually review the uncertain and hazardous cases
4. train a small student detector for the compact pickup taxonomy and a derived `pick / skip / flag / unknown` policy
5. export to `ONNX`
6. static-int8 quantize
7. benchmark latency and accuracy on the `Pi 5`

That is the fastest path to a deployable distilled model.

## H. Why This Is Well Founded

The basic ideas here are not speculative:

- Hinton et al. show that a small model can be trained on soft targets from a cumbersome model, including on a transfer set that can reuse the original training set or unlabeled data.
- Lopez-Paz et al. frame distillation and privileged information together, which matches our use of richer offline signals to supervise a smaller runtime model.
- General Instance Distillation and related detector papers show that object detectors benefit from instance-level and feature-level teacher guidance, not just final hard labels.
- ONNX Runtime explicitly supports static and dynamic quantization, and recommends static quantization for CNN-style models.

## Sources

- Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network*: https://www.cs.toronto.edu/~hinton/absps/distillation.pdf
- Lopez-Paz et al., *Unifying Distillation and Privileged Information*: https://arxiv.org/abs/1511.03643
- Wang et al., *General Instance Distillation for Object Detection*: https://arxiv.org/abs/2103.02340
- ONNX Runtime quantization docs: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

## Internal References

- original companion note: `dregsbane-product/docs/engineering/software_stack_and_ip_strategy.md`
- original companion note: `dregsbane-product/docs/engineering/reusable_dataset_inventory.md`

Those companion references remain private and are not part of this public repo.
