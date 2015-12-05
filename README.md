Streaming Anomaly Detection
===

## Description

My implementation of the algorithms described in:

- **[Huang and Kasiviswanathan]**<br>H. Huang and S. Kasiviswanathan, "Streaming Anomaly Detection Using Randomized Matrix Sketching," http://bit.ly/1FaDw6S.

And I have tried the Cod-RNA experiment and ROC curve-based evaluation, done by the authors.

## Implemented algorithms

- [x] Algorithm 1: **AnomDetect**
	- overall algorithm of this framework
- [ ] Algorithm 2: **GlobalUpdate**
	- incremental SVD-based updating
- [ ] Algorithm 3: **RandSketchUpdate**
	- randomized matrix sketching-based updating
- [x] Algorithm 4: **SketchUpdate**
	- Frequent Directions-based updating

Both for the incremental SVD-based and Frequent Directions-based updating, you can refer [my old experimental work](https://github.com/takuti/incremental-matrix-approximation).
