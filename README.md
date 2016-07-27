Streaming Anomaly Detection
===

## Description

My implementation of the algorithms introduced in:

- **[Huang and Kasiviswanathan]**<br>**[Streaming Anomaly Detection Using Randomized Matrix Sketching](http://dl.acm.org/citation.cfm?id=2850593)**<br>H. Huang and S. Kasiviswanathan<br>*Proceedings of the VLDB Endowment*, 9 (3), pp. 192-203, 2015

Additionally, I have tried the Cod-RNA experiment and ROC-curve-based evaluation as the authors did.

## Implemented algorithms

- [x] Algorithm 1: **AnomDetect**
	- Overall algorithm of this framework
- [x] Algorithm 2: **GlobalUpdate**
	- Incremental-SVD-based exact updating
- [x] Algorithm 3: **RandSketchUpdate**
	- Randomized-matrix-sketching-based fast updating
- [x] Algorithm 4: **SketchUpdate**
	- Matrix sketching (i.e. frequent directions) based deterministic updating

Both for the incremental-SVD-based and frequent-direction-based updating, you can refer [my old experimental work](https://github.com/takuti/incremental-matrix-approximation).

## License

MIT
