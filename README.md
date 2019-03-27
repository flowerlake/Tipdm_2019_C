(请在这个文件里加上对代码运行的解释以及工程所使用的技术。)

**说明**

该库目前处于private状态，等到竞赛结束后再公开。

- - -

**目录结构说明**

- data_processing: 包含数据处理的源代码文件

- data_visulization: 包含数据可视化的源代码文件

- data: 包含了样例数据，样例数据只上传一份

- output_data: 包含了输出的数据文件

- PythonCheat: 包含了numpy、pandas、matplotlib的小抄jupyter文件

- - -

**TODO**

完善 mapbox 相关信息

- - -

**issue**

- (flowerlake)尝试了tableau 以及 matplotlib 在直角坐标系中的绘图。如果在直角坐标系中能够发现离群点，可证明该数据为异常数据（是丢弃还是做其他的处理？）
(在这里写下对这个问题的想法)
- (flowerlake)存在GPS里程减少的情况，比如在车辆 AB00351 在第28411行数据中，mileage出现减少1的情况。在对100个车辆进行了分析，总共存在6个车辆出现这样的问题(也存在mileage值减少幅度特别大的情况)，分别为 AB00351.csv、AB00465.csv、AD00017.csv、AD00022.csv、AD00077.csv、AF00167.csv 6个文件。

- - -

**参考文档**

1. 里程计算的相关思路

58速运“里程计算”优化与演进(http://zhuanlan.51cto.com/art/201708/549224.htm)
GPS里程计算深度分析(http://www.cnlbs.com/4/2010/0426/297.html)
基于野值过滤的GPS统计车辆行驶里程算法的研究(来自知网，在paper文件夹中)
