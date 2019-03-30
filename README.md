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
(这个就是对数据噪声的处理，修正)
- (flowerlake)存在GPS里程减少的情况，比如在车辆 AB00351 在第28411行数据中，mileage出现减少1的情况。在对100个车辆进行了分析，总共存在6个车辆出现这样的问题(也存在mileage值减少幅度特别大的情况)，分别为 AB00351.csv、AB00465.csv、AD00017.csv、AD00022.csv、AD00077.csv、AF00167.csv 6个文件。
(人为修改里程的问题)
- 补点问题：在58速运的文章中，提到数据修正的 补点需要采用 地图路径规划 的方式进行修正，这个地图路径规划是什么东西？

- 行车特征提取问题：特征的提取，特征的建立，这里的特征说的是什么

- - -

**参考文档**

1. 里程计算的相关思路

58速运“里程计算”优化与演进(http://zhuanlan.51cto.com/art/201708/549224.htm)

GPS里程计算深度分析(http://www.cnlbs.com/4/2010/0426/297.html)

基于野值过滤的GPS统计车辆行驶里程算法的研究(来自知网，在paper文件夹中)

2. GPS轨迹的处理

GPS 数据处理简述(https://zhuanlan.zhihu.com/p/30859356) 处理gps数据噪声的方法(v)

数据压缩问题(https://zhuanlan.zhihu.com/p/51976835)

3. js点播

1、速度是不准确的，要通过里程确定速度，实验给的里程精度太低。速度以s为单位，是算出来的，更不精确

2、经纬度的计算里程（我问的，他说这个经纬度的准确的，精确的），通过这个应该可以算里程，进而算速度

3、GPS的传输延时问题，这个他提了很多次
（这个在写论文的时候要提一下。）
4、前期数据的抛弃，不准确（忘了为什么了，他提的）

5、中间缺省项的插入问题（就是1分钟没有60项数据的问题，他说要补，速度，里程，加速度什么的要平滑且连续，保证值的连续）
（反复验证的过程，行车数据的处理流程有一个循环的过程）

6、他说最重要的就是 特征的提取，特征的建立

7、对于速度，里程对于这些不准确的值，可以相互较验

噪声的处理（利用速度的异常值）

数据点的修正（补点60s、修正）:卡尔曼滤波算法

数据点的压缩（处理数据量过大的问题，去除掉那些坐标不变，车辆处于ACC=0状态下的点）

行车里程的计算（利用gps数据点之间的距离进行计算）