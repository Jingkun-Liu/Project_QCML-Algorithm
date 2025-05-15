# 对本项目代码结构进行分析
共六个数据集，因此会有六个LSTM文件夹，每个文件夹会有两份代码，train + HybridModel train_c + Model_c 以及 train_q + Model_q
其中 train + HybridModel 跑混合模型，量子比特暂时设为8，原因1是16个量子比特本地跑不了，原因2是大部分数据集特征数小于8
train_c + Model_c 跑纯经典的LSTM
train_q + Model_q 跑纯量子的LSTM

后续可能会加一个 train_bayes ，能够和 HybridModel 兼容。

对于不同的数据集，应该如何修改代码？在这里列举几点要修改的部分。
对于 train 文件(从上往下)：
1. 读取csv文件处，需要修改csv文件路径
2. 选择多特征处，修改data内部代号，看csv文件第一行
3. 最后存储测试数据时，需要修改dataFrame的column