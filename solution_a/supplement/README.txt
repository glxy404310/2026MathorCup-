补充分析复现说明

本目录对应论文5.6、6.2的分层Bootstrap、6.5及表B-3。全部分析在主结果完成后设计，仍使用相同250款酒，不是独立外部测试，也未替换原五折主结果或原相关排名。

使用原始赛题附件、原outputs/q1及outputs/q2结果，Python3.13与requirements-lock.txt中的精确依赖。
完整新副本在仓库根目录运行 python solution_a/run_all.py --rebuild，最后会自动建立补充结果。
仅重建本目录时，先完成主实验，并保证solution_a/supplement/results不存在或为空，运行：
python solution_a/run_supplement.py --rebuild
已有结果只读检查：
python solution_a/run_supplement.py
默认总入口run_all.py也会核查补充结果。

q1_comparison.py：固定种子20261024，相同随机三折比较均值、ExtraTrees、逐目标GBR/HGB；补值仅在训练折，保留所有45目标；另做随机整风格留出和20组配对测试缺失掩码。
q1_additional_checks.py：在观察到随机整风格分折样本量不平衡后补做按类别大小平衡的整风格三折；两套结果均保留。含NaN支持实测。
q2_stratified_bootstrap.py：复用原相关/并列排序规则，22风格内分别有放回抽取原样本数，共200次，固定风格构成；保留抽样索引与选择数组。
q2_permutation.py：原五折和固定参数重拟合，验证折内对231个变量分别置换5次，同时置换缺失指示，按原218化学候选排序；相关网络只在训练折建立，连通分量整体置换并逐折记录。批量预测与原始DataFrame置换的75组检查一致。
verify_supplement.py：重新计算8份完整折外预测和40组掩码指标、风格互斥与均值基线，核对200次分层抽样的类计数和9810项频率，并独立重算第1/100/200次；核对10395项置换汇总、原始置换预测、训练模块及输入哈希。

results/q1为模型对照和全部折外预测；results/q2_stratified为分层相关排名；results/q2_permutation为个体变量和模块的置换结果。npz保存掩码预测、抽样索引或置换检查数组，CSV保存人可读指标，JSON说明参数、输入哈希及范围。

误差、折间标准差和正值折数只描述本批样本和指定协议。对同一批酒补分折不能当作新的独立数据；测试掩码不覆盖检出限删失和仪器失效。分层Bootstrap不消除风格混杂，置换会打破相关输入关系，均不能作为因果贡献或实际加料效果。
