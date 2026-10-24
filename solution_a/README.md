# MathorCup 2026 赛道 A：啤酒风味建模

本目录保存数据审计、模型、原始实验结果和复现程序。第一问已完成45维连续预测与缺失压力实验，第二问已完成关联网络与Top-3稳定性基线，第三问已运行风格生成和带缺陷保护的迁移搜索。真实0–6量尺校准、部分机制核验和实际品评验证仍待完成。

论文定稿采用口径见 [最终采用方案](docs/final_model_selection.md)，工件和结果选择条件见 [final_model_selection.json](final_model_selection.json)。最新实验见 [beer 81诊断与兼顾接近度的迁移搜索](docs/q3_refinement_results.md)。[上轮未见风格探索](docs/q3_domain_results.md)、[生成优化结果](docs/q3_optimization_results.md)、[第三问首轮结果](docs/q3_results.md)保留历史对照。

本次提交文件为[论文PDF](submission/A2600730.pdf)与[程序及结果附件包](submission/A2600730附件.zip)，可编辑源稿保存在[Word文件](paper/啤酒风味建模提交稿.docx)。PDF含完整程序全文、135项化学证据、72组迁移策略表及逐感官折外置换结果。补充分析见[supplement/README.txt](supplement/README.txt)：保留原五折主表，另列三折固定参数比较、整风格留出、多掩码压力和分层Bootstrap。正文保留原始0–6量尺不可恢复、相关与因果区别及尚无实际品评的限制；这些限制不因排版完成而消失。

| 内容 | 位置 |
|---|---|
| 论文正文、30个原生公式、6图及结果附表 | [paper/manuscript.json](paper/manuscript.json) |
| 135项候选的逐行化学证据及最终引用 | [paper/chemical_evidence_final.csv](paper/chemical_evidence_final.csv) |
| 三问主方案、正文对照与结论边界 | [final_model_selection.md](docs/final_model_selection.md) |
| 实际数据结构、缺失、条码精度、变换与题面冲突 | [data_audit.md](docs/data_audit.md) |
| 评分卡、45维映射、原始量尺与官方来源 | [sensory_mapping_review.md](docs/sensory_mapping_review.md) |
| 第一问异常值和折外预测复核 | [q1_validation_review.md](docs/q1_validation_review.md) |
| 第二、三问模型设计与验证边界 | [q2_q3_design_review.md](docs/q2_q3_design_review.md) |
| 第三问生成、画像、迁移与失败案例 | [q3_results.md](docs/q3_results.md) |
| 第三问优化、跨阈值保护及负面结果 | [q3_optimization_results.md](docs/q3_optimization_results.md) |
| 非Pils逐风格留出预测与双委员会搜索 | [q3_domain_results.md](docs/q3_domain_results.md) |
| beer81误差来源、预测备选及迁移权衡 | [q3_refinement_results.md](docs/q3_refinement_results.md) |
| beer81精确连接、标签、化学邻域与影响诊断 | [q3_beer81_audit.md](docs/q3_beer81_audit.md) |
| 稳定性选型、训练辅助监督与尾部备选 | [q3_stale_refined.md](docs/q3_stale_refined.md) |
| 本轮独立审查与全量复算 | [q3_refinement_review.md](docs/q3_refinement_review.md) |
| Stale hops完整风格留出选型 | [q3_stale_domain.md](docs/q3_stale_domain.md) |
| 本轮审查与前轮委员会结构勘误 | [q3_domain_review.md](docs/q3_domain_review.md) |
| 优化生成方法与嵌套验证 | [q3_generation_optimized.md](docs/q3_generation_optimized.md) |
| Stale hops搜索、固定树复核与不确定性 | [q3_stale_optimized.md](docs/q3_stale_optimized.md) |
| 优化数值审查 | [q3_optimization_review.md](docs/q3_optimization_review.md) |
| 第三问数据表示与感官预测器 | [q3_representation.md](docs/q3_representation.md) |
| 第三问生成分布验证与敏感性 | [q3_generation.md](docs/q3_generation.md) |
| 第三问数值与方法审查 | [q3_method_review.md](docs/q3_method_review.md) |
| 第一问折外预测、指标和冻结模型 | `outputs/q1/` |
| 稳健岭回归探索性检查 | `outputs/q1_robust/` |
| 第二问Top-3、网络和200次bootstrap | [outputs/q2/README.md](outputs/q2/README.md) |
| 源数据哈希和审计输出 | `outputs/audit/` |
| 图表 | `outputs/figures/` |
| 第三问生成谱与逐项迁移检查 | `outputs/q3/` |
| 优化结果与冻结基线校验 | `outputs/q3_optimized/` |
| 非Pils域验证、失败探索与一致性对照 | `outputs/q3_domain/` |
| beer81审计、稳定预测与接近度保留网格 | `outputs/q3_refined/` |
| 使用声明简稿 | [use_statement.md](docs/use_statement.md) |

## 复现

在仓库根目录运行。原训练环境为Python 3.13.15；依赖版本列于`requirements-lock.txt`。原始赛题文件夹须与`solution_a`并列，目录名和附件名保持不变。

Windows PowerShell安装环境并检查现有结果：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r solution_a/requirements-lock.txt
.\.venv\Scripts\python.exe -X utf8 solution_a/run_all.py
```

macOS/Linux将上述Python路径换为`.venv/bin/python`。默认入口校验现有数据、模型、三层冻结历史及补充分析，报告写入`solution_a/review/verification/`，不重新训练，也不覆盖论文引用的`outputs/`。单独执行`src/05_verify_outputs.py`或`src/25_q3_refined_verify.py`时，默认仅向终端打印；可用`--output 路径`另存报告。

完整重建须在**新解压的源代码副本**中进行。确认其中没有旧`solution_a/outputs/`，放好原始赛题附件后执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 solution_a/run_all.py --rebuild
```

该路线会依次训练各轮模型、生成结果并建立新的冻结快照，运行时间明显长于保存结果校验。若`outputs/`已有文件，程序会拒绝重建；请另建副本，不要删除当前引用结果。`paper_results/`仅供查阅，不能改名为`outputs/`使用。

默认种子为20261023。joblib模型加载前须将`solution_a/src`加入Python模块路径，稳健模型依赖`common.QuantileClipper`。冻结文件按UTF-8、LF和POSIX相对路径记录，哈希保护仍逐字节核对。`src/platform_selfcheck.py`检查Windows路径、三层快照、历史来源路径及文件篡改拒绝。不同Python、NumPy或系统环境可能使重训树模型产生小幅数值差异，保存工件校验和从头重训应分别记录。

## 解释结果时的边界

`oof_*.csv`是折外预测；`fitted_standardized_NOT_validation.csv`是全量拟合值，不能作为测试证据。所有感官数值均为附件4的标准化尺度，当前不能当作原始0–6分。第二问关联不证明因果机制。优化后Kriek/Lambic生成能量差异下降约20.5%/23.4%；Stale hops预测提升尚未得到稳定证实。新轮使用239个非Pils样本的21个完整风格留出选择模型，Pils标签已在此前反复查看，其指标只作诊断。上一轮9/11和6/11属于原检查规则；新轮另报同一规则下的固定旧谱对照与新增筛选结果。原bootstrap委员会为单目标树，旧中心预测器才是45输出；两者不能混称。全部数值检查均不替代真实品评。

原始赛题附件未被更改。提交文件保存在submission/；尚未向竞赛平台上传。签署承诺书需由参赛队自行完成。
