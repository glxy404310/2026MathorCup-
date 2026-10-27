A题论文配套程序与数值结果

1. PDF附录E已印出建模计算、结果绘图及数值检验源程序；本包保留可直接运行的文件及缩进。
2. solution_a/src/为主分析程序，run_all.py为统一入口，supplement/为补充分析，paper/保存结果绘图程序、结构化论文和图表。
3. paper_results/及supplement_results/保存论文引用结果，不是运行输出目录。本包不含原始赛题附件、模型权重及大型候选缓存。
4. 原始赛题目录与solution_a并列，保持官方目录名。计算使用Python 3.13及requirements-lock.txt。
   Windows PowerShell在解压根目录运行：
   py -3.13 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r solution_a/requirements-lock.txt
   .\.venv\Scripts\python.exe -X utf8 solution_a/run_all.py --rebuild
   全部模型、候选缓存、outputs与supplement/results由源码重建。--rebuild拒绝覆盖非空输出目录。
5. 对重建完成的完整目录，去掉--rebuild即可只读验证。报告写入solution_a/review/verification/。
6. 本包中beer81/source_manifest.json的本机绝对路径已转换为仓库相对路径，所指文件的SHA-256保持不变；原始冻结记录和数值结果未改写。
7. MANIFEST.json记录包内文件SHA-256。verification内为原始冻结结果的只读验证和独立重建检查记录。
8. 原评分量尺不能从现有附件唯一恢复，文中标准化预测不冒充0—6评分；化学相关不等于感官因果，迁移结果尚未经实际品评。
